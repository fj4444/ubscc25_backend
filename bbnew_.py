import numpy as np
from scipy import interpolate
from pykalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')

def kalman_filter_imputation(arr):
    """使用卡尔曼滤波进行时间序列缺失值填充"""
    # 获取有效数据点
    observed_mask = ~np.isnan(arr)
    observed_data = arr.copy()
    observed_data[~observed_mask] = 0  # 卡尔曼滤波需要将缺失值设为0
    
    # 设置卡尔曼滤波参数
    try:
        # 尝试自动估计初始参数
        valid_data = arr[observed_mask]
        if len(valid_data) < 2:
            return arr  # 数据不足，返回原数组
        
        # 初始化卡尔曼滤波
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=np.mean(valid_data),
            initial_state_covariance=1,
            observation_covariance=np.var(valid_data) * 0.1 if len(valid_data) > 1 else 1,
            transition_covariance=np.var(valid_data) * 0.01 if len(valid_data) > 1 else 0.1
        )
        
        # 使用EM算法学习参数（如果数据量足够）
        if len(valid_data) > 10:
            kf = kf.em(observed_data, n_iter=10)
        
        # 进行卡尔曼平滑
        state_means, state_covariances = kf.smooth(observed_data)
        
        # 用平滑后的状态估计填充缺失值
        arr_filled = arr.copy()
        arr_filled[~observed_mask] = state_means[~observed_mask, 0]
        
        return arr_filled
        
    except Exception as e:
        # 如果卡尔曼滤波失败，退回线性插值
        return linear_interpolation(arr)

def linear_interpolation(arr):
    """线性插值作为备选方法"""
    nan_mask = np.isnan(arr)
    if not np.any(nan_mask):
        return arr
    
    x = np.arange(len(arr))
    valid_mask = ~nan_mask
    f = interpolate.interp1d(x[valid_mask], arr[valid_mask], 
                           kind='linear', bounds_error=False, 
                           fill_value="extrapolate")
    return f(x)

def adaptive_imputation(series_array):
    """
    自适应缺失值填充函数，结合卡尔曼滤波和线性插值
    """
    filled_series = []
    
    for i, series in enumerate(series_array):
        # 转换为numpy数组
        arr = np.array([x if x is not None else np.nan for x in series], dtype=float)
        nan_mask = np.isnan(arr)
        
        # 如果没有缺失值，直接返回
        if not np.any(nan_mask):
            filled_series.append(arr.tolist())
            continue
        
        # 根据缺失模式选择方法
        missing_ratio = np.mean(nan_mask)
        valid_data = arr[~nan_mask]
        
        if len(valid_data) < 5:
            # 数据太少，使用简单插值
            arr_filled = linear_interpolation(arr)
        elif missing_ratio > 0.5:
            # 缺失值太多，卡尔曼滤波可能不稳定，使用线性插值
            arr_filled = linear_interpolation(arr)
        else:
            # 使用卡尔曼滤波进行状态空间建模
            try:
                arr_filled = kalman_filter_imputation(arr)
            except:
                # 如果卡尔曼滤波失败，退回线性插值
                arr_filled = linear_interpolation(arr)
        
        # 确保填充值在合理范围内
        if len(valid_data) > 0:
            data_min, data_max = np.min(valid_data), np.max(valid_data)
            data_range = data_max - data_min
            arr_filled = np.clip(arr_filled, 
                               data_min - 0.2 * data_range,
                               data_max + 0.2 * data_range)
        
        filled_series.append(arr_filled.tolist())
    
    return filled_series

# 端点函数
def blankety_endpoint(series_data):
    """
    POST端点处理函数
    """
    filled_data = adaptive_imputation(series_data)
    return {"answer": filled_data}

def BlanketyBlanksAlgoTest(json_data):
    out = list()
    for series_list in json_data["series"]:
        processed_list = [np.nan if item is None else item for item in series_list]
        out.append(list(kalman_filter_imputation(processed_list)))
    result = {"answer": out}
    json_output = json.dumps(result, indent=2)
    return json_output