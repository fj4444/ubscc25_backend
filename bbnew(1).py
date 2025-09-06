from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import json
from typing import List, Any

def detect_period_fft(series, sampling_rate=1):
    clean_series = pd.Series(series).fillna(0).values
    
    n = len(clean_series)
    yf = fft(clean_series)
    xf = fftfreq(n, 1/sampling_rate)
    
    positive_freq = xf[:n//2]
    positive_magnitude = 2.0/n * np.abs(yf[:n//2])
    
    peaks, _ = signal.find_peaks(positive_magnitude, height=np.mean(positive_magnitude))
    
    peak_freqs = positive_freq[peaks]
    peak_mags = positive_magnitude[peaks]
    
    sorted_indices = np.argsort(peak_mags)[::-1]
    significant_peaks = []
    
    for idx in sorted_indices:
        freq = peak_freqs[idx]
        if freq > 0:
            period = 1 / freq
            if 2 < period < len(series)/2:
                significant_peaks.append(int(round(period)))
    try:
        return list(dict.fromkeys(significant_peaks))[0]
    except:
        return 1
    return 1

def fill_nan_with_trend_periodic(arr, period=None):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    if not np.isnan(arr).any():
        return arr.copy()

    result = arr.copy()
    nan_mask = np.isnan(arr)
    valid_mask = ~nan_mask
    indices = np.arange(len(arr))
    
    if np.sum(valid_mask) >= 2:
        f_linear = interpolate.interp1d(
            indices[valid_mask], 
            arr[valid_mask], 
            kind='linear', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        linear_filled = f_linear(indices)
        
        if period is not None and 1 < period < len(arr):

            periodic_component = np.zeros_like(arr)
            valid_count = np.zeros_like(arr)

            for i in range(len(arr)):
                periodic_indices = np.arange(i % period, len(arr), period)
                valid_periodic = arr[periodic_indices]
                valid_periodic = valid_periodic[~np.isnan(valid_periodic)]
                
                if len(valid_periodic) > 0:
                    periodic_component[i] = np.mean(valid_periodic)
                    valid_count[i] = len(valid_periodic)
            
            weight = valid_count / np.max(valid_count) if np.max(valid_count) > 0 else 0
            combined = weight * periodic_component + (1 - weight) * linear_filled
        else:
            combined = linear_filled

        result[nan_mask] = combined[nan_mask]
    
    return result

def BlanketyBlanksAlgoTest(json_data, js=True):
    if js:
        series_data = json_data.get('series',[])
    if not js:
        series_data = json_data
    numpy_arrays = []
    for series_list in series_data:
        processed_list = [np.nan if item is None else item for item in series_list]
        numpy_arrays.append(np.array(processed_list, dtype=np.float64))

    result_array = numpy_arrays
    output_array = result_array.copy()
    for i in range(len(result_array)):
        try:
            period = detect_period_fft(result_array[i], sampling_rate=1)
            if period == 1:
                period = None
            output_array[i] = fill_nan_with_trend_periodic(result_array[i], period)
        except:
            111
        output_array[i][np.isnan(output_array[i])] = 0.0
        output_array[i] = list(output_array[i])
    python_list = output_array
    # # 构造 JSON 对象
    result = {"answer": python_list}
    # 转换为 JSON 字符串
    json_output = json.dumps(result, indent=2)
    return json_output