from scipy import signal
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
    clean_series = pd.Series(series).dropna().values
    
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

def polynomial_trend_fit(X, y, max_degree=5, r2_threshold=0.7, mse_threshold=None, plot=False):
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    if len(X_valid) < 10: 
        return None, np.full(len(X), np.nan), 0
    
    best_model = None
    best_predictions = None
    best_score = -float('inf')
    best_degree = 0

    for degree in range(1, max_degree + 1):
        try:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X_valid.reshape(-1, 1))
            
            model = LinearRegression()
            model.fit(X_poly, y_valid)
            
            predictions = model.predict(poly.transform(X.reshape(-1, 1)))
            r2 = r2_score(y_valid, predictions[valid_mask])
            mse = mean_squared_error(y_valid, predictions[valid_mask])
            
            if r2 < r2_threshold:
                continue
                
            if mse_threshold and mse > mse_threshold:
                continue
            
            if r2 > best_score:
                best_score = r2
                best_model = (model, poly, degree)
                best_predictions = predictions
                best_degree = degree
                
            
        except Exception as e:
            continue
    
    if best_model is None:
        ma_predictions = pd.Series(y).rolling(window=min(20, len(y)//10), 
                                            min_periods=1, center=True).mean().values
        return None, ma_predictions, 0
    
    model, poly, degree = best_model
    
    return best_model, best_predictions, best_score

def robust_stl_interpolation_with_polynomial(y, period=None, max_poly_degree=2):
    y_init = pd.Series(y).interpolate(method='linear').values
    X = np.arange(len(y))
    
    try:
        stl = STL(pd.Series(y_init), period=period if period else 50)
        res = stl.fit()
        T, S, R = res.trend, res.seasonal, res.resid
    except:
        return fallback_interpolation(y)
    
    trend_model, T_filled, trend_score = polynomial_trend_fit(
        X, T, max_degree=max_poly_degree, r2_threshold=0.75, plot=True
    )

    if period:
        S_filled = seasonal_interpolation(S, period, np.isnan(y))
    else:
        S_filled = np.zeros_like(S)

    R_filled = residual_interpolation(R, np.isnan(y))
    
    y_filled = T_filled + S_filled + R_filled

    result = y.copy()
    result[np.isnan(y)] = y_filled[np.isnan(y)]
    
    return result

def seasonal_interpolation(S, period, nan_mask):
    S_filled = S.copy()
    for i in range(len(S)):
        if nan_mask[i]:
            pos_in_cycle = i % period
            known_indices = np.arange(pos_in_cycle, len(S), period)
            known_values = S[known_indices]
            known_values = known_values[~np.isnan(known_values)]
            
            if len(known_values) > 0:
                S_filled[i] = np.mean(known_values)
            else:
                S_filled[i] = np.nanmean(S[max(0, i-1):min(len(S), i+2)])

    S_filled = pd.Series(S_filled).interpolate(method='linear').values
    return S_filled

def residual_interpolation(R, nan_mask):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        known_R = R[~nan_mask]
        if len(known_R) > 10:
            model = ARIMA(known_R, order=(2, 0, 0))
            fit_model = model.fit()
            R_pred = fit_model.predict(start=0, end=len(R)-1)
            return R_pred.values
    except:
        111
    
    return pd.Series(R).rolling(window=5, min_periods=1, center=True).mean().values

def fallback_interpolation(y):
    result = y
    try:
        result = pd.Series(y).interpolate(method='spline', order=3).values
    except:
        try:
            result = pd.Series(y).interpolate(method='polynomial', order=2).values
        except:
            result = pd.Series(y).interpolate(method='linear').values
    
    return result

def BlanketyBlanksAlgoTest(json_data):
    try:
        with open(json_data, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, OSError):
        data = json.loads(json_data)
    series_data = data['series']
    numpy_arrays = []
    for series_list in series_data:
        processed_list = [np.nan if item is None else item for item in series_list]
        numpy_arrays.append(np.array(processed_list, dtype=np.float64))
    result_array = np.array(numpy_arrays)
    output_array = result_array.copy()
    for i in range(len(result_array)):
        try:
            output_array[i, :] = robust_stl_interpolation_with_polynomial(result_array[i, :])
        except:
            111
    output_array[np.isnan(output_array)] = 0.0
    return output_array