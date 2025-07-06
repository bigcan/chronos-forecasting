"""
Utility functions for log return-based forecasting analysis
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def calculate_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate log returns from price series
    
    Args:
        prices: Price series
        periods: Number of periods for return calculation (default: 1)
        
    Returns:
        Log returns series
    """
    return np.log(prices / prices.shift(periods)).dropna()


def test_stationarity(series: pd.Series) -> dict:
    """
    Test stationarity using ADF and KPSS tests
    
    Args:
        series: Time series to test
        
    Returns:
        Dictionary with test results
    """
    global STATSMODELS_AVAILABLE
    
    if not STATSMODELS_AVAILABLE:
        try:
            print("⚠️  statsmodels not available. Installing...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
            STATSMODELS_AVAILABLE = True
        except Exception as e:
            print(f"❌ Failed to install statsmodels: {e}")
            return {
                'adf_statistic': np.nan,
                'adf_pvalue': np.nan,
                'adf_critical_values': {},
                'adf_is_stationary': False,
                'kpss_statistic': np.nan,
                'kpss_pvalue': np.nan,
                'kpss_critical_values': {},
                'kpss_is_stationary': False
            }
    
    # Import functions when needed
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError as e:
        print(f"❌ Failed to import statsmodels functions: {e}")
        return {
            'adf_statistic': np.nan,
            'adf_pvalue': np.nan,
            'adf_critical_values': {},
            'adf_is_stationary': False,
            'kpss_statistic': np.nan,
            'kpss_pvalue': np.nan,
            'kpss_critical_values': {},
            'kpss_is_stationary': False
        }
    
    # ADF test (null hypothesis: non-stationary)
    adf_result = adfuller(series.dropna())
    
    # KPSS test (null hypothesis: stationary)
    kpss_result = kpss(series.dropna(), regression='c')
    
    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_critical_values': adf_result[4],
        'adf_is_stationary': adf_result[1] < 0.05,
        'kpss_statistic': kpss_result[0],
        'kpss_pvalue': kpss_result[1],
        'kpss_critical_values': kpss_result[3],
        'kpss_is_stationary': kpss_result[1] > 0.05
    }


def reconstruct_prices(initial_price: float, log_returns: pd.Series) -> pd.Series:
    """
    Reconstruct price series from log returns
    
    Args:
        initial_price: Starting price
        log_returns: Series of log returns
        
    Returns:
        Reconstructed price series
    """
    cumulative_returns = log_returns.cumsum()
    prices = initial_price * np.exp(cumulative_returns)
    return prices


def calculate_return_metrics(actual_returns: pd.Series, predicted_returns: pd.Series) -> dict:
    """
    Calculate return-specific performance metrics
    
    Args:
        actual_returns: Actual log returns
        predicted_returns: Predicted log returns
        
    Returns:
        Dictionary with metrics
    """
    # Basic error metrics
    mae = np.mean(np.abs(actual_returns - predicted_returns))
    mse = np.mean((actual_returns - predicted_returns) ** 2)
    rmse = np.sqrt(mse)
    
    # Directional accuracy
    direction_actual = np.sign(actual_returns)
    direction_predicted = np.sign(predicted_returns)
    hit_rate = np.mean(direction_actual == direction_predicted)
    
    # Volatility prediction accuracy
    vol_actual = actual_returns.std()
    vol_predicted = predicted_returns.std()
    vol_ratio = vol_predicted / vol_actual if vol_actual != 0 else np.nan
    
    # Sharpe ratio comparison (assuming risk-free rate = 0)
    sharpe_actual = actual_returns.mean() / actual_returns.std() if actual_returns.std() != 0 else np.nan
    sharpe_predicted = predicted_returns.mean() / predicted_returns.std() if predicted_returns.std() != 0 else np.nan
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'hit_rate': hit_rate,
        'volatility_ratio': vol_ratio,
        'sharpe_actual': sharpe_actual,
        'sharpe_predicted': sharpe_predicted,
        'mean_actual': actual_returns.mean(),
        'mean_predicted': predicted_returns.mean(),
        'std_actual': actual_returns.std(),
        'std_predicted': predicted_returns.std()
    }


def prepare_returns_for_chronos(returns: pd.Series, 
                               context_window: int = 126,
                               prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare log returns data for Chronos model training/inference
    
    Args:
        returns: Log returns series
        context_window: Number of historical returns to use as context
        prediction_horizon: Number of future returns to predict
        
    Returns:
        Tuple of (context_data, target_data)
    """
    context_data = []
    target_data = []
    
    for i in range(context_window, len(returns) - prediction_horizon + 1):
        context = returns.iloc[i-context_window:i].values
        target = returns.iloc[i:i+prediction_horizon].values
        
        context_data.append(context)
        target_data.append(target)
    
    return np.array(context_data), np.array(target_data)


def analyze_return_distribution(returns: pd.Series) -> dict:
    """
    Analyze statistical properties of return distribution
    
    Args:
        returns: Log returns series
        
    Returns:
        Dictionary with distribution statistics
    """
    # Basic statistics
    mean = returns.mean()
    std = returns.std()
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # Normality test
    jarque_bera = stats.jarque_bera(returns)
    
    # Percentiles
    percentiles = np.percentile(returns, [1, 5, 25, 50, 75, 95, 99])
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jarque_bera_statistic': jarque_bera[0],
        'jarque_bera_pvalue': jarque_bera[1],
        'is_normal': jarque_bera[1] > 0.05,
        'percentiles': {
            '1%': percentiles[0],
            '5%': percentiles[1],
            '25%': percentiles[2],
            '50%': percentiles[3],
            '75%': percentiles[4],
            '95%': percentiles[5],
            '99%': percentiles[6]
        }
    }