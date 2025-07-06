#!/usr/bin/env python3
"""
2016-2019 Gold Futures Baseline Analysis
Focus on baseline models and market regime comparison
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta

print("🚀 Starting 2016-2019 Gold Futures Baseline Analysis")
print("="*70)

# 1. Load and preprocess data
def load_data_periods():
    """Load and preprocess data for both periods"""
    try:
        df = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
        print("✅ Data loaded successfully")
    except FileNotFoundError:
        print("❌ Error: GCUSD_MAX_FROM_PERPLEXITY.csv not found")
        return None, None
    
    # Convert dates and filter
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Filter for both periods
    mask_2016_2019 = (df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')
    mask_2020_2021 = (df['Date'] >= '2020-01-01') & (df['Date'] <= '2021-12-31')
    
    data_2016_2019 = df[mask_2016_2019].reset_index(drop=True)
    data_2020_2021 = df[mask_2020_2021].reset_index(drop=True)
    
    print(f"✅ 2016-2019 data: {len(data_2016_2019)} trading days")
    print(f"✅ 2020-2021 data: {len(data_2020_2021)} trading days")
    
    return data_2016_2019, data_2020_2021

# 2. Market regime analysis
def analyze_market_regime(data, period_name):
    """Analyze market characteristics"""
    returns = data['Close'].pct_change().dropna()
    
    analysis = {
        'period': period_name,
        'trading_days': len(data),
        'price_start': data['Close'].iloc[0],
        'price_end': data['Close'].iloc[-1],
        'total_return': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100,
        'volatility': returns.std() * np.sqrt(252) * 100,
        'daily_volatility': returns.std() * 100,
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'max_drawdown': ((data['Close'] / data['Close'].cummax()) - 1).min() * 100,
        'positive_days': (returns > 0).mean() * 100,
        'large_moves': (np.abs(returns) > 0.02).mean() * 100,
        'price_range': data['Close'].max() - data['Close'].min(),
        'avg_volume': data['Volume'].mean(),
        'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    }
    
    return analysis

# 3. Baseline models
def calculate_baseline_predictions(data, method='naive'):
    """Calculate baseline model predictions"""
    if method == 'naive':
        return data['Close'].iloc[:-1].values
    elif method == 'ma':
        return data['Close'].rolling(window=5).mean().iloc[:-1].values
    elif method == 'seasonal_naive':
        return data['Close'].shift(5).iloc[:-1].values
    elif method == 'linear_trend':
        # Simple linear trend
        x = np.arange(len(data))
        y = data['Close'].values
        slope, intercept = np.polyfit(x, y, 1)
        trend = slope * x + intercept
        return trend[:-1]
    else:
        return np.full(len(data)-1, data['Close'].mean())

# 4. Calculate metrics
def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive forecasting metrics"""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    # Calculate metrics manually
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    mse = np.mean((y_true_clean - y_pred_clean) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    # MASE (Mean Absolute Scaled Error)
    naive_mae = np.mean(np.abs(y_true_clean[1:] - y_true_clean[:-1]))
    mase = mae / naive_mae if naive_mae > 0 else np.inf
    
    # Directional accuracy
    y_true_diff = np.diff(y_true_clean)
    y_pred_diff = np.diff(y_pred_clean)
    dir_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
    
    # Bias
    bias = np.mean(y_pred_clean - y_true_clean)
    
    # R-squared
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'Model': model_name,
        'MASE': mase,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Directional_Accuracy': dir_accuracy,
        'Bias': bias,
        'R_Squared': r_squared,
        'samples': len(y_true_clean)
    }

# 5. Main analysis
def main():
    """Run the complete baseline analysis"""
    
    # Load data
    data_2016_2019, data_2020_2021 = load_data_periods()
    if data_2016_2019 is None or data_2020_2021 is None:
        return
    
    # Analyze market regimes
    print("\n📊 MARKET REGIME ANALYSIS")
    print("-" * 50)
    
    regime_2016_2019 = analyze_market_regime(data_2016_2019, "2016-2019")
    regime_2020_2021 = analyze_market_regime(data_2020_2021, "2020-2021")
    
    print(f"📈 2016-2019 Period:")
    print(f"  Price: ${regime_2016_2019['price_start']:.2f} → ${regime_2016_2019['price_end']:.2f} ({regime_2016_2019['total_return']:.1f}%)")
    print(f"  Volatility: {regime_2016_2019['volatility']:.2f}% (annualized)")
    print(f"  Max Drawdown: {regime_2016_2019['max_drawdown']:.1f}%")
    print(f"  Sharpe Ratio: {regime_2016_2019['sharpe_ratio']:.2f}")
    
    print(f"\n📉 2020-2021 Period:")
    print(f"  Price: ${regime_2020_2021['price_start']:.2f} → ${regime_2020_2021['price_end']:.2f} ({regime_2020_2021['total_return']:.1f}%)")
    print(f"  Volatility: {regime_2020_2021['volatility']:.2f}% (annualized)")
    print(f"  Max Drawdown: {regime_2020_2021['max_drawdown']:.1f}%")
    print(f"  Sharpe Ratio: {regime_2020_2021['sharpe_ratio']:.2f}")
    
    # Calculate performance differences
    vol_ratio = regime_2020_2021['volatility'] / regime_2016_2019['volatility']
    return_diff = regime_2020_2021['total_return'] - regime_2016_2019['total_return']
    
    print(f"\n🔍 KEY DIFFERENCES:")
    print(f"  Volatility: 2020-2021 was {vol_ratio:.1f}x more volatile")
    print(f"  Return difference: {return_diff:.1f} percentage points")
    
    # Hypothesis for forecasting
    print(f"\n💡 FORECASTING HYPOTHESIS:")
    if regime_2016_2019['volatility'] < regime_2020_2021['volatility']:
        print(f"  ✅ Lower volatility in 2016-2019 should favor sophisticated models")
        print(f"  ✅ More stable patterns may be easier to predict")
    else:
        print(f"  ⚠️ Higher volatility in 2016-2019 may favor naive methods")
    
    # Run baseline evaluations for both periods
    results_2016_2019 = run_baseline_evaluation(data_2016_2019, "2016-2019")
    results_2020_2021 = run_baseline_evaluation(data_2020_2021, "2020-2021")
    
    # Compare results
    print("\n📊 COMPARATIVE RESULTS")
    print("-" * 70)
    
    comparison_data = []
    models = ['Naive', 'Moving_Average', 'Seasonal_Naive', 'Linear_Trend']
    
    for model in models:
        if model in results_2016_2019 and model in results_2020_2021:
            mase_2016 = results_2016_2019[model]['MASE']
            mase_2021 = results_2020_2021[model]['MASE']
            improvement = ((mase_2021 - mase_2016) / mase_2016) * 100
            
            dir_acc_2016 = results_2016_2019[model]['Directional_Accuracy']
            dir_acc_2021 = results_2020_2021[model]['Directional_Accuracy']
            
            comparison_data.append({
                'Model': model,
                'MASE_2016_2019': mase_2016,
                'MASE_2020_2021': mase_2021,
                'MASE_Change': improvement,
                'Dir_Acc_2016_2019': dir_acc_2016,
                'Dir_Acc_2020_2021': dir_acc_2021
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4))
    
    # Save results
    results_2016_2019_df = pd.DataFrame(results_2016_2019).T
    results_2020_2021_df = pd.DataFrame(results_2020_2021).T
    
    results_2016_2019_df.to_csv('baseline_2016_2019_results.csv')
    results_2020_2021_df.to_csv('baseline_2020_2021_results.csv')
    comparison_df.to_csv('baseline_comparison_2016_2019_vs_2020_2021.csv', index=False)
    
    # Key insights
    print(f"\n🎯 KEY INSIGHTS:")
    naive_change = comparison_df[comparison_df['Model'] == 'Naive']['MASE_Change'].iloc[0]
    print(f"1. Naive baseline: {naive_change:+.1f}% change from 2016-2019 to 2020-2021")
    
    if naive_change > 0:
        print(f"   → Naive performed WORSE in 2020-2021 (higher volatility period)")
    else:
        print(f"   → Naive performed BETTER in 2020-2021")
    
    # Market regime effect
    print(f"\n📈 MARKET REGIME IMPACT:")
    print(f"Lower volatility period (2016-2019): More predictable patterns")
    print(f"Higher volatility period (2020-2021): More challenging for forecasting")
    
    # Prepare for Chronos comparison
    print(f"\n📋 CHRONOS COMPARISON READY:")
    print(f"Use these baseline results to evaluate Chronos performance:")
    print(f"- If Chronos beats naive in 2016-2019 but not 2020-2021:")
    print(f"  → Supports hypothesis that sophisticated models work better in stable periods")
    print(f"- If Chronos consistently trails naive in both periods:")
    print(f"  → Suggests gold futures are inherently difficult to predict")
    
    return results_2016_2019_df, results_2020_2021_df, comparison_df

def run_baseline_evaluation(data, period_name):
    """Run baseline model evaluation for a specific period"""
    print(f"\n📈 BASELINE EVALUATION - {period_name}")
    print("-" * 40)
    
    # Prepare targets
    targets = data['Close'].iloc[1:].values
    
    # Calculate baseline predictions
    baselines = {
        'Naive': calculate_baseline_predictions(data, 'naive'),
        'Moving_Average': calculate_baseline_predictions(data, 'ma'),
        'Seasonal_Naive': calculate_baseline_predictions(data, 'seasonal_naive'),
        'Linear_Trend': calculate_baseline_predictions(data, 'linear_trend')
    }
    
    # Calculate metrics
    results = {}
    for model_name, predictions in baselines.items():
        # Remove NaN values
        mask = ~np.isnan(predictions)
        if np.sum(mask) > 0:
            metrics = calculate_metrics(targets[mask], predictions[mask], model_name)
            results[model_name] = metrics
            print(f"✅ {model_name}: MASE={metrics['MASE']:.4f}, MAE=${metrics['MAE']:.2f}, Dir.Acc={metrics['Directional_Accuracy']:.1f}%")
    
    return results

if __name__ == "__main__":
    main()