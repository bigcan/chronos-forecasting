#!/usr/bin/env python3
"""
2016-2019 Gold Futures Analysis with Chronos
Comparative analysis with 2020-2021 period
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    print("⚠️ Seaborn not available, using matplotlib only")
    sns = None

from datetime import datetime, timedelta
import json
from pathlib import Path

# Statistical analysis
try:
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Scipy/sklearn not available, using basic metrics")
    SKLEARN_AVAILABLE = False

# Chronos imports
try:
    import torch
    from chronos import BaseChronosPipeline
    CHRONOS_AVAILABLE = True
    print("✅ Chronos imports successful")
except ImportError as e:
    print(f"❌ Chronos not available: {e}")
    CHRONOS_AVAILABLE = False

print("🚀 Starting 2016-2019 Gold Futures Analysis")
print("="*70)

# 1. Load and preprocess data
def load_data_2016_2019():
    """Load and preprocess 2016-2019 gold futures data"""
    try:
        df = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
        print("✅ Data loaded successfully")
    except FileNotFoundError:
        print("❌ Error: GCUSD_MAX_FROM_PERPLEXITY.csv not found")
        return None
    
    # Convert dates and filter
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Filter for 2016-2019
    mask = (df['Date'] >= '2016-01-01') & (df['Date'] <= '2019-12-31')
    data = df[mask].reset_index(drop=True)
    
    print(f"✅ 2016-2019 data loaded: {len(data)} trading days")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    return data

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
        'avg_volume': data['Volume'].mean()
    }
    
    return analysis

# 3. Baseline models
def calculate_baseline_predictions(data, method='naive'):
    """Calculate baseline model predictions"""
    if method == 'naive':
        # Use previous day's close as prediction
        return data['Close'].iloc[:-1].values
    elif method == 'ma':
        # 5-day moving average
        return data['Close'].rolling(window=5).mean().iloc[:-1].values
    elif method == 'seasonal_naive':
        # Use same day of week from previous week
        return data['Close'].shift(5).iloc[:-1].values
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
    
    # Calculate metrics
    if SKLEARN_AVAILABLE:
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
    else:
        # Calculate manually
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        mse = np.mean((y_true_clean - y_pred_clean) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    # MASE (Mean Absolute Scaled Error)
    if SKLEARN_AVAILABLE:
        naive_mae = mean_absolute_error(y_true_clean[1:], y_true_clean[:-1])
    else:
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

# 5. Chronos evaluation
def evaluate_chronos_2016_2019(data, context_length=63):
    """Evaluate Chronos model on 2016-2019 data"""
    if not CHRONOS_AVAILABLE:
        print("❌ Chronos not available, skipping model evaluation")
        return None
    
    try:
        # Load the model
        print("Loading Chronos-Bolt-Base model...")
        pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base")
        print("✅ Chronos model loaded successfully")
        
        # Prepare evaluation data
        predictions = []
        actuals = []
        
        print(f"Running evaluation with {context_length}-day context window...")
        
        # Rolling window evaluation
        for i in range(context_length, len(data) - 1):
            # Get context window
            context = data['Close'].iloc[i-context_length:i].values
            actual = data['Close'].iloc[i]
            
            # Make prediction
            try:
                forecast = pipeline.predict(
                    context=torch.tensor(context, dtype=torch.float32).unsqueeze(0),
                    prediction_length=1
                )
                pred = forecast[0].median.item()
                predictions.append(pred)
                actuals.append(actual)
            except Exception as e:
                print(f"⚠️ Prediction error at index {i}: {e}")
                continue
            
            # Progress update
            if (i - context_length) % 100 == 0:
                progress = (i - context_length) / (len(data) - context_length - 1) * 100
                print(f"Progress: {progress:.1f}%")
        
        print(f"✅ Chronos evaluation completed: {len(predictions)} predictions")
        return np.array(predictions), np.array(actuals)
    
    except Exception as e:
        print(f"❌ Chronos evaluation failed: {e}")
        return None

# 6. Main analysis
def main():
    """Run the complete 2016-2019 analysis"""
    
    # Load data
    data_2016_2019 = load_data_2016_2019()
    if data_2016_2019 is None:
        return
    
    # Analyze market regime
    print("\n📊 MARKET REGIME ANALYSIS")
    print("-" * 50)
    regime_2016_2019 = analyze_market_regime(data_2016_2019, "2016-2019")
    
    # Compare with 2020-2021 if available
    try:
        df_full = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
        df_full['Date'] = pd.to_datetime(df_full['Date'])
        mask_2020_2021 = (df_full['Date'] >= '2020-01-01') & (df_full['Date'] <= '2021-12-31')
        data_2020_2021 = df_full[mask_2020_2021].reset_index(drop=True)
        regime_2020_2021 = analyze_market_regime(data_2020_2021, "2020-2021")
        
        print(f"2016-2019 vs 2020-2021 Comparison:")
        print(f"Volatility: {regime_2016_2019['volatility']:.2f}% vs {regime_2020_2021['volatility']:.2f}%")
        print(f"Total Return: {regime_2016_2019['total_return']:.1f}% vs {regime_2020_2021['total_return']:.1f}%")
        print(f"Max Drawdown: {regime_2016_2019['max_drawdown']:.1f}% vs {regime_2020_2021['max_drawdown']:.1f}%")
        
    except Exception as e:
        print(f"⚠️ Could not load 2020-2021 data for comparison: {e}")
    
    # Run baseline evaluations
    print("\n📈 BASELINE MODEL EVALUATION")
    print("-" * 50)
    
    # Prepare targets (next day prices)
    targets = data_2016_2019['Close'].iloc[1:].values
    
    # Calculate baseline predictions
    baselines = {
        'Naive': calculate_baseline_predictions(data_2016_2019, 'naive'),
        'Moving_Average': calculate_baseline_predictions(data_2016_2019, 'ma'),
        'Seasonal_Naive': calculate_baseline_predictions(data_2016_2019, 'seasonal_naive')
    }
    
    # Calculate metrics for baselines
    results = []
    for model_name, predictions in baselines.items():
        # Remove NaN values
        mask = ~np.isnan(predictions)
        if np.sum(mask) > 0:
            metrics = calculate_metrics(targets[mask], predictions[mask], model_name)
            results.append(metrics)
            print(f"✅ {model_name}: MASE={metrics['MASE']:.4f}, MAE=${metrics['MAE']:.2f}")
    
    # Run Chronos evaluation
    print("\n🤖 CHRONOS MODEL EVALUATION")
    print("-" * 50)
    
    if CHRONOS_AVAILABLE:
        chronos_result = evaluate_chronos_2016_2019(data_2016_2019)
        if chronos_result is not None:
            chronos_preds, chronos_actuals = chronos_result
            chronos_metrics = calculate_metrics(chronos_actuals, chronos_preds, 'Chronos')
            results.append(chronos_metrics)
            print(f"✅ Chronos: MASE={chronos_metrics['MASE']:.4f}, MAE=${chronos_metrics['MAE']:.2f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.set_index('Model', inplace=True)
    results_df.to_csv('gold_futures_forecast_2016_2019_metrics.csv')
    
    print("\n📊 FINAL RESULTS 2016-2019")
    print("-" * 50)
    print(results_df.round(4))
    
    # Compare with 2020-2021 if available
    try:
        results_2020_2021 = pd.read_csv('gold_futures_forecast_metrics.csv', index_col=0)
        print("\n📈 COMPARATIVE ANALYSIS")
        print("-" * 50)
        
        # Compare key models
        models_to_compare = ['Naive', 'Chronos']
        for model in models_to_compare:
            if model in results_df.index and model in results_2020_2021.index:
                mase_2016 = results_df.loc[model, 'MASE']
                mase_2021 = results_2020_2021.loc[model, 'MASE']
                improvement = ((mase_2021 - mase_2016) / mase_2016) * 100
                print(f"{model}: 2016-2019 MASE={mase_2016:.4f}, 2020-2021 MASE={mase_2021:.4f}")
                print(f"  → {'Worse' if improvement > 0 else 'Better'} performance in 2020-2021 ({improvement:+.1f}%)")
        
        # Calculate gaps
        if 'Naive' in results_df.index and 'Chronos' in results_df.index:
            naive_2016 = results_df.loc['Naive', 'MASE']
            chronos_2016 = results_df.loc['Chronos', 'MASE']
            gap_2016 = ((chronos_2016 - naive_2016) / naive_2016) * 100
            
            naive_2021 = results_2020_2021.loc['Naive', 'MASE']
            chronos_2021 = results_2020_2021.loc['Chronos', 'MASE']
            gap_2021 = ((chronos_2021 - naive_2021) / naive_2021) * 100
            
            print(f"\nChronos vs Naive Gap:")
            print(f"2016-2019: {gap_2016:+.1f}%")
            print(f"2020-2021: {gap_2021:+.1f}%")
            print(f"Market regime effect: {gap_2021 - gap_2016:+.1f} percentage points")
        
    except Exception as e:
        print(f"⚠️ Could not compare with 2020-2021 results: {e}")
    
    print("\n✅ 2016-2019 analysis completed successfully!")
    return results_df

if __name__ == "__main__":
    main()