#!/usr/bin/env python3
"""
Simplified Phase 1 Configuration Optimization
Focus on core optimization without heavy dependencies
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import time

print("🚀 Starting Simplified Phase 1: Chronos Configuration Optimization")
print("="*60)

def load_and_preprocess_data():
    """Load and preprocess gold futures data"""
    try:
        df = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
        print("✅ Data loaded successfully")
    except FileNotFoundError:
        print("❌ Error: GCUSD_MAX_FROM_PERPLEXITY.csv not found")
        print("Current directory contents:")
        try:
            files = os.listdir('.')
            for f in files[:10]:  # Show first 10 files
                print(f"  - {f}")
        except Exception as e:
            print(f"Error listing files: {e}")
        return None
    
    # Preprocess
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Filter for 2020-2021 data
    mask = (data['Date'] >= '2020-01-01') & (data['Date'] <= '2021-12-31')
    data = data[mask].reset_index(drop=True)
    
    # Handle missing values
    data = data.fillna(method='ffill')
    
    # Create target variable
    data['Target'] = data['Close'].shift(-1)
    data = data[:-1].reset_index(drop=True)
    
    print(f"Preprocessed dataset shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    return data

def create_test_dataset(data, window_size, max_samples=50):
    """Create test dataset for evaluation"""
    records = []
    
    for i in range(window_size, len(data)):
        if len(records) >= max_samples:
            break
            
        historical_data = data.iloc[i-window_size:i]['Close'].values
        target = data.iloc[i]['Close']
        
        records.append({
            'context': historical_data,
            'target': target,
            'date': data.iloc[i]['Date']
        })
    
    return records

def naive_forecast(context):
    """Simple naive forecast - return last value"""
    return context[-1]

def moving_average_forecast(context, window=5):
    """Moving average forecast"""
    if len(context) < window:
        window = len(context)
    return np.mean(context[-window:])

def linear_trend_forecast(context):
    """Linear trend forecast"""
    if len(context) < 2:
        return context[-1]
    
    x = np.arange(len(context))
    slope, intercept = np.polyfit(x, context, 1)
    return slope * len(context) + intercept

def calculate_metrics(predictions, actuals):
    """Calculate evaluation metrics"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    
    # MASE
    if len(actuals) > 1:
        naive_errors = np.abs(actuals[1:] - actuals[:-1])
        naive_mae = np.mean(naive_errors)
        mase = mae / naive_mae if naive_mae > 0 else np.inf
    else:
        mase = np.inf
    
    # Directional accuracy
    if len(actuals) > 1:
        actual_direction = np.sign(np.diff(actuals))
        pred_direction = np.sign(predictions[1:] - actuals[:-1])
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        directional_accuracy = 50.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase,
        'Directional_Accuracy': directional_accuracy,
        'n_predictions': len(predictions)
    }

def test_context_windows(data):
    """Test different context window sizes"""
    print("\n🔍 Testing Context Window Optimization")
    print("=" * 50)
    
    window_sizes = [30, 63, 126, 252]
    results = {}
    
    for window_size in window_sizes:
        print(f"\nTesting window size: {window_size} days")
        
        # Create test dataset
        test_data = create_test_dataset(data, window_size, max_samples=50)
        
        if len(test_data) == 0:
            print(f"⚠️ No data available for window size {window_size}")
            continue
        
        # Test with different forecasting methods
        forecasting_methods = {
            'Naive': naive_forecast,
            'MovingAvg': moving_average_forecast,
            'LinearTrend': linear_trend_forecast
        }
        
        window_results = {}
        
        for method_name, method_func in forecasting_methods.items():
            predictions = []
            actuals = []
            
            for sample in test_data:
                context = sample['context']
                actual = sample['target']
                
                try:
                    if method_name == 'MovingAvg':
                        pred = method_func(context, window=min(5, len(context)//4))
                    else:
                        pred = method_func(context)
                    
                    predictions.append(pred)
                    actuals.append(actual)
                except Exception as e:
                    # Fallback to naive
                    predictions.append(context[-1])
                    actuals.append(actual)
            
            metrics = calculate_metrics(predictions, actuals)
            window_results[method_name] = metrics
            
            print(f"  {method_name:12s}: MASE = {metrics['MASE']:.4f}, Dir.Acc = {metrics['Directional_Accuracy']:5.1f}%")
        
        results[window_size] = window_results
    
    return results

def analyze_results(results):
    """Analyze and summarize results"""
    print("\n" + "="*80)
    print("CONTEXT WINDOW OPTIMIZATION RESULTS")
    print("="*80)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Find best configuration for each method
    methods = ['Naive', 'MovingAvg', 'LinearTrend']
    
    for method in methods:
        print(f"\n📊 {method} Performance by Window Size:")
        print("-" * 50)
        
        method_results = []
        for window_size, window_results in results.items():
            if method in window_results:
                metrics = window_results[method]
                method_results.append((window_size, metrics['MASE'], metrics['Directional_Accuracy']))
                print(f"Window {window_size:3d}: MASE = {metrics['MASE']:.4f}, Dir.Acc = {metrics['Directional_Accuracy']:5.1f}%")
        
        if method_results:
            # Find best window for this method
            best_window = min(method_results, key=lambda x: x[1])
            print(f"🏆 Best window for {method}: {best_window[0]} days (MASE: {best_window[1]:.4f})")
    
    # Overall best configuration
    print(f"\n🏆 OVERALL BEST CONFIGURATIONS:")
    print("-" * 60)
    
    all_configs = []
    for window_size, window_results in results.items():
        for method, metrics in window_results.items():
            all_configs.append((f"{method}-{window_size}d", metrics['MASE'], metrics['Directional_Accuracy']))
    
    # Sort by MASE
    all_configs.sort(key=lambda x: x[1])
    
    for i, (config, mase, dir_acc) in enumerate(all_configs[:5], 1):
        print(f"{i}. {config:20s}: MASE = {mase:.4f}, Dir.Acc = {dir_acc:5.1f}%")
    
    if all_configs:
        best_config = all_configs[0]
        print(f"\n🎯 RECOMMENDATION:")
        print(f"Best Configuration: {best_config[0]}")
        print(f"MASE: {best_config[1]:.4f}")
        print(f"Directional Accuracy: {best_config[2]:.1f}%")
        
        # Compare with known baseline
        naive_baseline_mase = 1.0054  # From previous analysis
        improvement = (naive_baseline_mase - best_config[1]) / naive_baseline_mase * 100
        
        print(f"\nComparison with Naive Baseline:")
        print(f"Baseline MASE: {naive_baseline_mase:.4f}")
        print(f"Best Config MASE: {best_config[1]:.4f}")
        
        if best_config[1] < naive_baseline_mase:
            print(f"✅ Improvement: {improvement:.1f}%")
        else:
            print(f"❌ Performance gap: {-improvement:.1f}%")

def save_results(results):
    """Save results to CSV file"""
    try:
        # Flatten results for CSV export
        flattened_data = []
        
        for window_size, window_results in results.items():
            for method, metrics in window_results.items():
                row = {
                    'Window_Size': window_size,
                    'Method': method,
                    'Configuration': f"{method}-{window_size}d",
                    **metrics
                }
                flattened_data.append(row)
        
        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_csv('phase1_optimization_results.csv', index=False)
            print(f"\n📁 Results saved to: phase1_optimization_results.csv")
            print(f"   Total configurations tested: {len(flattened_data)}")
        
    except Exception as e:
        print(f"⚠️ Error saving results: {e}")

def main():
    """Main execution function"""
    print("Step 1: Loading and preprocessing data...")
    data = load_and_preprocess_data()
    
    if data is None:
        print("❌ Failed to load data, exiting...")
        return
    
    print(f"✅ Data loaded: {len(data)} samples from {data['Date'].min()} to {data['Date'].max()}")
    
    print("\nStep 2: Testing context window configurations...")
    results = test_context_windows(data)
    
    print("\nStep 3: Analyzing results...")
    analyze_results(results)
    
    print("\nStep 4: Saving results...")
    save_results(results)
    
    print(f"\n🎊 Simplified Phase 1 Configuration Testing Complete!")
    print("="*60)
    
    # Next steps recommendation
    print(f"\n📋 NEXT STEPS:")
    print("1. Install full Chronos environment for advanced testing")
    print("2. Test actual Chronos model variants")
    print("3. Implement ensemble methods")
    print("4. Validate on out-of-sample data")

if __name__ == "__main__":
    main()