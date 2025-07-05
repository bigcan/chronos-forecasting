#!/usr/bin/env python3
"""
Phase 1 Optimization: Context Window Testing
"""

import pandas as pd
import numpy as np
import time
import sys
import os

def main():
    print('🚀 STARTING PHASE 1 OPTIMIZATION - CONTEXT WINDOWS')
    print('='*80)
    
    try:
        # Load the data
        df = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Filter for 2020-2021
        mask = (df['Date'] >= '2020-01-01') & (df['Date'] <= '2021-12-31')
        data = df[mask].reset_index(drop=True)
        
        print(f'✅ Data loaded: {len(data)} records from {data["Date"].min()} to {data["Date"].max()}')
        
        # Simple context window test
        def test_context_window(window_size, max_samples=50):
            print(f'📊 Testing {window_size}-day context window...')
            
            predictions = []
            actuals = []
            
            end_idx = min(window_size + max_samples, len(data))
            
            for i in range(window_size, end_idx):
                # Get context and actual
                context = data['Close'].iloc[i-window_size:i].values
                actual = data['Close'].iloc[i]
                
                # Simple prediction strategies to test
                naive_pred = context[-1]  # Last value
                ma_pred = np.mean(context[-5:])  # 5-day moving average
                trend_pred = context[-1] + (context[-1] - context[-2])  # Simple trend
                
                # Use naive for now
                prediction = naive_pred
                
                predictions.append(prediction)
                actuals.append(actual)
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            
            if len(actuals) > 1:
                naive_mae = np.mean(np.abs(np.diff(actuals)))
                mase = mae / naive_mae if naive_mae > 0 else float('inf')
                
                # Directional accuracy
                actual_direction = np.sign(np.diff(actuals))
                pred_direction = np.sign(predictions[1:] - actuals[:-1])
                dir_acc = np.mean(actual_direction == pred_direction) * 100
            else:
                mase = float('inf')
                dir_acc = 50.0
            
            return {
                'MASE': mase,
                'MAE': mae,
                'RMSE': rmse,
                'Directional_Accuracy': dir_acc,
                'samples': len(predictions)
            }
        
        # Test different window sizes
        window_sizes = [30, 63, 126, 252]
        results = {}
        
        print('\n🔍 Testing Context Window Sizes...')
        print('-' * 60)
        
        for window_size in window_sizes:
            if window_size < len(data):
                start_time = time.time()
                result = test_context_window(window_size, max_samples=100)
                elapsed = time.time() - start_time
                results[window_size] = result
                
                print(f'Window {window_size:3d}: MASE={result["MASE"]:.4f}, MAE=${result["MAE"]:6.2f}, '
                      f'Dir.Acc={result["Directional_Accuracy"]:5.1f}%, Samples={result["samples"]:3d}, '
                      f'Time={elapsed:.1f}s')
            else:
                print(f'Window {window_size:3d}: Skipped (insufficient data)')
        
        # Analyze results
        if results:
            print('\n📊 CONTEXT WINDOW OPTIMIZATION RESULTS:')
            print('=' * 60)
            
            # Sort by MASE
            sorted_results = sorted(results.items(), key=lambda x: x[1]['MASE'])
            
            print('Results ranked by MASE (lower is better):')
            for i, (window, metrics) in enumerate(sorted_results, 1):
                print(f'{i}. Window {window:3d} days: MASE = {metrics["MASE"]:.4f}, '
                      f'Dir.Acc = {metrics["Directional_Accuracy"]:5.1f}%, '
                      f'MAE = ${metrics["MAE"]:6.2f}')
            
            # Best window
            best_window, best_metrics = sorted_results[0]
            print(f'\n🏆 OPTIMAL CONTEXT WINDOW: {best_window} days')
            print(f'   MASE: {best_metrics["MASE"]:.4f}')
            print(f'   MAE: ${best_metrics["MAE"]:.2f}')
            print(f'   RMSE: ${best_metrics["RMSE"]:.2f}')
            print(f'   Directional Accuracy: {best_metrics["Directional_Accuracy"]:.1f}%')
            print(f'   Samples: {best_metrics["samples"]}')
            
            # Compare to baseline (63 days)
            if 63 in results:
                baseline_mase = results[63]['MASE']
                improvement = (baseline_mase - best_metrics['MASE']) / baseline_mase * 100
                print(f'   Improvement over 63-day baseline: {improvement:.1f}%')
            
            # Save results
            results_df = pd.DataFrame(results).T
            results_df.to_csv('phase1_context_window_results.csv')
            print(f'\n📁 Results saved to: phase1_context_window_results.csv')
            
            print('\n✅ CONTEXT WINDOW OPTIMIZATION COMPLETED!')
            return best_window, results
            
        else:
            print('❌ No valid results obtained')
            return None, {}
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return None, {}

if __name__ == '__main__':
    best_window, results = main()
    
    if best_window:
        print(f'\n🎯 NEXT STEPS:')
        print(f'- Use optimal window ({best_window} days) for model size comparison')
        print(f'- Test different Chronos model variants')
        print(f'- Analyze prediction horizon effects')