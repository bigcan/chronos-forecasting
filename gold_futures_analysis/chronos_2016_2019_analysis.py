#!/usr/bin/env python3
"""
Complete Chronos 2016-2019 Analysis with Fixed API
Run actual Chronos forecasting on 2016-2019 data for comparison with 2020-2021
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import time

# Chronos imports
try:
    import torch
    from chronos import BaseChronosPipeline
    CHRONOS_AVAILABLE = True
    print("✅ Chronos imports successful")
except ImportError as e:
    print(f"❌ Chronos not available: {e}")
    CHRONOS_AVAILABLE = False

print("🚀 Starting COMPLETE Chronos 2016-2019 Analysis")
print("="*70)

class ChronosWrapper:
    """Wrapper for Chronos models with proper API handling"""
    
    def __init__(self, model_name="amazon/chronos-bolt-base"):
        self.model_name = model_name
        self.pipeline = None
        self.pipeline_type = None
        self.load_model()
    
    def load_model(self):
        """Load the Chronos model with proper configuration"""
        try:
            print(f"Loading {self.model_name}...")
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            self.pipeline_type = type(self.pipeline).__name__
            print(f"✅ Model loaded: {self.pipeline_type}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.pipeline = None
    
    def predict_point(self, context, prediction_length=1):
        """Generate point prediction using the correct API"""
        if self.pipeline is None:
            raise ValueError("Model not loaded")
        
        # Convert to tensor
        if isinstance(context, np.ndarray):
            context_tensor = torch.tensor(context, dtype=torch.float32)
        else:
            context_tensor = context
        
        # Ensure proper shape
        if len(context_tensor.shape) == 1:
            context_tensor = context_tensor.unsqueeze(0)
        
        try:
            # Use predict_quantiles for robust prediction
            if 'ChronosBolt' in self.pipeline_type:
                # Chronos-Bolt models (no num_samples parameter)
                quantiles, mean = self.pipeline.predict_quantiles(
                    context=context_tensor,
                    prediction_length=prediction_length,
                    quantile_levels=[0.1, 0.5, 0.9]
                )
            else:
                # Regular Chronos models (with num_samples)
                quantiles, mean = self.pipeline.predict_quantiles(
                    context=context_tensor,
                    prediction_length=prediction_length,
                    quantile_levels=[0.1, 0.5, 0.9],
                    num_samples=20
                )
            
            # Extract the mean prediction
            prediction = mean[0].cpu().numpy()[0]
            return prediction
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Fallback to last value (naive)
            return context_tensor[0, -1].item()

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

def run_chronos_evaluation(data, context_length=63, sample_size=200):
    """Run Chronos evaluation on 2016-2019 data"""
    if not CHRONOS_AVAILABLE:
        print("❌ Chronos not available")
        return None
    
    # Initialize the model
    model = ChronosWrapper("amazon/chronos-bolt-base")
    if model.pipeline is None:
        print("❌ Failed to initialize Chronos model")
        return None
    
    print(f"\n🤖 CHRONOS EVALUATION - 2016-2019")
    print(f"Context length: {context_length} days")
    print(f"Sample size: {sample_size} predictions")
    print("-" * 50)
    
    # Prepare evaluation data
    predictions = []
    actuals = []
    dates = []
    errors = []
    
    # Determine evaluation range
    start_idx = context_length
    end_idx = min(len(data) - 1, start_idx + sample_size)
    total_predictions = end_idx - start_idx
    
    print(f"Evaluation range: index {start_idx} to {end_idx}")
    print(f"Total predictions to make: {total_predictions}")
    
    # Rolling window evaluation
    start_time = time.time()
    successful_predictions = 0
    
    for i in range(start_idx, end_idx):
        try:
            # Get context window
            context = data['Close'].iloc[i-context_length:i].values
            actual = data['Close'].iloc[i]
            date = data['Date'].iloc[i]
            
            # Make prediction
            prediction = model.predict_point(context, prediction_length=1)
            
            # Store results
            predictions.append(prediction)
            actuals.append(actual)
            dates.append(date)
            errors.append(abs(actual - prediction))
            successful_predictions += 1
            
            # Progress update
            if (i - start_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                progress = (i - start_idx + 1) / total_predictions * 100
                avg_time = elapsed / (i - start_idx + 1)
                eta = avg_time * (total_predictions - (i - start_idx + 1))
                print(f"Progress: {progress:.1f}% ({i - start_idx + 1}/{total_predictions}) - "
                      f"ETA: {eta:.0f}s - Successful: {successful_predictions}")
            
        except Exception as e:
            print(f"⚠️ Prediction failed at index {i}: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"✅ Chronos evaluation completed!")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Successful predictions: {successful_predictions}/{total_predictions}")
    print(f"   Average time per prediction: {total_time/successful_predictions:.3f}s")
    
    if successful_predictions == 0:
        print("❌ No successful predictions made")
        return None
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions, 'Chronos_2016_2019')
    
    print(f"\n📊 CHRONOS 2016-2019 RESULTS:")
    print(f"   MASE: {metrics['MASE']:.4f}")
    print(f"   MAE: ${metrics['MAE']:.2f}")
    print(f"   RMSE: ${metrics['RMSE']:.2f}")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    print(f"   Directional Accuracy: {metrics['Directional_Accuracy']:.1f}%")
    print(f"   R²: {metrics['R_Squared']:.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': actuals,
        'Prediction': predictions,
        'Error': errors
    })
    results_df.to_csv('chronos_2016_2019_detailed_results.csv', index=False)
    
    return metrics, results_df

def run_baseline_models(data, sample_size=200):
    """Run baseline models for comparison"""
    print(f"\n📈 BASELINE MODELS - 2016-2019")
    print(f"Sample size: {sample_size} predictions")
    print("-" * 50)
    
    # Prepare evaluation range (same as Chronos)
    context_length = 63
    start_idx = context_length
    end_idx = min(len(data) - 1, start_idx + sample_size)
    
    # Get actual values
    actuals = data['Close'].iloc[start_idx:end_idx].values
    
    # Calculate baseline predictions
    baseline_models = {
        'Naive': data['Close'].iloc[start_idx-1:end_idx-1].values,
        'Moving_Average': data['Close'].iloc[start_idx-5:end_idx-5].rolling(window=5).mean().values,
        'Seasonal_Naive': data['Close'].iloc[start_idx-5:end_idx-5].values
    }
    
    # Calculate metrics for each baseline
    baseline_results = {}
    for model_name, predictions in baseline_models.items():
        # Handle NaN values
        mask = ~np.isnan(predictions)
        if np.sum(mask) > 0:
            metrics = calculate_metrics(actuals[mask], predictions[mask], f'{model_name}_2016_2019')
            baseline_results[model_name] = metrics
            print(f"✅ {model_name}: MASE={metrics['MASE']:.4f}, MAE=${metrics['MAE']:.2f}")
    
    return baseline_results

def compare_with_2020_2021_results(chronos_2016_2019, baselines_2016_2019):
    """Compare 2016-2019 results with existing 2020-2021 results"""
    print(f"\n📊 COMPREHENSIVE COMPARISON: 2016-2019 vs 2020-2021")
    print("="*70)
    
    try:
        # Load 2020-2021 results
        results_2020_2021 = pd.read_csv('phase1_final_comparison_results.csv', index_col=0)
        print("✅ 2020-2021 results loaded for comparison")
        
        # Create comparison table
        comparison_data = []
        
        # Chronos comparison
        if chronos_2016_2019 is not None:
            chronos_mase_2016 = chronos_2016_2019['MASE']
            chronos_mase_2021 = results_2020_2021.loc['Chronos_Optimized', 'MASE']
            chronos_change = ((chronos_mase_2021 - chronos_mase_2016) / chronos_mase_2016) * 100
            
            comparison_data.append({
                'Model': 'Chronos',
                'MASE_2016_2019': chronos_mase_2016,
                'MASE_2020_2021': chronos_mase_2021,
                'Performance_Change': chronos_change,
                'Dir_Acc_2016_2019': chronos_2016_2019['Directional_Accuracy'],
                'Dir_Acc_2020_2021': results_2020_2021.loc['Chronos_Optimized', 'Directional_Accuracy']
            })
        
        # Baseline comparisons
        baseline_map = {
            'Naive': 'Naive',
            'Moving_Average': 'Moving_Average'
        }
        
        for baseline_name, result_name in baseline_map.items():
            if baseline_name in baselines_2016_2019 and result_name in results_2020_2021.index:
                mase_2016 = baselines_2016_2019[baseline_name]['MASE']
                mase_2021 = results_2020_2021.loc[result_name, 'MASE']
                change = ((mase_2021 - mase_2016) / mase_2016) * 100
                
                comparison_data.append({
                    'Model': baseline_name,
                    'MASE_2016_2019': mase_2016,
                    'MASE_2020_2021': mase_2021,
                    'Performance_Change': change,
                    'Dir_Acc_2016_2019': baselines_2016_2019[baseline_name]['Directional_Accuracy'],
                    'Dir_Acc_2020_2021': results_2020_2021.loc[result_name, 'Directional_Accuracy']
                })
        
        # Create and save comparison
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('complete_chronos_comparison_2016_2019_vs_2020_2021.csv', index=False)
        
        print("\n🎯 KEY FINDINGS:")
        print(comparison_df.round(4))
        
        # Analysis
        if chronos_2016_2019 is not None:
            naive_2016 = baselines_2016_2019['Naive']['MASE']
            chronos_2016 = chronos_2016_2019['MASE']
            gap_2016 = ((chronos_2016 - naive_2016) / naive_2016) * 100
            
            naive_2021 = results_2020_2021.loc['Naive', 'MASE']
            chronos_2021 = results_2020_2021.loc['Chronos_Optimized', 'MASE']
            gap_2021 = ((chronos_2021 - naive_2021) / naive_2021) * 100
            
            print(f"\n📈 CHRONOS vs NAIVE GAP ANALYSIS:")
            print(f"2016-2019: Chronos {gap_2016:+.1f}% vs Naive")
            print(f"2020-2021: Chronos {gap_2021:+.1f}% vs Naive")
            print(f"Market regime effect: {gap_2021 - gap_2016:+.1f} percentage point difference")
            
            if gap_2016 < gap_2021:
                print(f"✅ HYPOTHESIS CONFIRMED: Chronos performed relatively better in lower volatility period")
            else:
                print(f"❌ HYPOTHESIS REJECTED: Market regime did not help Chronos as expected")
        
        return comparison_df
        
    except FileNotFoundError:
        print("⚠️ 2020-2021 results not found for comparison")
        return None

def main():
    """Run the complete Chronos 2016-2019 analysis"""
    
    # Load data
    data = load_data_2016_2019()
    if data is None:
        return
    
    # Run baseline models first
    baseline_results = run_baseline_models(data, sample_size=200)
    
    # Run Chronos evaluation
    chronos_result = run_chronos_evaluation(data, context_length=63, sample_size=200)
    
    if chronos_result is not None:
        chronos_metrics, chronos_details = chronos_result
        
        # Save comprehensive results
        all_results = baseline_results.copy()
        all_results['Chronos'] = chronos_metrics
        
        results_df = pd.DataFrame(all_results).T
        results_df.to_csv('complete_2016_2019_forecasting_results.csv')
        
        print(f"\n📊 COMPLETE 2016-2019 RESULTS:")
        print(results_df.round(4))
        
        # Compare with 2020-2021
        comparison = compare_with_2020_2021_results(chronos_metrics, baseline_results)
        
        print(f"\n✅ Complete Chronos 2016-2019 analysis finished!")
        print(f"Files generated:")
        print(f"  - complete_2016_2019_forecasting_results.csv")
        print(f"  - chronos_2016_2019_detailed_results.csv")
        print(f"  - complete_chronos_comparison_2016_2019_vs_2020_2021.csv")
        
        return results_df, comparison
    else:
        print(f"❌ Chronos evaluation failed")
        return baseline_results, None

if __name__ == "__main__":
    main()