#!/usr/bin/env python3
"""
Phase 1 Configuration Optimization Script
Extracted from the notebook for standalone execution
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

import torch
from chronos import BaseChronosPipeline

import itertools
from typing import Dict, List, Tuple, Any
import time

# Try to import plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    pio.renderers.default = "plotly_mimetype+notebook"
    plotly_available = True
    print("✅ Plotly imports successful")
except ImportError:
    plotly_available = False
    print("⚠️ Plotly not available, using matplotlib fallbacks")

from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

print("🚀 Starting Phase 1: Chronos Configuration Optimization")
print("="*60)

# Load and preprocess data
def load_and_preprocess_data():
    """Load and preprocess gold futures data"""
    try:
        df = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
        print("✅ Data loaded successfully")
    except FileNotFoundError:
        print("❌ Error: GCUSD_MAX_FROM_PERPLEXITY.csv not found")
        return None
    
    # Preprocess
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Filter for 2020-2021 data
    mask = (data['Date'] >= '2020-01-01') & (data['Date'] <= '2021-12-31')
    data = data[mask].reset_index(drop=True)
    
    # Handle missing values
    data = data.ffill()
    
    # Create target variable
    data['Target'] = data['Close'].shift(-1)
    data = data[:-1].reset_index(drop=True)
    
    print(f"Preprocessed dataset shape: {data.shape}")
    return data

# Chronos wrapper class
class ChronosWrapper:
    """Wrapper to make Chronos compatible with evaluation framework"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.name = "Chronos-Bolt-Base"
        self.pipeline_type = type(pipeline).__name__
        print(f"Detected pipeline type: {self.pipeline_type}")
    
    def predict(self, context, prediction_length=1, num_samples=100):
        """Generate predictions using Chronos pipeline"""
        if isinstance(context, np.ndarray):
            context_tensor = torch.tensor(context, dtype=torch.float32)
        else:
            context_tensor = context
        
        if len(context_tensor.shape) == 1:
            context_tensor = context_tensor.unsqueeze(0)
        
        try:
            if hasattr(self.pipeline, 'predict_quantiles'):
                if 'ChronosBolt' in self.pipeline_type:
                    quantiles, mean = self.pipeline.predict_quantiles(
                        context=context_tensor,
                        prediction_length=prediction_length,
                        quantile_levels=[0.1, 0.5, 0.9]
                    )
                else:
                    quantiles, mean = self.pipeline.predict_quantiles(
                        context=context_tensor,
                        prediction_length=prediction_length,
                        quantile_levels=[0.1, 0.5, 0.9],
                        num_samples=num_samples
                    )
                
                return {
                    'mean': mean[0].cpu().numpy(),
                    'quantiles': quantiles[0].cpu().numpy(),
                    'q10': quantiles[0, :, 0].cpu().numpy(),
                    'q50': quantiles[0, :, 1].cpu().numpy(),
                    'q90': quantiles[0, :, 2].cpu().numpy()
                }
            else:
                # Fallback
                last_value = context_tensor[-1, -1].item()
                return {
                    'mean': np.array([last_value]),
                    'quantiles': np.array([[last_value * 0.99, last_value, last_value * 1.01]])
                }
                
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            last_value = context_tensor[-1, -1].item()
            return {
                'mean': np.array([last_value]),
                'quantiles': np.array([[last_value * 0.99, last_value, last_value * 1.01]])
            }
    
    def predict_point(self, context, prediction_length=1):
        """Generate point predictions (mean)"""
        result = self.predict(context, prediction_length)
        return result['mean']

# Configuration Optimizer Class
class ConfigurationOptimizer:
    """Systematic configuration optimization for Chronos models"""
    
    def __init__(self, data):
        self.data = data
        self.optimization_results = {}
        
    def create_fev_dataset_with_window(self, data, window_size):
        """Create FEV dataset with custom window size"""
        records = []
        
        for i in range(window_size, len(data)):
            historical_data = data.iloc[i-window_size:i]
            target = data.iloc[i]['Close']
            
            record = {
                'unique_id': f'gold_futures_{i}',
                'ds': data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                'y': target,
                'historical_data': historical_data['Close'].values.tolist(),
                'context_length': window_size,
                'prediction_length': 1
            }
            records.append(record)
        
        return records
    
    def load_chronos_model(self, model_name):
        """Load different Chronos model variants"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            pipeline = BaseChronosPipeline.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
            )
            return ChronosWrapper(pipeline)
        except Exception as e:
            print(f"⚠️ Error loading {model_name}: {e}")
            return None
    
    def test_context_windows(self, window_sizes=[30, 63, 126, 252], max_samples=100):
        """Test different context window lengths"""
        print("🔍 Testing Context Window Optimization")
        print("=" * 50)
        
        # Load base model once
        base_model = self.load_chronos_model("amazon/chronos-bolt-base")
        if base_model is None:
            print("❌ Failed to load base model")
            return {}
        
        context_results = {}
        
        for window_size in window_sizes:
            print(f"\nTesting context window: {window_size} days")
            
            try:
                test_dataset = self.create_fev_dataset_with_window(self.data, window_size)
                
                if len(test_dataset) < max_samples:
                    max_samples_actual = len(test_dataset)
                else:
                    max_samples_actual = max_samples
                
                test_dataset = test_dataset[:max_samples_actual]
                
                results = self.evaluate_configuration(
                    base_model, 
                    test_dataset, 
                    f"Context-{window_size}",
                    window_size=window_size
                )
                
                context_results[window_size] = results
                
                print(f"✅ Window {window_size}: MASE = {results['MASE']:.4f}, Dir.Acc = {results['Directional_Accuracy']:.1f}%")
                
            except Exception as e:
                print(f"❌ Error testing window {window_size}: {e}")
                continue
        
        return context_results
    
    def test_model_sizes(self, max_samples=100):
        """Test different Chronos model sizes"""
        print("\n🔍 Testing Model Size Optimization")
        print("=" * 50)
        
        model_variants = {
            'Chronos-Bolt-Tiny': 'amazon/chronos-bolt-tiny',
            'Chronos-Bolt-Small': 'amazon/chronos-bolt-small', 
            'Chronos-Bolt-Base': 'amazon/chronos-bolt-base'
        }
        
        model_results = {}
        test_dataset = self.create_fev_dataset_with_window(self.data, 63)[:max_samples]
        
        for model_name, model_path in model_variants.items():
            print(f"\nTesting model: {model_name}")
            
            try:
                start_time = time.time()
                model = self.load_chronos_model(model_path)
                
                if model is None:
                    continue
                
                load_time = time.time() - start_time
                
                start_eval = time.time()
                results = self.evaluate_configuration(
                    model, 
                    test_dataset, 
                    model_name,
                    window_size=63
                )
                eval_time = time.time() - start_eval
                
                results['load_time'] = load_time
                results['eval_time'] = eval_time
                results['time_per_prediction'] = eval_time / len(test_dataset)
                
                model_results[model_name] = results
                
                print(f"✅ {model_name}: MASE = {results['MASE']:.4f}, Dir.Acc = {results['Directional_Accuracy']:.1f}%, Time/pred = {results['time_per_prediction']:.3f}s")
                
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
                continue
        
        return model_results
    
    def test_prediction_horizons(self, horizons=[1, 3, 7], max_samples=50):
        """Test different prediction horizons"""
        print("\n🔍 Testing Prediction Horizon Optimization")
        print("=" * 50)
        
        # Load base model
        base_model = self.load_chronos_model("amazon/chronos-bolt-base")
        if base_model is None:
            print("❌ Failed to load base model")
            return {}
        
        horizon_results = {}
        
        for horizon in horizons:
            print(f"\nTesting prediction horizon: {horizon} days")
            
            try:
                test_dataset = self.create_multistep_dataset(self.data, 63, horizon, max_samples)
                
                if len(test_dataset) == 0:
                    print(f"⚠️ Not enough data for {horizon}-day horizon")
                    continue
                
                results = self.evaluate_multistep_configuration(
                    base_model,
                    test_dataset,
                    f"Horizon-{horizon}",
                    prediction_length=horizon
                )
                
                horizon_results[horizon] = results
                
                print(f"✅ Horizon {horizon}: MASE = {results['MASE']:.4f}, Dir.Acc = {results['Directional_Accuracy']:.1f}%")
                
            except Exception as e:
                print(f"❌ Error testing horizon {horizon}: {e}")
                continue
        
        return horizon_results
    
    def create_multistep_dataset(self, data, window_size, prediction_length, max_samples):
        """Create dataset for multi-step prediction evaluation"""
        records = []
        
        for i in range(window_size, len(data) - prediction_length + 1):
            historical_data = data.iloc[i-window_size:i]
            targets = data.iloc[i:i+prediction_length]['Close'].values.tolist()
            
            record = {
                'unique_id': f'gold_futures_{i}',
                'ds': data.iloc[i]['Date'].strftime('%Y-%m-%d'),
                'y': targets,
                'historical_data': historical_data['Close'].values.tolist(),
                'context_length': window_size,
                'prediction_length': prediction_length
            }
            records.append(record)
            
            if len(records) >= max_samples:
                break
        
        return records
    
    def evaluate_configuration(self, model, test_dataset, config_name, window_size=63):
        """Evaluate a single configuration"""
        predictions = []
        actuals = []
        
        for i, sample in enumerate(test_dataset):
            try:
                context = np.array(sample['historical_data'])
                actual = sample['y']
                
                pred = model.predict_point(context, prediction_length=1)
                prediction = pred[0] if isinstance(pred, np.ndarray) else pred
                
                predictions.append(prediction)
                actuals.append(actual)
                
            except Exception as e:
                predictions.append(context[-1])
                actuals.append(actual)
        
        return self.calculate_comprehensive_metrics(predictions, actuals, config_name)
    
    def evaluate_multistep_configuration(self, model, test_dataset, config_name, prediction_length=1):
        """Evaluate multi-step prediction configuration"""
        all_predictions = []
        all_actuals = []
        
        for sample in test_dataset:
            try:
                context = np.array(sample['historical_data'])
                actuals = sample['y']
                
                pred = model.predict_point(context, prediction_length=prediction_length)
                predictions = pred if isinstance(pred, (list, np.ndarray)) else [pred]
                
                for i in range(min(len(predictions), len(actuals))):
                    all_predictions.append(predictions[i])
                    all_actuals.append(actuals[i])
                
            except Exception as e:
                for actual in actuals:
                    all_predictions.append(context[-1])
                    all_actuals.append(actual)
        
        return self.calculate_comprehensive_metrics(all_predictions, all_actuals, config_name)
    
    def calculate_comprehensive_metrics(self, predictions, actuals, config_name):
        """Calculate comprehensive metrics for configuration evaluation"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        if len(actuals) > 1:
            naive_forecast = actuals[:-1]
            naive_mae = np.mean(np.abs(actuals[1:] - naive_forecast))
            mase = mae / naive_mae if naive_mae > 0 else np.inf
        else:
            mase = np.inf
        
        if len(actuals) > 1:
            actual_direction = np.sign(np.diff(actuals))
            pred_direction = np.sign(predictions[1:] - actuals[:-1])
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = 50.0
        
        bias = np.mean(predictions - actuals)
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'config_name': config_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MASE': mase,
            'Directional_Accuracy': directional_accuracy,
            'Bias': bias,
            'R_Squared': r_squared,
            'n_predictions': len(predictions)
        }

def main():
    """Main execution function"""
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    
    if data is None:
        print("❌ Failed to load data, exiting...")
        return
    
    # Initialize optimizer
    optimizer = ConfigurationOptimizer(data)
    
    # Run optimizations
    print("\n🚀 Starting Configuration Optimization Tests...")
    
    # Test 1: Context Windows
    context_results = optimizer.test_context_windows([30, 63, 126, 252], max_samples=100)
    
    # Test 2: Model Sizes
    model_results = optimizer.test_model_sizes(max_samples=100)
    
    # Test 3: Prediction Horizons
    horizon_results = optimizer.test_prediction_horizons([1, 3, 7], max_samples=50)
    
    # Analyze results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    all_results = {}
    
    if context_results:
        all_results.update({f"Context-{k}d": v for k, v in context_results.items()})
        print("\n📊 Context Window Results:")
        context_df = pd.DataFrame(context_results).T.round(4)
        context_sorted = context_df.sort_values('MASE')
        print(context_sorted[['MASE', 'Directional_Accuracy', 'MAE', 'RMSE']])
        
        best_window = context_sorted.index[0]
        print(f"🏆 Best Context Window: {best_window} days (MASE: {context_sorted.iloc[0]['MASE']:.4f})")
    
    if model_results:
        all_results.update(model_results)
        print("\n📊 Model Size Results:")
        model_df = pd.DataFrame(model_results).T.round(4)
        model_sorted = model_df.sort_values('MASE')
        print(model_sorted[['MASE', 'Directional_Accuracy', 'MAE', 'time_per_prediction']])
        
        best_model = model_sorted.index[0]
        print(f"🏆 Best Model: {best_model} (MASE: {model_sorted.iloc[0]['MASE']:.4f})")
    
    if horizon_results:
        all_results.update({f"Horizon-{k}d": v for k, v in horizon_results.items()})
        print("\n📊 Prediction Horizon Results:")
        horizon_df = pd.DataFrame(horizon_results).T.round(4)
        horizon_sorted = horizon_df.sort_values('MASE')
        print(horizon_sorted[['MASE', 'Directional_Accuracy', 'MAE', 'RMSE']])
        
        best_horizon = horizon_sorted.index[0]
        print(f"🏆 Best Horizon: {best_horizon} days (MASE: {horizon_sorted.iloc[0]['MASE']:.4f})")
    
    # Overall best configuration
    if all_results:
        best_overall = min(all_results.items(), key=lambda x: x[1]['MASE'])
        best_config_name, best_metrics = best_overall
        
        print(f"\n🏆 OVERALL BEST CONFIGURATION: {best_config_name}")
        print("=" * 50)
        print(f"MASE: {best_metrics['MASE']:.4f}")
        print(f"MAE: ${best_metrics['MAE']:.2f}")
        print(f"RMSE: ${best_metrics['RMSE']:.2f}")
        print(f"MAPE: {best_metrics['MAPE']:.2f}%")
        print(f"Directional Accuracy: {best_metrics['Directional_Accuracy']:.1f}%")
        print(f"R-squared: {best_metrics['R_Squared']:.4f}")
        
        # Compare with baselines
        original_mase = 1.0951  # From previous evaluation
        naive_mase = 1.0054    # Naive baseline
        
        improvement = (original_mase - best_metrics['MASE']) / original_mase * 100
        vs_naive = (naive_mase - best_metrics['MASE']) / naive_mase * 100
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"Original Chronos MASE: {original_mase:.4f}")
        print(f"Optimized MASE: {best_metrics['MASE']:.4f}")
        print(f"Improvement: {improvement:.1f}%")
        
        print(f"\nNaive Baseline MASE: {naive_mase:.4f}")
        if best_metrics['MASE'] < naive_mase:
            print(f"✅ SUCCESS: Beats naive by {vs_naive:.1f}%!")
        else:
            print(f"❌ Still behind naive by {-vs_naive:.1f}%")
        
        # Export results
        results_df = pd.DataFrame(all_results).T
        results_df.to_csv('chronos_optimization_results.csv')
        print(f"\n📁 Results exported to: chronos_optimization_results.csv")
    
    print(f"\n🎊 Phase 1 Configuration Optimization Complete!")
    print("="*60)

if __name__ == "__main__":
    main()