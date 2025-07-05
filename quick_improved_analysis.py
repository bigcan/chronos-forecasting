#!/usr/bin/env python3
"""
Quick Improved Chronos Analysis - Addresses key performance issues
"""

import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline
import warnings
warnings.filterwarnings('ignore')

def improved_preprocessing(df):
    """Improved data preprocessing"""
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Use more data and returns instead of raw prices
    mask = (data['Date'] >= '2020-01-01') & (data['Date'] <= '2023-12-31')
    data = data[mask].reset_index(drop=True)
    
    # Use log returns for better stationarity
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna().reset_index(drop=True)
    
    # Simple outlier removal
    q1, q3 = data['Returns'].quantile([0.05, 0.95])
    data = data[(data['Returns'] >= q1) & (data['Returns'] <= q3)]
    
    return data

class ImprovedChronosWrapper:
    """Improved Chronos wrapper with proper quantile handling"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.name = "Improved-Chronos"
        
    def predict_point(self, context, prediction_length=1, num_samples=50):
        """Improved prediction with ensemble and proper quantiles"""
        try:
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            
            # Use proper quantile levels for ChronosBolt
            quantile_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # Within training range
            
            quantiles, _ = self.pipeline.predict_quantiles(
                context=context_tensor,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels
            )
            
            # Use multiple samples for ensemble
            predictions = []
            for _ in range(num_samples):
                # Sample from quantile distribution
                sample_idx = np.random.choice(len(quantile_levels))
                pred = quantiles[0, 0, sample_idx].cpu().item()
                predictions.append(pred)
            
            # Return ensemble median
            return np.array([np.median(predictions)])
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Conservative fallback
            return np.array([context[-1] * 0.99])  # Slight downward bias

def evaluate_improved_chronos():
    """Run improved evaluation"""
    print("🚀 Quick Improved Chronos Analysis")
    print("="*50)
    
    # Load and preprocess data
    try:
        df = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
        data = improved_preprocessing(df)
        print(f"✅ Processed {len(data)} data points")
    except:
        print("❌ Could not load data")
        return
    
    # Load model with better configuration
    print("🤖 Loading Chronos model...")
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",  # Smallest model for quick testing
            device_map="cpu",
            torch_dtype=torch.float32
        )
        improved_chronos = ImprovedChronosWrapper(pipeline)
        print(f"✅ Loaded {improved_chronos.name}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Create baseline models
    def naive_forecast(context, prediction_length=1):
        return np.array([context[-1]])
    
    def mean_revert_forecast(context, prediction_length=1):
        # Simple mean reversion
        recent_mean = np.mean(context[-20:])
        return np.array([context[-1] * 0.7 + recent_mean * 0.3])
    
    models = {
        'Naive': type('Model', (), {'predict_point': lambda self, c, p=1: naive_forecast(c, p)})(),
        'Mean_Revert': type('Model', (), {'predict_point': lambda self, c, p=1: mean_revert_forecast(c, p)})(),
        'Improved_Chronos': improved_chronos
    }
    
    # Evaluation
    print("📊 Running evaluation...")
    results = {}
    window_size = 60
    
    for model_name in models:
        results[model_name] = {'predictions': [], 'actuals': [], 'errors': []}
    
    # Rolling window evaluation
    n_predictions = 100
    start_idx = len(data) - n_predictions - window_size
    
    for i in range(start_idx, start_idx + n_predictions):
        context = data['Returns'].iloc[i-window_size:i].values
        actual = data['Returns'].iloc[i]
        
        for model_name, model in models.items():
            try:
                pred = model.predict_point(context)[0]
                error = abs(actual - pred)
                
                results[model_name]['predictions'].append(pred)
                results[model_name]['actuals'].append(actual)
                results[model_name]['errors'].append(error)
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        if i % 20 == 0:
            print(f"Progress: {i - start_idx + 1}/{n_predictions}")
    
    # Calculate metrics
    print("\n📈 Results:")
    print("-" * 50)
    
    for model_name, model_results in results.items():
        if not model_results['predictions']:
            continue
            
        predictions = np.array(model_results['predictions'])
        actuals = np.array(model_results['actuals'])
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        # Directional accuracy
        if len(actuals) > 1:
            actual_direction = np.sign(actuals[1:] - actuals[:-1])
            pred_direction = np.sign(predictions[1:] - actuals[:-1])
            hit_rate = np.mean(actual_direction == pred_direction) * 100
        else:
            hit_rate = 50.0
        
        print(f"{model_name:20s}: MAE={mae:.4f}, RMSE={rmse:.4f}, Hit Rate={hit_rate:.1f}%")
    
    # Key improvements achieved
    if 'Improved_Chronos' in results and 'Naive' in results:
        chronos_mae = np.mean(results['Improved_Chronos']['errors'])
        naive_mae = np.mean(results['Naive']['errors'])
        improvement = ((naive_mae - chronos_mae) / naive_mae) * 100
        
        print(f"\n💡 Improvement over Naive: {improvement:+.1f}%")
        
        if improvement > 0:
            print("✅ Improved Chronos shows better performance!")
        else:
            print("⚠️ Further optimization needed")
    
    print("\n🔍 Key Improvements Applied:")
    print("✅ Used log returns instead of raw prices")
    print("✅ Proper quantile levels for ChronosBolt")
    print("✅ Ensemble predictions with multiple samples")
    print("✅ Better error handling and fallbacks")
    print("✅ Extended dataset for more robust evaluation")
    
    return results

if __name__ == "__main__":
    results = evaluate_improved_chronos()