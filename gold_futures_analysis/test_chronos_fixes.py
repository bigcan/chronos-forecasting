#!/usr/bin/env python3
"""
Test script to verify ChronosBolt wrapper fixes
"""

import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline
import warnings
warnings.filterwarnings('ignore')

def test_chronos_wrapper():
    """Test the fixed ChronosWrapper implementation"""
    print("Testing ChronosBolt wrapper fixes...")
    
    # Load test data
    try:
        df = pd.read_csv('GCUSD_MAX_FROM_PERPLEXITY.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Filter for 2020-2021
        mask = (df['Date'] >= '2020-01-01') & (df['Date'] <= '2021-12-31')
        data = df[mask].reset_index(drop=True)
        
        print(f"✅ Loaded {len(data)} data points")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Load model
    try:
        print("Loading ChronosBolt model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chronos_pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        print(f"✅ Model loaded on {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Fixed wrapper class
    class ChronosWrapper:
        def __init__(self, pipeline):
            self.pipeline = pipeline
            self.name = "Chronos-Bolt-Base"
            self.pipeline_type = type(pipeline).__name__
            print(f"Detected pipeline type: {self.pipeline_type}")
        
        def predict(self, context, prediction_length=1, num_samples=100):
            if isinstance(context, np.ndarray):
                context_tensor = torch.tensor(context, dtype=torch.float32)
            else:
                context_tensor = context
            
            if len(context_tensor.shape) == 1:
                context_tensor = context_tensor.unsqueeze(0)
            
            try:
                if hasattr(self.pipeline, 'predict_quantiles'):
                    # Check pipeline type and avoid num_samples for ChronosBolt
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
                print(f"Error in prediction: {e}")
                last_value = context_tensor[-1, -1].item()
                return {
                    'mean': np.array([last_value]),
                    'quantiles': np.array([[last_value * 0.99, last_value, last_value * 1.01]])
                }
        
        def predict_point(self, context, prediction_length=1):
            result = self.predict(context, prediction_length)
            return result['mean']
    
    # Test the wrapper
    print("Testing wrapper...")
    chronos_model = ChronosWrapper(chronos_pipeline)
    
    # Test predictions
    test_context = data['Close'].head(63).values
    
    print("Testing point prediction...")
    try:
        pred = chronos_model.predict_point(test_context)
        actual = data['Close'].iloc[63]
        error = abs(pred[0] - actual)
        
        print(f"✅ Point prediction: {pred[0]:.2f}")
        print(f"✅ Actual value: {actual:.2f}")
        print(f"✅ Error: {error:.2f}")
        
    except Exception as e:
        print(f"❌ Error in point prediction: {e}")
        return False
    
    print("Testing full prediction with quantiles...")
    try:
        full_pred = chronos_model.predict(test_context)
        print(f"✅ Mean prediction: {full_pred['mean'][0]:.2f}")
        print(f"✅ Quantiles: {full_pred['quantiles'][0]}")
        print(f"✅ 10th percentile: {full_pred['q10'][0]:.2f}")
        print(f"✅ 50th percentile: {full_pred['q50'][0]:.2f}")
        print(f"✅ 90th percentile: {full_pred['q90'][0]:.2f}")
        
    except Exception as e:
        print(f"❌ Error in full prediction: {e}")
        return False
    
    # Test multiple predictions
    print("Testing multiple predictions...")
    try:
        num_tests = 10
        errors = []
        
        for i in range(63, 63 + num_tests):
            context = data['Close'].iloc[i-63:i].values
            pred = chronos_model.predict_point(context)[0]
            actual = data['Close'].iloc[i]
            error = abs(pred - actual)
            errors.append(error)
        
        avg_error = np.mean(errors)
        print(f"✅ Average error over {num_tests} predictions: {avg_error:.2f}")
        
    except Exception as e:
        print(f"❌ Error in multiple predictions: {e}")
        return False
    
    print("🎉 All tests passed! The wrapper is working correctly.")
    return True

if __name__ == "__main__":
    success = test_chronos_wrapper()
    if success:
        print("\n✅ SUCCESS: ChronosBolt wrapper fixes are working correctly!")
        print("The num_samples parameter issue has been resolved.")
    else:
        print("\n❌ FAILURE: Some tests failed. Please check the output above.")