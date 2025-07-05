import torch
from chronos import BaseChronosPipeline
import numpy as np
from typing import Dict, List, Tuple

class ChronosModelOptimizer:
    """
    Optimize Chronos model selection and configuration
    """
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.available_models = [
            "amazon/chronos-bolt-tiny",
            "amazon/chronos-bolt-mini", 
            "amazon/chronos-bolt-small",
            "amazon/chronos-bolt-base",
            "amazon/chronos-t5-tiny",
            "amazon/chronos-t5-mini",
            "amazon/chronos-t5-small",
            "amazon/chronos-t5-base"
        ]
    
    def load_model_with_optimal_config(self, model_name: str):
        """
        Load model with optimal configuration
        """
        try:
            # Use appropriate dtype based on device
            if self.device == "cuda":
                torch_dtype = torch.float16  # Better than bfloat16 for most GPUs
            else:
                torch_dtype = torch.float32
            
            pipeline = BaseChronosPipeline.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            
            # Optimize model for inference
            pipeline.model.eval()
            if hasattr(pipeline.model, 'half') and self.device == "cuda":
                pipeline.model.half()
            
            return pipeline
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None
    
    def compare_models(self, data, target_col, context_length=60, n_samples=50):
        """
        Compare different Chronos model variants
        """
        results = {}
        
        # Test subset of data
        test_start = len(data) - n_samples - context_length
        
        for model_name in self.available_models:
            print(f"Testing {model_name}...")
            
            try:
                pipeline = self.load_model_with_optimal_config(model_name)
                if pipeline is None:
                    continue
                
                errors = []
                
                for i in range(test_start, test_start + n_samples):
                    context = data[target_col].iloc[i-context_length:i].values
                    actual = data[target_col].iloc[i]
                    
                    try:
                        # Simple prediction test
                        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
                        
                        if 'ChronosBolt' in type(pipeline).__name__:
                            quantiles, _ = pipeline.predict_quantiles(
                                context=context_tensor,
                                prediction_length=1,
                                quantile_levels=[0.5]
                            )
                            pred = quantiles[0, 0, 0].cpu().item()
                        else:
                            quantiles, mean = pipeline.predict_quantiles(
                                context=context_tensor,
                                prediction_length=1,
                                quantile_levels=[0.5],
                                num_samples=20
                            )
                            pred = mean[0, 0].cpu().item()
                        
                        error = abs(actual - pred)
                        errors.append(error)
                        
                    except Exception as e:
                        print(f"Prediction error for {model_name}: {e}")
                        continue
                
                if errors:
                    avg_error = np.mean(errors)
                    std_error = np.std(errors)
                    
                    results[model_name] = {
                        'avg_error': avg_error,
                        'std_error': std_error,
                        'n_predictions': len(errors),
                        'success_rate': len(errors) / n_samples
                    }
                    
                    print(f"  Average error: {avg_error:.4f}")
                    print(f"  Success rate: {len(errors)}/{n_samples}")
                
                # Clean up memory
                del pipeline
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Failed to test {model_name}: {e}")
                continue
        
        return results
    
    def optimize_hyperparameters(self, pipeline, data, target_col):
        """
        Optimize hyperparameters for a given model
        """
        best_params = {}
        best_score = float('inf')
        
        # Parameter grid
        context_lengths = [30, 60, 90, 120]
        num_samples_options = [20, 50, 100]
        quantile_combinations = [
            [0.1, 0.5, 0.9],
            [0.05, 0.25, 0.5, 0.75, 0.95],
            [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        ]
        
        # Use validation set
        val_start = int(len(data) * 0.8)
        n_val_samples = 30
        
        for context_length in context_lengths:
            if val_start + context_length >= len(data):
                continue
                
            for num_samples in num_samples_options:
                for quantile_levels in quantile_combinations:
                    try:
                        errors = []
                        
                        for i in range(val_start + context_length, 
                                     min(val_start + context_length + n_val_samples, len(data))):
                            context = data[target_col].iloc[i-context_length:i].values
                            actual = data[target_col].iloc[i]
                            
                            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
                            
                            try:
                                if 'ChronosBolt' in type(pipeline).__name__:
                                    quantiles, _ = pipeline.predict_quantiles(
                                        context=context_tensor,
                                        prediction_length=1,
                                        quantile_levels=quantile_levels
                                    )
                                    # Use median
                                    median_idx = len(quantile_levels) // 2
                                    pred = quantiles[0, 0, median_idx].cpu().item()
                                else:
                                    quantiles, mean = pipeline.predict_quantiles(
                                        context=context_tensor,
                                        prediction_length=1,
                                        quantile_levels=quantile_levels,
                                        num_samples=num_samples
                                    )
                                    pred = mean[0, 0].cpu().item()
                                
                                error = abs(actual - pred)
                                errors.append(error)
                                
                            except:
                                continue
                        
                        if len(errors) > 10:  # Ensure enough valid predictions
                            avg_error = np.mean(errors)
                            
                            if avg_error < best_score:
                                best_score = avg_error
                                best_params = {
                                    'context_length': context_length,
                                    'num_samples': num_samples,
                                    'quantile_levels': quantile_levels,
                                    'validation_error': avg_error,
                                    'n_valid_predictions': len(errors)
                                }
                    
                    except Exception as e:
                        continue
        
        return best_params
    
    def get_recommended_model(self, comparison_results: Dict) -> str:
        """
        Get recommended model based on comparison results
        """
        if not comparison_results:
            return "amazon/chronos-bolt-base"  # Default fallback
        
        # Score models based on error and success rate
        best_model = None
        best_score = float('inf')
        
        for model_name, metrics in comparison_results.items():
            # Combined score: avg_error weighted by success rate
            score = metrics['avg_error'] / metrics['success_rate']
            
            if score < best_score:
                best_score = score
                best_model = model_name
        
        return best_model

def create_ensemble_model(models: List, weights: List = None):
    """
    Create ensemble of multiple Chronos models
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    class ChronosEnsemble:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
            self.name = "Chronos-Ensemble"
        
        def predict_point(self, context, prediction_length=1):
            predictions = []
            total_weight = 0
            
            for model, weight in zip(self.models, self.weights):
                try:
                    pred = model.predict_point(context, prediction_length)
                    predictions.append(pred * weight)
                    total_weight += weight
                except:
                    continue
            
            if predictions:
                ensemble_pred = np.sum(predictions, axis=0) / total_weight
                return ensemble_pred
            else:
                # Fallback
                return np.array([context[-1]])
        
        def predict_with_uncertainty(self, context, prediction_length=1):
            all_results = []
            
            for model in self.models:
                try:
                    if hasattr(model, 'predict_with_uncertainty'):
                        result = model.predict_with_uncertainty(context, prediction_length)
                        all_results.append(result)
                except:
                    continue
            
            if all_results:
                # Combine results
                ensemble_mean = np.mean([r['mean'] for r in all_results], axis=0)
                ensemble_std = np.sqrt(np.mean([r['std']**2 for r in all_results], axis=0))
                
                return {
                    'mean': ensemble_mean,
                    'std': ensemble_std,
                    'median': ensemble_mean,  # Approximate
                    'confidence_intervals': all_results[0]['confidence_intervals']  # Use first model's
                }
            else:
                # Fallback
                return {
                    'mean': np.array([context[-1]]),
                    'std': np.array([0.01]),
                    'median': np.array([context[-1]]),
                    'confidence_intervals': None
                }
    
    return ChronosEnsemble(models, weights)