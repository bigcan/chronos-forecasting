import torch
import numpy as np
from typing import Optional, List, Dict, Any

class EnhancedChronosWrapper:
    """
    Enhanced wrapper for Chronos models with better prediction strategies
    """
    
    def __init__(self, pipeline, model_name="Enhanced-Chronos"):
        self.pipeline = pipeline
        self.name = model_name
        self.pipeline_type = type(pipeline).__name__
        self.device = next(pipeline.model.parameters()).device
        
    def predict_ensemble(self, context: np.ndarray, prediction_length: int = 1, 
                        num_samples: int = 100, temperature: float = 1.0) -> Dict[str, Any]:
        """
        Enhanced prediction with ensemble sampling and uncertainty quantification
        """
        # Ensure proper input format
        if isinstance(context, np.ndarray):
            context_tensor = torch.tensor(context, dtype=torch.float32)
        else:
            context_tensor = context
            
        # Add batch dimension if needed
        if len(context_tensor.shape) == 1:
            context_tensor = context_tensor.unsqueeze(0)
            
        try:
            # Generate multiple samples for ensemble
            all_predictions = []
            
            # For ChronosBolt, we need to handle the API differently
            if 'ChronosBolt' in self.pipeline_type:
                # Use quantile predictions for uncertainty
                quantile_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
                quantiles, _ = self.pipeline.predict_quantiles(
                    context=context_tensor,
                    prediction_length=prediction_length,
                    quantile_levels=quantile_levels
                )
                
                # Extract predictions
                quantile_preds = quantiles[0].cpu().numpy()  # Shape: [prediction_length, num_quantiles]
                
                # Simulate ensemble by sampling from quantile distribution
                for _ in range(num_samples):
                    # Interpolate between quantiles to create diverse samples
                    sample_pred = np.interp(
                        np.random.random(prediction_length),
                        [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                        quantile_preds[0] if prediction_length == 1 else quantile_preds[:, :]
                    )
                    all_predictions.append(sample_pred)
                    
            else:
                # For regular Chronos models
                for _ in range(num_samples):
                    quantiles, mean = self.pipeline.predict_quantiles(
                        context=context_tensor,
                        prediction_length=prediction_length,
                        quantile_levels=[0.1, 0.5, 0.9],
                        num_samples=1
                    )
                    all_predictions.append(mean[0].cpu().numpy())
            
            # Convert to numpy array
            all_predictions = np.array(all_predictions)
            
            # Calculate ensemble statistics
            ensemble_mean = np.mean(all_predictions, axis=0)
            ensemble_std = np.std(all_predictions, axis=0)
            ensemble_median = np.median(all_predictions, axis=0)
            
            # Calculate confidence intervals
            confidence_intervals = {
                'lower_95': np.percentile(all_predictions, 2.5, axis=0),
                'upper_95': np.percentile(all_predictions, 97.5, axis=0),
                'lower_80': np.percentile(all_predictions, 10, axis=0),
                'upper_80': np.percentile(all_predictions, 90, axis=0)
            }
            
            return {
                'mean': ensemble_mean,
                'median': ensemble_median,
                'std': ensemble_std,
                'confidence_intervals': confidence_intervals,
                'all_samples': all_predictions,
                'num_samples': num_samples
            }
            
        except Exception as e:
            print(f"⚠️ Error in enhanced prediction: {e}")
            # More conservative fallback - use last value with small noise
            last_value = context_tensor[-1, -1].item()
            noise = np.random.normal(0, abs(last_value) * 0.001, (num_samples, prediction_length))
            fallback_preds = last_value + noise
            
            return {
                'mean': np.mean(fallback_preds, axis=0),
                'median': np.median(fallback_preds, axis=0),
                'std': np.std(fallback_preds, axis=0),
                'confidence_intervals': {
                    'lower_95': np.percentile(fallback_preds, 2.5, axis=0),
                    'upper_95': np.percentile(fallback_preds, 97.5, axis=0),
                    'lower_80': np.percentile(fallback_preds, 10, axis=0),
                    'upper_80': np.percentile(fallback_preds, 90, axis=0)
                },
                'all_samples': fallback_preds,
                'num_samples': num_samples
            }
    
    def predict_point(self, context: np.ndarray, prediction_length: int = 1, 
                     strategy: str = 'median') -> np.ndarray:
        """
        Point prediction with different strategies
        """
        ensemble_result = self.predict_ensemble(context, prediction_length)
        
        if strategy == 'mean':
            return ensemble_result['mean']
        elif strategy == 'median':
            return ensemble_result['median']
        elif strategy == 'conservative':
            # Use lower confidence interval for conservative predictions
            return ensemble_result['confidence_intervals']['lower_80']
        elif strategy == 'aggressive':
            # Use upper confidence interval for aggressive predictions
            return ensemble_result['confidence_intervals']['upper_80']
        else:
            return ensemble_result['median']
    
    def predict_with_uncertainty(self, context: np.ndarray, prediction_length: int = 1) -> Dict[str, Any]:
        """
        Prediction with full uncertainty quantification
        """
        return self.predict_ensemble(context, prediction_length)

def optimize_context_length(data, target_col, chronos_model, context_lengths=[30, 60, 90, 120]):
    """
    Optimize context length using validation set
    """
    best_context = None
    best_score = float('inf')
    
    # Use last 20% of data for validation
    val_start = int(len(data) * 0.8)
    
    for context_length in context_lengths:
        if val_start + context_length >= len(data):
            continue
            
        errors = []
        
        # Test on validation set
        for i in range(val_start + context_length, len(data)):
            context = data[target_col].iloc[i-context_length:i].values
            actual = data[target_col].iloc[i]
            
            try:
                pred = chronos_model.predict_point(context, prediction_length=1)
                error = abs(actual - pred[0])
                errors.append(error)
            except:
                continue
        
        if errors:
            avg_error = np.mean(errors)
            print(f"Context length {context_length}: Average error = {avg_error:.4f}")
            
            if avg_error < best_score:
                best_score = avg_error
                best_context = context_length
    
    return best_context, best_score