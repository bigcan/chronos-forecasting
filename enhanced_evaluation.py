import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error

class EnhancedEvaluationFramework:
    """
    Enhanced evaluation framework with better validation and metrics
    """
    
    def __init__(self, data, target_col, scaler=None):
        self.data = data
        self.target_col = target_col
        self.scaler = scaler
        
    def walk_forward_validation(self, models: Dict, initial_train_size: int = 500, 
                               step_size: int = 20, max_predictions: int = 100):
        """
        Walk-forward validation for more robust evaluation
        """
        results = {}
        
        for model_name in models.keys():
            results[model_name] = {
                'predictions': [],
                'actuals': [],
                'dates': [],
                'errors': [],
                'confidence_intervals': []
            }
        
        current_pos = initial_train_size
        
        while current_pos < len(self.data) and len(results[list(models.keys())[0]]['predictions']) < max_predictions:
            # Get current context and target
            context = self.data[self.target_col].iloc[current_pos-63:current_pos].values
            actual = self.data[self.target_col].iloc[current_pos]
            date = self.data['Date'].iloc[current_pos]
            
            # Generate predictions for all models
            for model_name, model in models.items():
                try:
                    if hasattr(model, 'predict_with_uncertainty'):
                        pred_result = model.predict_with_uncertainty(context, prediction_length=1)
                        prediction = pred_result['median'][0]
                        confidence = pred_result['confidence_intervals']
                    else:
                        prediction = model.predict_point(context, prediction_length=1)[0]
                        confidence = None
                    
                    # Store results
                    results[model_name]['predictions'].append(prediction)
                    results[model_name]['actuals'].append(actual)
                    results[model_name]['dates'].append(date)
                    results[model_name]['errors'].append(abs(actual - prediction))
                    results[model_name]['confidence_intervals'].append(confidence)
                    
                except Exception as e:
                    print(f"Error with {model_name} at position {current_pos}: {e}")
                    # Skip this prediction
                    continue
            
            current_pos += step_size
        
        return results
    
    def calculate_enhanced_metrics(self, results: Dict) -> Dict:
        """
        Calculate enhanced metrics including financial-specific ones
        """
        metrics = {}
        
        for model_name, model_results in results.items():
            if not model_results['predictions']:
                continue
                
            predictions = np.array(model_results['predictions'])
            actuals = np.array(model_results['actuals'])
            
            # Basic metrics
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
            
            # Financial metrics
            # Hit rate (directional accuracy)
            if len(actuals) > 1:
                actual_direction = np.sign(np.diff(actuals))
                pred_direction = np.sign(predictions[1:] - actuals[:-1])
                hit_rate = np.mean(actual_direction == pred_direction) * 100
            else:
                hit_rate = 50.0
            
            # Profitability metrics (assuming simple trading strategy)
            returns = np.diff(actuals) / actuals[:-1]  # Actual returns
            predicted_returns = predictions[1:] - actuals[:-1]  # Predicted changes
            
            # Trading signals based on predictions
            signals = np.sign(predicted_returns)
            strategy_returns = signals * returns
            
            # Calculate financial metrics
            total_return = np.prod(1 + strategy_returns) - 1
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(strategy_returns)
            
            # Uncertainty metrics (if available)
            if model_results['confidence_intervals'][0] is not None:
                # Coverage probability
                coverage_80 = self._calculate_coverage(
                    actuals, predictions, model_results['confidence_intervals'], 80
                )
                coverage_95 = self._calculate_coverage(
                    actuals, predictions, model_results['confidence_intervals'], 95
                )
            else:
                coverage_80 = coverage_95 = None
            
            metrics[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Hit_Rate': hit_rate,
                'Total_Return': total_return * 100,  # Convert to percentage
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown * 100,  # Convert to percentage
                'Coverage_80': coverage_80,
                'Coverage_95': coverage_95,
                'N_Predictions': len(predictions)
            }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_coverage(self, actuals: np.ndarray, predictions: np.ndarray, 
                          confidence_intervals: List, confidence_level: int) -> float:
        """Calculate coverage probability of confidence intervals"""
        if confidence_level == 80:
            key_lower, key_upper = 'lower_80', 'upper_80'
        else:
            key_lower, key_upper = 'lower_95', 'upper_95'
        
        coverage_count = 0
        total_count = 0
        
        for i, (actual, ci) in enumerate(zip(actuals, confidence_intervals)):
            if ci is not None:
                lower = ci[key_lower][0] if isinstance(ci[key_lower], np.ndarray) else ci[key_lower]
                upper = ci[key_upper][0] if isinstance(ci[key_upper], np.ndarray) else ci[key_upper]
                
                if lower <= actual <= upper:
                    coverage_count += 1
                total_count += 1
        
        return (coverage_count / total_count * 100) if total_count > 0 else None
    
    def market_regime_analysis(self, results: Dict) -> Dict:
        """
        Analyze performance across different market regimes
        """
        regime_results = {}
        
        for model_name, model_results in results.items():
            if not model_results['predictions']:
                continue
                
            actuals = np.array(model_results['actuals'])
            predictions = np.array(model_results['predictions'])
            
            # Define market regimes based on volatility and trend
            returns = np.diff(actuals) / actuals[:-1]
            volatility = pd.Series(returns).rolling(window=10).std()
            trend = pd.Series(returns).rolling(window=10).mean()
            
            # Classify regimes
            high_vol_mask = volatility > volatility.quantile(0.7)
            low_vol_mask = volatility < volatility.quantile(0.3)
            up_trend_mask = trend > 0
            down_trend_mask = trend < 0
            
            regimes = {
                'High_Volatility': high_vol_mask,
                'Low_Volatility': low_vol_mask,
                'Uptrend': up_trend_mask,
                'Downtrend': down_trend_mask
            }
            
            regime_metrics = {}
            for regime_name, mask in regimes.items():
                mask_indices = mask.dropna().index
                if len(mask_indices) > 5:  # Ensure enough data points
                    regime_preds = predictions[mask_indices]
                    regime_actuals = actuals[mask_indices]
                    
                    mae = np.mean(np.abs(regime_preds - regime_actuals))
                    hit_rate = np.mean(np.sign(regime_preds - regime_actuals[:-1]) == 
                                     np.sign(np.diff(regime_actuals))) * 100
                    
                    regime_metrics[regime_name] = {
                        'MAE': mae,
                        'Hit_Rate': hit_rate,
                        'N_Samples': len(regime_preds)
                    }
            
            regime_results[model_name] = regime_metrics
        
        return regime_results