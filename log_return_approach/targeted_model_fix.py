#!/usr/bin/env python3
"""
Targeted fix for model prediction issues
"""

# TARGETED FIX: Model prediction pipeline errors
print("🎯 TARGETED MODEL PREDICTION FIXES")

def create_model_prediction_fix():
    """
    Create specific fixes for model prediction failures
    """
    return '''
# Fix 1: Model Loading and Validation
print("🔧 Validating model loading...")

# Ensure models are properly loaded and accessible
for model_name, pipeline in models.items():
    print(f"Testing model: {model_name}")
    try:
        # Test basic model attributes
        print(f"   Model type: {type(pipeline)}")
        print(f"   Model device: {getattr(pipeline, 'device', 'unknown')}")
        
        # Test with dummy data
        dummy_tensor = torch.randn(63, dtype=torch.float32)
        if CONFIG.get('device') == 'cuda' and torch.cuda.is_available():
            dummy_tensor = dummy_tensor.cuda()
        
        # Test prediction
        if 'bolt' in model_name:
            test_pred = pipeline.predict(context=dummy_tensor, prediction_length=1)
        else:
            test_pred = pipeline.predict(context=dummy_tensor, prediction_length=1, num_samples=10)
        
        print(f"   ✅ Model {model_name} working correctly")
        
    except Exception as e:
        print(f"   ❌ Model {model_name} has issues: {e}")

# Fix 2: Device Compatibility
print("\\n🔧 Fixing device compatibility...")

def ensure_device_compatibility(tensor, target_device):
    """Ensure tensor is on correct device"""
    try:
        if target_device == 'cuda' and torch.cuda.is_available():
            if not tensor.is_cuda:
                tensor = tensor.cuda()
        else:
            if tensor.is_cuda:
                tensor = tensor.cpu()
        return tensor
    except Exception as e:
        print(f"Device compatibility error: {e}")
        return tensor.cpu()  # Fallback to CPU

# Fix 3: Model Prediction Wrapper
def robust_model_predict(pipeline, model_name, context_tensor, horizon, config):
    """
    Robust wrapper for model prediction with multiple fallback strategies
    """
    try:
        # Strategy 1: Original prediction
        if 'bolt' in model_name:
            forecast = pipeline.predict(
                context=context_tensor,
                prediction_length=horizon
            )
            
            # Extract predictions with multiple strategies
            if hasattr(forecast, 'shape') and len(forecast.shape) == 3:
                median_idx = forecast.shape[1] // 2
                predicted_returns = forecast[0, median_idx, :].cpu().numpy()
            elif hasattr(forecast, 'median'):
                predicted_returns = forecast.median(dim=0).values.cpu().numpy()
            else:
                # Fallback: take mean if median not available
                predicted_returns = forecast.mean(dim=0).cpu().numpy()
                
        else:
            # Regular Chronos models
            forecast = pipeline.predict(
                context=context_tensor,
                prediction_length=horizon,
                num_samples=config.get('num_samples', 100)
            )
            
            if isinstance(forecast, tuple):
                predicted_returns = forecast[0].median(dim=0).values.cpu().numpy()
            elif hasattr(forecast, 'median'):
                predicted_returns = forecast.median(dim=0).values.cpu().numpy()
            else:
                predicted_returns = forecast.mean(dim=0).cpu().numpy()
        
        return predicted_returns, None
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Strategy 2: Clear memory and retry on CPU
            try:
                torch.cuda.empty_cache()
                context_cpu = context_tensor.cpu()
                
                if 'bolt' in model_name:
                    forecast = pipeline.predict(
                        context=context_cpu,
                        prediction_length=horizon
                    )
                else:
                    forecast = pipeline.predict(
                        context=context_cpu,
                        prediction_length=horizon,
                        num_samples=min(config.get('num_samples', 100), 50)  # Reduce samples
                    )
                
                if hasattr(forecast, 'median'):
                    predicted_returns = forecast.median(dim=0).values.cpu().numpy()
                else:
                    predicted_returns = forecast.mean(dim=0).cpu().numpy()
                
                return predicted_returns, "memory_fallback"
                
            except Exception as e2:
                return None, f"memory_error: {e2}"
        else:
            return None, f"runtime_error: {e}"
            
    except Exception as e:
        return None, f"prediction_error: {e}"

# Fix 4: Enhanced forecasting loop with robust prediction
def zero_shot_rolling_forecast_fixed(model_name, pipeline, context_window, horizon, max_samples=None):
    """
    Fixed forecasting with robust model prediction handling
    """
    print(f"🎯 TARGETED FIX: {model_name}, Context: {context_window}, Horizon: {horizon}")
    
    all_returns = close_returns
    all_prices = df['Close']
    start_idx = context_window
    max_possible_forecasts = len(all_returns) - start_idx - horizon + 1
    
    if max_samples is None:
        end_idx = len(all_returns) - horizon + 1
        actual_forecasts = max_possible_forecasts
    else:
        end_idx = min(start_idx + max_samples, len(all_returns) - horizon + 1)
        actual_forecasts = min(max_samples, max_possible_forecasts)
    
    print(f"🎯 Expected iterations: {end_idx - start_idx}")
    
    predictions_list = []
    actuals_list = []
    forecast_count = 0
    error_count = 0
    error_details = {}
    
    for i in range(start_idx, end_idx):
        
        if (i - start_idx) % 50 == 0:
            print(f"🎯 Progress: {i - start_idx}/{end_idx - start_idx} ({forecast_count} successes, {error_count} errors)")
        
        try:
            # Data preparation
            context_data = all_returns.iloc[i-context_window:i].values
            actual_returns = all_returns.iloc[i:i+horizon].values
            
            if len(context_data) != context_window or len(actual_returns) != horizon:
                error_count += 1
                continue
            
            # Tensor creation
            context_tensor = torch.tensor(context_data, dtype=torch.float32)
            context_tensor = ensure_device_compatibility(context_tensor, CONFIG.get('device', 'cpu'))
            
            # Robust model prediction
            predicted_returns, error_msg = robust_model_predict(
                pipeline, model_name, context_tensor, horizon, CONFIG
            )
            
            if predicted_returns is not None:
                predictions_list.append(predicted_returns)
                actuals_list.append(actual_returns)
                forecast_count += 1
                
                if error_msg:  # Track fallback usage
                    error_details[error_msg] = error_details.get(error_msg, 0) + 1
            else:
                error_count += 1
                if error_msg:
                    error_details[error_msg] = error_details.get(error_msg, 0) + 1
        
        except Exception as e:
            error_count += 1
            error_key = f"loop_error: {type(e).__name__}"
            error_details[error_key] = error_details.get(error_key, 0) + 1
            continue
        
        # Safety check
        if error_count > 100 and forecast_count == 0:
            print(f"🚨 No successes after 100 errors - stopping")
            break
    
    print(f"\\n🎯 TARGETED FIX RESULTS:")
    print(f"   Successful forecasts: {forecast_count}")
    print(f"   Errors: {error_count}")
    print(f"   Success rate: {forecast_count/(forecast_count + error_count)*100:.1f}%")
    
    if error_details:
        print(f"\\n❌ ERROR DETAILS:")
        for error_type, count in error_details.items():
            print(f"   {error_type}: {count}")
    
    if len(predictions_list) == 0:
        return None
    
    # Calculate basic metrics
    all_predictions = np.concatenate(predictions_list)
    all_actuals = np.concatenate(actuals_list)
    
    return_mae = np.mean(np.abs(all_predictions - all_actuals))
    hit_rate = np.mean(np.sign(all_predictions) == np.sign(all_actuals))
    
    return {
        'model': model_name,
        'context_window': context_window,
        'horizon': horizon,
        'num_forecasts': len(predictions_list),
        'return_mae': return_mae,
        'hit_rate': hit_rate,
        'error_count': error_count,
        'error_details': error_details,
        'success_rate': forecast_count/(forecast_count + error_count) if (forecast_count + error_count) > 0 else 0,
        'data_utilization': len(predictions_list) / max_possible_forecasts
    }
'''

print("✅ Targeted model prediction fixes created")
print("🎯 Key targeted fixes:")
print("   • Model validation before forecasting")
print("   • Device compatibility handling")
print("   • Memory management fallbacks")
print("   • Multiple prediction extraction strategies")
print("   • Detailed error categorization")
print("   • Progress tracking and safety limits")

if __name__ == "__main__":
    fix_code = create_model_prediction_fix()
    print("\\n📁 Ready to apply targeted fixes")