#!/usr/bin/env python3
"""
Comprehensive error fixes for forecasting pipeline
"""

def create_robust_forecasting_function():
    """
    Create a fully robust forecasting function with comprehensive error handling
    """
    return '''
# ULTRA-ROBUST ZERO-SHOT FORECASTING WITH COMPREHENSIVE ERROR HANDLING
print("🔧 Implementing ULTRA-ROBUST Zero-Shot Forecasting...")
print("✅ Comprehensive error handling and debugging enabled")

def zero_shot_rolling_forecast_robust(model_name, pipeline, context_window, horizon, max_samples=None):
    """
    ULTRA-ROBUST: Zero-shot forecasting with comprehensive error handling and debugging
    """
    print(f"🔧 ROBUST FORECAST: {model_name}, Context: {context_window}, Horizon: {horizon}")
    
    predictions_list = []
    actuals_list = []
    dates_list = []
    
    # Use ALL available returns data
    all_returns = close_returns
    all_prices = df['Close']
    
    # Input validation
    if len(all_returns) < context_window + horizon:
        print(f"❌ Insufficient data: {len(all_returns)} < {context_window + horizon}")
        return None
    
    # Start forecasting as soon as we have enough context
    start_idx = context_window
    
    # Calculate maximum possible forecasts
    max_possible_forecasts = len(all_returns) - start_idx - horizon + 1
    
    if max_samples is None:
        end_idx = len(all_returns) - horizon + 1
        actual_forecasts = max_possible_forecasts
    else:
        end_idx = min(start_idx + max_samples, len(all_returns) - horizon + 1)
        actual_forecasts = min(max_samples, max_possible_forecasts)
    
    print(f"🔧 ROBUST DEBUG PARAMETERS:")
    print(f"   Data length: {len(all_returns)}")
    print(f"   Start index: {start_idx}")
    print(f"   End index: {end_idx}")
    print(f"   Expected iterations: {end_idx - start_idx}")
    print(f"   Max possible forecasts: {max_possible_forecasts}")
    print(f"   Target forecasts: {actual_forecasts}")
    print(f"   Device: {CONFIG.get('device', 'cpu')}")
    
    # Error tracking
    forecast_count = 0
    error_count = 0
    error_types = {}
    
    # Test first iteration thoroughly
    print(f"\\n🧪 TESTING FIRST ITERATION:")
    test_i = start_idx
    try:
        test_context = all_returns.iloc[test_i-context_window:test_i].values
        test_actual = all_returns.iloc[test_i:test_i+horizon].values
        test_tensor = torch.tensor(test_context, dtype=torch.float32)
        
        print(f"   ✅ Context shape: {test_context.shape}")
        print(f"   ✅ Actual shape: {test_actual.shape}")
        print(f"   ✅ Tensor creation: {test_tensor.shape}, {test_tensor.dtype}")
        print(f"   ✅ Data range: [{test_context.min():.6f}, {test_context.max():.6f}]")
        
        # Test model prediction on first iteration
        try:
            if 'bolt' in model_name:
                test_forecast = pipeline.predict(
                    context=test_tensor,
                    prediction_length=horizon
                )
                print(f"   ✅ Model prediction test successful: {test_forecast.shape}")
            else:
                test_forecast = pipeline.predict(
                    context=test_tensor,
                    prediction_length=horizon,
                    num_samples=CONFIG.get('num_samples', 100)
                )
                print(f"   ✅ Model prediction test successful")
        except Exception as e:
            print(f"   ❌ Model prediction test failed: {e}")
            print(f"   🚨 This indicates the model pipeline is the issue!")
            return None
            
    except Exception as e:
        print(f"   ❌ First iteration test failed: {e}")
        return None
    
    print(f"\\n🚀 STARTING ROBUST FORECASTING LOOP:")
    
    for i in range(start_idx, end_idx):
        
        # Progress tracking every 25 iterations
        if (i - start_idx) % 25 == 0:
            progress = (i - start_idx) / (end_idx - start_idx) * 100
            print(f"🔧 Progress: {progress:.1f}% ({forecast_count} successes, {error_count} errors)")
        
        try:
            # Step 1: Data extraction with validation
            try:
                context_data = all_returns.iloc[i-context_window:i].values
                actual_returns = all_returns.iloc[i:i+horizon].values
                
                # Robust validation
                if len(context_data) != context_window:
                    raise ValueError(f"Context length {len(context_data)} != {context_window}")
                if len(actual_returns) != horizon:
                    raise ValueError(f"Horizon length {len(actual_returns)} != {horizon}")
                if np.any(np.isnan(context_data)):
                    raise ValueError(f"NaN values in context data")
                if np.any(np.isnan(actual_returns)):
                    raise ValueError(f"NaN values in actual returns")
                
            except Exception as e:
                error_count += 1
                error_types['data_extraction'] = error_types.get('data_extraction', 0) + 1
                if error_count <= 3:  # Show first few errors
                    print(f"❌ Data extraction error at {i}: {e}")
                continue
            
            # Step 2: Tensor creation with validation
            try:
                context_tensor = torch.tensor(context_data, dtype=torch.float32)
                
                # Validate tensor
                if torch.any(torch.isnan(context_tensor)):
                    raise ValueError(f"NaN in tensor")
                if torch.any(torch.isinf(context_tensor)):
                    raise ValueError(f"Inf in tensor")
                
                # Device handling
                device = CONFIG.get('device', 'cpu')
                if device == 'cuda' and torch.cuda.is_available():
                    context_tensor = context_tensor.cuda()
                
            except Exception as e:
                error_count += 1
                error_types['tensor_creation'] = error_types.get('tensor_creation', 0) + 1
                if error_count <= 3:
                    print(f"❌ Tensor creation error at {i}: {e}")
                continue
            
            # Step 3: Model prediction with robust error handling
            try:
                if 'bolt' in model_name:
                    # ChronosBolt models
                    forecast = pipeline.predict(
                        context=context_tensor,
                        prediction_length=horizon
                    )
                    
                    # Extract predictions with validation
                    if len(forecast.shape) == 3:  # (batch, quantiles, horizon)
                        median_idx = forecast.shape[1] // 2
                        predicted_returns = forecast[0, median_idx, :].cpu().numpy()
                    else:
                        predicted_returns = forecast.median(dim=0).values.cpu().numpy()
                        
                else:
                    # Regular Chronos models
                    forecast = pipeline.predict(
                        context=context_tensor,
                        prediction_length=horizon,
                        num_samples=CONFIG.get('num_samples', 100)
                    )
                    
                    if isinstance(forecast, tuple):
                        predicted_returns = forecast[0].median(dim=0).values.cpu().numpy()
                    else:
                        predicted_returns = forecast.median(dim=0).values.cpu().numpy()
                
                # Validate predictions
                if len(predicted_returns) != horizon:
                    raise ValueError(f"Prediction length {len(predicted_returns)} != {horizon}")
                if np.any(np.isnan(predicted_returns)):
                    raise ValueError(f"NaN in predictions")
                if np.any(np.isinf(predicted_returns)):
                    raise ValueError(f"Inf in predictions")
                
            except Exception as e:
                error_count += 1
                error_types['model_prediction'] = error_types.get('model_prediction', 0) + 1
                if error_count <= 3:
                    print(f"❌ Model prediction error at {i}: {e}")
                
                # Clear GPU memory if applicable
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                continue
            
            # Step 4: Store results
            try:
                predictions_list.append(predicted_returns)
                actuals_list.append(actual_returns)
                dates_list.append(all_returns.index[i:i+horizon])
                forecast_count += 1
                
            except Exception as e:
                error_count += 1
                error_types['result_storage'] = error_types.get('result_storage', 0) + 1
                if error_count <= 3:
                    print(f"❌ Result storage error at {i}: {e}")
                continue
            
            # Step 5: Memory management
            if forecast_count % 100 == 0:
                try:
                    # Clear GPU memory periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass  # Non-critical
            
        except Exception as e:
            # Catch-all for unexpected errors
            error_count += 1
            error_types['unexpected'] = error_types.get('unexpected', 0) + 1
            if error_count <= 3:
                print(f"❌ Unexpected error at {i}: {e}")
            continue
        
        # Safety check: prevent infinite error loops
        if error_count > 100 and forecast_count == 0:
            print(f"🚨 Too many errors ({error_count}) with no successes - stopping")
            break
        elif error_count > 500:
            print(f"🚨 Error limit reached ({error_count}) - stopping")
            break
    
    # Final status report
    print(f"\\n🔧 ROBUST FORECAST COMPLETION:")
    print(f"   Successful forecasts: {forecast_count}")
    print(f"   Total errors: {error_count}")
    print(f"   Success rate: {forecast_count/(forecast_count + error_count)*100:.1f}%")
    
    # Error breakdown
    if error_types:
        print(f"\\n❌ ERROR BREAKDOWN:")
        for error_type, count in error_types.items():
            print(f"   {error_type}: {count}")
    
    # Memory cleanup
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    if len(predictions_list) == 0:
        print(f"❌ No successful forecasts generated")
        print(f"🔍 Check error breakdown above for main issues")
        return None
    
    # Process results (same as before but with error handling)
    try:
        all_predictions = np.concatenate(predictions_list)
        all_actuals = np.concatenate(actuals_list)
        
        # Calculate metrics with validation
        if len(all_predictions) == 0 or len(all_actuals) == 0:
            raise ValueError("Empty prediction or actual arrays")
        
        return_mae = np.mean(np.abs(all_predictions - all_actuals))
        return_rmse = np.sqrt(np.mean((all_predictions - all_actuals) ** 2))
        
        pred_directions = np.sign(all_predictions)
        actual_directions = np.sign(all_actuals)
        hit_rate = np.mean(pred_directions == actual_directions)
        
        vol_actual = np.std(all_actuals)
        vol_predicted = np.std(all_predictions)
        vol_ratio = vol_predicted / vol_actual if vol_actual != 0 else np.nan
        
        # Price reconstruction with error handling
        first_predictions = [pred[0] for pred in predictions_list]
        first_actuals = [actual[0] for actual in actuals_list]
        
        predicted_prices = []
        actual_prices = []
        
        for i, (pred_ret, actual_ret) in enumerate(zip(first_predictions, first_actuals)):
            try:
                initial_price = all_prices.iloc[start_idx + i - 1]
                pred_price = initial_price * np.exp(pred_ret)
                actual_price = all_prices.iloc[start_idx + i]
                
                predicted_prices.append(pred_price)
                actual_prices.append(actual_price)
            except Exception as e:
                # Skip problematic price reconstructions
                continue
        
        if len(predicted_prices) > 0:
            price_mae = np.mean(np.abs(np.array(predicted_prices) - np.array(actual_prices)))
            price_mape = np.mean(np.abs((np.array(predicted_prices) - np.array(actual_prices)) / np.array(actual_prices))) * 100
        else:
            price_mae = np.nan
            price_mape = np.nan
        
        print(f"✅ ROBUST COMPLETION: {len(predictions_list)} forecasts")
        print(f"   Return MAE: {return_mae:.6f}, Hit Rate: {hit_rate:.3f}")
        print(f"   Price MAE: ${price_mae:.2f}, MAPE: {price_mape:.2f}%")
        
        return {
            'model': model_name,
            'context_window': context_window,
            'horizon': horizon,
            'return_mae': return_mae,
            'return_rmse': return_rmse,
            'hit_rate': hit_rate,
            'volatility_ratio': vol_ratio,
            'price_mae': price_mae,
            'price_mape': price_mape,
            'num_forecasts': len(predictions_list),
            'total_samples': len(all_predictions),
            'max_possible_forecasts': max_possible_forecasts,
            'data_utilization': len(predictions_list) / max_possible_forecasts,
            'forecast_period_start': all_returns.index[start_idx],
            'forecast_period_end': all_returns.index[start_idx + len(predictions_list) - 1],
            'predictions_list': predictions_list,
            'actuals_list': actuals_list,
            'predicted_prices': predicted_prices,
            'actual_prices': actual_prices,
            'error_count': error_count,
            'error_types': error_types,
            'success_rate': forecast_count/(forecast_count + error_count) if (forecast_count + error_count) > 0 else 0,
            'model_type': 'Zero_Shot_Chronos_UltraRobust'
        }
        
    except Exception as e:
        print(f"❌ Results processing error: {e}")
        return None
'''

print("✅ Ultra-robust forecasting function created")
print("🔧 Key robustness features:")
print("   • Step-by-step validation with detailed error tracking")
print("   • Model prediction testing before main loop")
print("   • Comprehensive error categorization")
print("   • Memory management and cleanup")
print("   • Safety limits to prevent infinite loops")
print("   • Progress reporting every 25 iterations")
print("   • Graceful handling of prediction failures")

if __name__ == "__main__":
    robust_function = create_robust_forecasting_function()
    print("\\n📁 Ready to replace in notebook for debugging")