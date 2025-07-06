# MINIMAL CRITICAL FIX - Replace problematic prediction section
# This targets the exact issue causing 1-7 forecasts instead of 700-900+

# The issue is likely in the model prediction pipeline
# Let's add critical debugging right at the prediction step

MINIMAL_FIX = '''
            # Generate zero-shot prediction using pre-trained Chronos
            try:
                if 'bolt' in model_name:
                    # ChronosBolt models - zero-shot ready
                    forecast = pipeline.predict(
                        context=context_tensor,
                        prediction_length=horizon
                    )
                    
                    # Extract median quantile prediction
                    if len(forecast.shape) == 3:  # (batch, quantiles, horizon)
                        median_idx = forecast.shape[1] // 2
                        predicted_returns = forecast[0, median_idx, :].cpu().numpy()
                    else:
                        predicted_returns = forecast.median(dim=0).values.cpu().numpy()
                else:
                    # Regular Chronos models - zero-shot ready
                    forecast = pipeline.predict(
                        context=context_tensor,
                        prediction_length=horizon,
                        num_samples=CONFIG['num_samples']
                    )
                    
                    if isinstance(forecast, tuple):
                        predicted_returns = forecast[0].median(dim=0).values.cpu().numpy()
                    else:
                        predicted_returns = forecast.median(dim=0).values.cpu().numpy()
                
                # CRITICAL: Validate prediction success
                if predicted_returns is None or len(predicted_returns) != horizon:
                    raise ValueError(f"Invalid prediction: {predicted_returns}")
                
                # Store results
                predictions_list.append(predicted_returns)
                actuals_list.append(actual_returns)
                dates_list.append(all_returns.index[i:i+horizon])
                
                forecast_count += 1
                
                # CRITICAL: Debug output every iteration for first 10
                if forecast_count <= 10:
                    print(f"    ✅ Iteration {i}: Success {forecast_count}, Shape: {predicted_returns.shape}")
                
            except Exception as prediction_error:
                print(f"    ❌ PREDICTION ERROR at step {i}: {prediction_error}")
                print(f"       Context shape: {context_tensor.shape}")
                print(f"       Model: {model_name}")
                print(f"       Horizon: {horizon}")
                
                # CRITICAL: Continue instead of breaking the loop
                continue
'''

print("🎯 MINIMAL CRITICAL FIX identified:")
print("   • Add detailed debugging at prediction step")
print("   • Validate prediction results immediately")
print("   • Show success/failure for first 10 iterations")
print("   • Continue on error instead of silent failure")
print("   • Track specific prediction errors")

print("\\n🔧 This should reveal exactly where and why the loop terminates early")
print("📁 Apply this to the notebook forecasting function")