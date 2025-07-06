# ZERO-SHOT ROLLING WINDOW FORECASTING - FIXED IMPLEMENTATION
print("🔧 Implementing FIXED Zero-Shot Rolling Window Forecasting...")
print("✅ Chronos models are pre-trained and don't need training on specific data!")
print("🎯 Using ALL available data for maximum statistical robustness")
print("🔧 DEBUGGING ENABLED: Will show detailed progress and error information")

def zero_shot_rolling_forecast(model_name, pipeline, context_window, horizon, max_samples=None):
    """
    FIXED: Perform zero-shot rolling window forecasting with comprehensive debugging
    
    Args:
        model_name: Name of the model
        pipeline: Loaded Chronos pipeline (pre-trained, ready for zero-shot)
        context_window: Days of context to use for each prediction
        horizon: Days ahead to predict
        max_samples: Maximum number of forecasts to make (None = use all available data)
        
    Returns:
        Dictionary with predictions, actuals, and metrics
    """
    print(f"🔧 FIXED FORECAST: {model_name}, Context: {context_window}, Horizon: {horizon}")
    
    predictions_list = []
    actuals_list = []
    dates_list = []
    
    # Use ALL available returns data - no artificial train/test split needed for zero-shot
    all_returns = close_returns  # Complete 2020-2023 dataset
    all_prices = df['Close']     # Corresponding prices for reconstruction
    
    # Start forecasting as soon as we have enough context
    start_idx = context_window
    
    # Calculate maximum possible forecasts
    max_possible_forecasts = len(all_returns) - start_idx - horizon + 1
    
    if max_samples is None:
        # Use ALL available data for maximum statistical power
        end_idx = len(all_returns) - horizon + 1
        actual_forecasts = max_possible_forecasts
    else:
        # Use specified limit
        end_idx = min(start_idx + max_samples, len(all_returns) - horizon + 1)
        actual_forecasts = min(max_samples, max_possible_forecasts)
    
    print(f"🔧 DEBUG PARAMETERS:")
    print(f"   Data length: {len(all_returns)}")
    print(f"   Start index: {start_idx}")
    print(f"   End index: {end_idx}")
    print(f"   Expected iterations: {end_idx - start_idx}")
    print(f"   Max possible forecasts: {max_possible_forecasts}")
    print(f"   Target forecasts: {actual_forecasts}")
    
    forecast_count = 0
    error_count = 0
    
    for i in range(start_idx, end_idx):
        
        # Progress tracking EVERY 50 iterations (not every 100)
        if (i - start_idx) % 50 == 0:
            progress = (i - start_idx) / (end_idx - start_idx) * 100
            print(f"🔧 Progress: {progress:.1f}% ({i - start_idx}/{end_idx - start_idx}) forecasts")
        
        try:
            # Get context data (rolling window) - just the lookback, no training needed
            context_data = all_returns.iloc[i-context_window:i].values
            
            # Validate context data
            if len(context_data) != context_window:
                print(f"❌ Context data length error at step {i}: {len(context_data)} != {context_window}")
                error_count += 1
                continue
            
            # Get actual future returns for evaluation
            actual_returns = all_returns.iloc[i:i+horizon].values
            
            # Validate actual returns
            if len(actual_returns) != horizon:
                print(f"❌ Actual returns length error at step {i}: {len(actual_returns)} != {horizon}")
                error_count += 1
                continue
            
            # Prepare input tensor for zero-shot prediction
            context_tensor = torch.tensor(context_data, dtype=torch.float32)
            if CONFIG['device'] == 'cuda':
                context_tensor = context_tensor.cuda()
            
            # Generate zero-shot prediction using pre-trained Chronos
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
            
            # Store results
            predictions_list.append(predicted_returns)
            actuals_list.append(actual_returns)
            dates_list.append(all_returns.index[i:i+horizon])
            
            forecast_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"❌ ERROR at step {i}: {e}")
            
            # Log detailed error information
            print(f"   Context window: {context_window}, Horizon: {horizon}")
            print(f"   Data slice: [{i-context_window}:{i}] and [{i}:{i+horizon}]")
            print(f"   Data length: {len(all_returns)}")
            
            # Continue with next iteration instead of breaking
            if error_count > 50:  # Prevent infinite error loops
                print(f"🚨 Too many errors ({error_count}), stopping forecasting")
                break
            continue
    
    print(f"🔧 FORECAST COMPLETION:")
    print(f"   Successful forecasts: {forecast_count}")
    print(f"   Errors encountered: {error_count}")
    if (forecast_count + error_count) > 0:
        print(f"   Success rate: {forecast_count/(forecast_count + error_count)*100:.1f}%")
    
    if len(predictions_list) == 0:
        print(f"❌ No successful forecasts generated")
        return None
    
    # Convert to arrays for analysis
    all_predictions = np.concatenate(predictions_list)
    all_actuals = np.concatenate(actuals_list)
    
    # Calculate return metrics
    return_mae = np.mean(np.abs(all_predictions - all_actuals))
    return_rmse = np.sqrt(np.mean((all_predictions - all_actuals) ** 2))
    
    # Directional accuracy (hit rate)
    pred_directions = np.sign(all_predictions)
    actual_directions = np.sign(all_actuals)
    hit_rate = np.mean(pred_directions == actual_directions)
    
    # Volatility metrics
    vol_actual = np.std(all_actuals)
    vol_predicted = np.std(all_predictions)
    vol_ratio = vol_predicted / vol_actual if vol_actual != 0 else np.nan
    
    # Price reconstruction for the first prediction of each forecast
    first_predictions = [pred[0] for pred in predictions_list]
    first_actuals = [actual[0] for actual in actuals_list]
    
    # Reconstruct prices from log returns using actual price series
    predicted_prices = []
    actual_prices = []
    
    for i, (pred_ret, actual_ret) in enumerate(zip(first_predictions, first_actuals)):
        # Get initial price from the day before prediction
        initial_price = all_prices.iloc[start_idx + i - 1]
        
        # Predicted and actual prices
        pred_price = initial_price * np.exp(pred_ret)
        actual_price = all_prices.iloc[start_idx + i]  # Actual next day price
        
        predicted_prices.append(pred_price)
        actual_prices.append(actual_price)
    
    # Price-based metrics  
    price_mae = np.mean(np.abs(np.array(predicted_prices) - np.array(actual_prices)))
    price_mape = np.mean(np.abs((np.array(predicted_prices) - np.array(actual_prices)) / np.array(actual_prices))) * 100
    
    print(f"✅ COMPLETED: {len(predictions_list)} forecasts")
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
        'success_rate': forecast_count/(forecast_count + error_count) if (forecast_count + error_count) > 0 else 0,
        'model_type': 'Zero_Shot_Chronos_Fixed'
    }

# Run ZERO-SHOT rolling window forecasting with FIXED DEBUGGING
zero_shot_results = []

print(f"\n🔧 Starting FIXED Zero-Shot Rolling Window Forecasting...")
print(f"🎯 Using pre-trained Chronos models - no training on gold data needed!")
print(f"📊 Models: {len(models)}, Windows: {len(CONFIG['context_windows'])}, Horizons: {len(CONFIG['prediction_horizons'])}")
print(f"🔧 DEBUG MODE: Comprehensive error tracking and progress reporting enabled")

experiment_count = 0
total_experiments = len(models) * len(CONFIG['context_windows']) * len(CONFIG['prediction_horizons'])

for model_name, pipeline in models.items():
    for context_window in CONFIG['context_windows']:
        for horizon in CONFIG['prediction_horizons']:
            
            experiment_count += 1
            print(f"\n[{experiment_count}/{total_experiments}] 🔧 DEBUGGING: {model_name} - Context: {context_window} - Horizon: {horizon}")
            
            # Check if we have enough data for context window
            if len(close_returns) < context_window + horizon:
                print(f"  ⚠️ Insufficient data for context window {context_window}")
                continue
            
            result = zero_shot_rolling_forecast(
                model_name=model_name,
                pipeline=pipeline, 
                context_window=context_window,
                horizon=horizon,
                max_samples=None  # Use ALL available data
            )
            
            if result is not None:
                zero_shot_results.append(result)
                print(f"  ✅ Success: {result['num_forecasts']} forecasts, {result['hit_rate']:.1%} hit rate")
            else:
                print(f"  ❌ Failed to generate forecasts - check debug output above")

print(f"\n🔧 FIXED Zero-Shot Rolling Window Forecasting Complete!")
print(f"📊 Generated {len(zero_shot_results)} successful experiment results")

if len(zero_shot_results) > 0:
    total_forecasts = sum(r['num_forecasts'] for r in zero_shot_results)
    total_samples = sum(r['total_samples'] for r in zero_shot_results)
    max_possible_total = sum(r['max_possible_forecasts'] for r in zero_shot_results)
    avg_utilization = np.mean([r['data_utilization'] for r in zero_shot_results])
    avg_success_rate = np.mean([r['success_rate'] for r in zero_shot_results])
    
    print(f"📈 FIXED Results Summary:")
    print(f"   Total forecasts: {total_forecasts:,} (vs previous ~72)")
    print(f"   Total prediction samples: {total_samples:,}")
    print(f"   Maximum possible forecasts: {max_possible_total:,}")
    print(f"   Average data utilization: {avg_utilization:.1%}")
    print(f"   Average success rate: {avg_success_rate:.1%}")
    
    # Show if we achieved the expected improvements
    if total_forecasts > 1000:
        print(f"✅ SUCCESS: Massive improvement in forecast count!")
        print(f"📊 Statistical power: Now adequate for robust analysis")
    elif total_forecasts > 100:
        print(f"✅ PARTIAL SUCCESS: Significant improvement but still room for optimization")
    else:
        print(f"⚠️ LIMITED SUCCESS: Some improvement but issue may persist")
        print(f"   Check debug output for specific error patterns")
    
else:
    print("❌ No successful zero-shot forecasts generated")
    print("🔧 Check debug output above for specific error patterns")
    print("   Common issues: GPU memory, tensor compatibility, model loading")

print(f"\n🔧 DEBUG SUMMARY:")
print(f"✅ Fixed forecasting function implemented with:")
print(f"   • Comprehensive error logging")
print(f"   • Data validation at each step")
print(f"   • Progress tracking every 50 iterations")
print(f"   • Error count limiting (max 50 per config)")
print(f"   • Success rate calculation")
print(f"   • Detailed debug parameter output")