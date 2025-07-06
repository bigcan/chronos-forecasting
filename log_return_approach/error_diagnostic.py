#!/usr/bin/env python3
"""
Comprehensive error diagnostic for forecasting issues
"""
import pandas as pd
import numpy as np

print('🔍 COMPREHENSIVE FORECASTING ERROR DIAGNOSTIC')
print('=' * 60)

# Load the data to simulate potential errors
try:
    data_path = '../gold_futures_analysis/GCUSD_MAX_FROM_PERPLEXITY.csv'
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Filter to 2020-2023
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df[mask]
    
    close_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    
    print(f'✅ Data loaded successfully')
    print(f'   Data shape: {df.shape}')
    print(f'   Returns shape: {close_returns.shape}')
    
except Exception as e:
    print(f'❌ Data loading error: {e}')
    exit(1)

# Test common error scenarios
print(f'\n🧪 TESTING COMMON ERROR SCENARIOS:')

# Error 1: Array indexing out of bounds
print(f'\n1. Array indexing bounds test:')
context_window = 63
horizon = 1
start_idx = context_window

error_count = 0
for i in range(start_idx, start_idx + 10):
    try:
        context_data = close_returns.iloc[i-context_window:i].values
        actual_returns = close_returns.iloc[i:i+horizon].values
        
        if len(context_data) != context_window:
            print(f'   ❌ Context length mismatch at {i}: {len(context_data)} != {context_window}')
            error_count += 1
        if len(actual_returns) != horizon:
            print(f'   ❌ Horizon length mismatch at {i}: {len(actual_returns)} != {horizon}')
            error_count += 1
        if i == start_idx and error_count == 0:
            print(f'   ✅ Array indexing works correctly')
            
    except Exception as e:
        print(f'   ❌ Indexing error at {i}: {e}')
        error_count += 1

# Error 2: Data quality issues
print(f'\n2. Data quality test:')
nan_count = close_returns.isna().sum()
inf_count = np.isinf(close_returns).sum()
zero_count = (close_returns == 0).sum()
extreme_count = (np.abs(close_returns) > 0.5).sum()

print(f'   NaN values: {nan_count}')
print(f'   Inf values: {inf_count}')
print(f'   Zero values: {zero_count}')
print(f'   Extreme values (>50%): {extreme_count}')

if nan_count > 0 or inf_count > 0:
    print(f'   ❌ Data quality issues detected')
    # Show where the issues are
    if nan_count > 0:
        nan_indices = close_returns[close_returns.isna()].index
        print(f'   NaN at: {nan_indices[:5].tolist()}...')
    if inf_count > 0:
        inf_indices = close_returns[np.isinf(close_returns)].index
        print(f'   Inf at: {inf_indices[:5].tolist()}...')
else:
    print(f'   ✅ Data quality OK')

# Error 3: Memory/tensor creation issues
print(f'\n3. Tensor creation test:')
try:
    import torch
    test_data = close_returns.iloc[start_idx-context_window:start_idx].values
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    print(f'   ✅ Tensor creation works: shape {test_tensor.shape}')
    print(f'   ✅ Tensor dtype: {test_tensor.dtype}')
    print(f'   ✅ Tensor device: {test_tensor.device}')
    
    # Test GPU if available
    if torch.cuda.is_available():
        try:
            test_tensor_gpu = test_tensor.cuda()
            print(f'   ✅ GPU tensor works: device {test_tensor_gpu.device}')
        except Exception as e:
            print(f'   ❌ GPU tensor error: {e}')
    else:
        print(f'   ℹ️  CPU only (no GPU available)')
        
except ImportError:
    print(f'   ❌ PyTorch not available')
except Exception as e:
    print(f'   ❌ Tensor creation error: {e}')

# Error 4: Loop bounds calculation
print(f'\n4. Loop bounds test:')
bounds_errors = 0
for context in [63, 126, 252]:
    for horizon in [1, 3, 7]:
        start_idx_test = context
        end_idx_test = len(close_returns) - horizon + 1
        expected_iterations = end_idx_test - start_idx_test
        
        if expected_iterations <= 0:
            print(f'   ❌ Invalid loop bounds: Context {context}, Horizon {horizon}')
            bounds_errors += 1
        elif expected_iterations < 10:
            print(f'   ⚠️  Very few iterations: Context {context}, Horizon {horizon} -> {expected_iterations}')
        else:
            print(f'   ✅ Context {context}, Horizon {horizon} -> {expected_iterations} iterations')

# Error 5: Common Python/Pandas issues
print(f'\n5. Common Python issues test:')
try:
    # Test datetime operations
    test_date = close_returns.index[0]
    test_str = test_date.strftime('%Y-%m-%d')
    print(f'   ✅ Datetime operations work: {test_str}')
    
    # Test array concatenation
    test_arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    concat_result = np.concatenate(test_arrays)
    print(f'   ✅ Array concatenation works: {concat_result.shape}')
    
    # Test dictionary operations
    test_dict = {'test': [1, 2, 3]}
    test_value = test_dict['test'][0]
    print(f'   ✅ Dictionary operations work')
    
except Exception as e:
    print(f'   ❌ Python operations error: {e}')

# Error 6: Device and memory issues
print(f'\n6. Device and memory test:')
try:
    import torch
    
    # Check available memory
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2  # MB
        memory_cached = torch.cuda.memory_reserved(current_device) / 1024**2  # MB
        
        print(f'   GPU devices available: {device_count}')
        print(f'   Current device: {current_device} ({device_name})')
        print(f'   Memory allocated: {memory_allocated:.1f} MB')
        print(f'   Memory cached: {memory_cached:.1f} MB')
        
        # Test large tensor creation
        try:
            large_tensor = torch.randn(1000, 1000, device='cuda')
            print(f'   ✅ Large GPU tensor creation works')
            del large_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'   ❌ Large GPU tensor error: {e}')
    else:
        print(f'   ℹ️  No GPU available')
        
except Exception as e:
    print(f'   ❌ Device test error: {e}')

# Summary
print(f'\n📊 ERROR DIAGNOSTIC SUMMARY:')
print(f'   Array indexing errors: {error_count}')
print(f'   Data quality issues: {nan_count + inf_count}')
print(f'   Loop bounds errors: {bounds_errors}')

print(f'\n🎯 MOST LIKELY ERROR SOURCES:')
if error_count > 0:
    print(f'   🚨 Array indexing issues detected')
if nan_count + inf_count > 0:
    print(f'   🚨 Data quality issues detected')
if bounds_errors > 0:
    print(f'   🚨 Loop bounds calculation errors detected')

print(f'\n💡 RECOMMENDED FIXES:')
print(f'   1. Add comprehensive data validation before tensor creation')
print(f'   2. Implement robust error handling with detailed logging')
print(f'   3. Add device compatibility checks')
print(f'   4. Use try-except for each iteration with continue (not break)')
print(f'   5. Add memory monitoring and cleanup')

print(f'\n🔧 NEXT STEPS:')
print(f'   1. Apply these fixes to the forecasting function')
print(f'   2. Run with detailed debugging on single configuration first')
print(f'   3. Monitor specific error patterns and frequencies')
print(f'   4. Gradually scale up to full analysis')