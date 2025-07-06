#!/usr/bin/env python3
"""
Debug script to identify why only 1-7 forecasts are generated instead of 700-900+
"""
import pandas as pd
import numpy as np

# Load the data to simulate the issue
data_path = '../gold_futures_analysis/GCUSD_MAX_FROM_PERPLEXITY.csv'
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Filter to 2020-2023 period
start_date = '2020-01-01'
end_date = '2023-12-31'
mask = (df.index >= start_date) & (df.index <= end_date)
df = df[mask]

# Calculate log returns
def calculate_log_returns_safe(prices, periods=1):
    return np.log(prices / prices.shift(periods)).dropna()

close_returns = calculate_log_returns_safe(df['Close'])

print("🔍 DEBUGGING SAMPLE SIZE ISSUE")
print("=" * 50)

print(f"📊 Dataset Information:")
print(f"   Total days (2020-2023): {len(df)}")
print(f"   Log returns available: {len(close_returns)}")
print(f"   Date range: {close_returns.index.min().date()} to {close_returns.index.max().date()}")

# Test forecasting loop parameters
configs = [
    (63, 1), (63, 3), (63, 7),
    (126, 1), (126, 3), (126, 7),
    (252, 1), (252, 3), (252, 7)
]

print(f"\n📈 Expected Forecast Counts:")
print(f"{'Context':<8} {'Horizon':<8} {'Expected':<10} {'Actual Pattern'}")
print("-" * 45)

for context_window, horizon in configs:
    # Calculate expected forecasts
    start_idx = context_window
    end_idx = len(close_returns) - horizon + 1
    expected_forecasts = end_idx - start_idx
    
    # Determine actual pattern from results
    if horizon == 1:
        actual_pattern = "1 (0% hit rate)"
    elif horizon == 3:
        actual_pattern = "3 (66.7% hit rate)"
    elif horizon == 7:
        actual_pattern = "7 (71.4% hit rate)"
    
    print(f"{context_window:<8} {horizon:<8} {expected_forecasts:<10} {actual_pattern}")

print(f"\n🚨 CRITICAL DISCREPANCY ANALYSIS:")
print(f"   Expected total forecasts: ~15,000+")
print(f"   Actual total forecasts: ~72")
print(f"   Utilization rate: <0.5%")

print(f"\n🔍 HYPOTHESIS: Loop Termination Issues")
print(f"   1. Early break/return in forecasting loop")
print(f"   2. Exception handling causing silent failures")
print(f"   3. Memory issues with tensor operations")
print(f"   4. Incorrect data indexing/slicing")

print(f"\n📝 LIKELY ROOT CAUSES:")
print(f"   • Tensor creation failures (GPU memory)")
print(f"   • Model prediction timeouts")
print(f"   • Array indexing out of bounds")
print(f"   • Silent exceptions in prediction pipeline")

# Simulate the array slicing to check for issues
print(f"\n🧪 ARRAY SLICING TEST:")
context_window = 63
horizon = 1
start_idx = context_window

for i in range(start_idx, start_idx + 10):  # Test first 10 iterations
    try:
        context_data = close_returns.iloc[i-context_window:i].values
        actual_returns = close_returns.iloc[i:i+horizon].values
        
        print(f"   Step {i}: Context shape {context_data.shape}, Actual shape {actual_returns.shape}")
        
        if len(context_data) != context_window:
            print(f"     ❌ Context data length mismatch: {len(context_data)} != {context_window}")
        if len(actual_returns) != horizon:
            print(f"     ❌ Actual returns length mismatch: {len(actual_returns)} != {horizon}")
            
    except Exception as e:
        print(f"     ❌ Error at step {i}: {e}")
        break

print(f"\n🎯 RECOMMENDED FIXES:")
print(f"   1. Add detailed error logging in forecasting loop")
print(f"   2. Check tensor memory allocation and device compatibility")
print(f"   3. Validate array bounds before slicing")
print(f"   4. Add progress tracking every iteration (not every 100)")
print(f"   5. Test with smaller context windows first")

print(f"\n💡 NEXT STEPS:")
print(f"   1. Run notebook cell-by-cell with debugging enabled")
print(f"   2. Check model loading and device compatibility")
print(f"   3. Test with single configuration first")
print(f"   4. Monitor memory usage during execution")