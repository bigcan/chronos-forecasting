import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gold_futures_analysis.enhanced_chronos_forecasting import run_forecast
from gold_futures_analysis.improved_data_preprocessing import improved_data_preprocessing

def main():
    # Load and preprocess the data
    df = pd.read_csv("c:/QuantConnect/Chronos-Forecasting/gold_futures_analysis/GCUSD_MAX_FROM_PERPLEXITY.csv")
    data, target_col, scaler = improved_data_preprocessing(
        df,
        use_returns=True,
        remove_outliers=True,
        scaling_method='robust'
    )

    # Define the parameter grid
    context_lengths = [30, 63, 126, 252]
    model_names = ["amazon/chronos-t5-tiny", "amazon/chronos-t5-small", "amazon/chronos-t5-base"]

    results = []

    # Run the optimization loop
    for model_name in model_names:
        for context_length in context_lengths:
            print(f"Running forecast for {model_name} with context length {context_length}...")
            result = run_forecast(data, target_col, scaler, model_name, context_length)
            results.append(result)
            print(f"Finished forecast for {model_name} with context length {context_length}.")

    # Create and save the results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv("c:/QuantConnect/Chronos-Forecasting/gold_futures_analysis/optimization_results.csv", index=False)
    print("\nOptimization results:")
    print(results_df)

if __name__ == "__main__":
    main()
