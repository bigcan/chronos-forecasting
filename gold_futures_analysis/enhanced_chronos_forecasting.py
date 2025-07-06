
import pandas as pd
from chronos import ChronosPipeline
import torch
from gold_futures_analysis.improved_data_preprocessing import improved_data_preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_forecast(
    data, 
    target_col, 
    scaler, 
    model_name="amazon/chronos-t5-small", 
    context_length=63
):
    # Load the Chronos model
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    # Rolling forecast
    predicted_returns_scaled = []
    for i in range(context_length, len(data)):
        context = torch.tensor(data[target_col].iloc[i-context_length:i].values, dtype=torch.float32)
        forecast = pipeline.predict(
            context=context.unsqueeze(0),
            prediction_length=1,
            num_samples=20,
        )
        predicted_returns_scaled.append(forecast[0].mean().item())

    # Inverse transform the predicted returns
    predicted_returns = scaler.inverse_transform(np.array(predicted_returns_scaled).reshape(-1, 1)).flatten()

    # Get the actual prices and previous day's close prices
    actual_prices = data['Close'].iloc[context_length:].values
    prev_close_prices = data['Close'].iloc[context_length-1:-1].values

    # Reconstruct the predicted prices
    predicted_prices = prev_close_prices * (1 + predicted_returns)

    # Calculate metrics
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae_naive = mean_absolute_error(actual_prices[1:], actual_prices[:-1])
    mase = mae / mae_naive
    directional_accuracy = np.mean(np.sign(predicted_prices - prev_close_prices) == np.sign(actual_prices - prev_close_prices)) * 100

    return {
        "model": model_name,
        "context_length": context_length,
        "MAE": mae,
        "RMSE": rmse,
        "MASE": mase,
        "Directional_Accuracy": directional_accuracy
    }

def main():
    # Load the data
    df = pd.read_csv("c:/QuantConnect/Chronos-Forecasting/gold_futures_analysis/GCUSD_MAX_FROM_PERPLEXITY.csv")

    # Preprocess the data using returns
    data, target_col, scaler = improved_data_preprocessing(
        df,
        use_returns=True,
        remove_outliers=True,
        scaling_method='robust'
    )

    results = run_forecast(data, target_col, scaler)

    print(f"MAE: {results['MAE']}")
    print(f"RMSE: {results['RMSE']}")
    print(f"MASE: {results['MASE']}")
    print(f"Directional Accuracy: {results['Directional_Accuracy']:.2f}%")

if __name__ == "__main__":
    main()
