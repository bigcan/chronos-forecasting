import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats

def improved_data_preprocessing(df, use_returns=True, remove_outliers=True, 
                              outlier_threshold=3, scaling_method='robust'):
    """
    Enhanced data preprocessing for Chronos models
    """
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Use more data (extend beyond 2020-2021)
    mask = (data['Date'] >= '2018-01-01') & (data['Date'] <= '2023-12-31')
    data = data[mask].reset_index(drop=True)
    
    # Option 1: Use returns instead of raw prices
    if use_returns:
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        target_col = 'Returns'
    else:
        # Option 2: Use log-transformed prices
        data['Log_Close'] = np.log(data['Close'])
        target_col = 'Log_Close'
    
    # Remove outliers
    if remove_outliers:
        # Drop NaN values first
        clean_data = data.dropna(subset=[target_col])
        z_scores = np.abs(stats.zscore(clean_data[target_col]))
        outlier_mask = z_scores < outlier_threshold
        data = clean_data[outlier_mask]
    
    # Apply scaling
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = None
    
    if scaler is not None:
        data[f'Scaled_{target_col}'] = scaler.fit_transform(
            data[target_col].values.reshape(-1, 1)
        ).flatten()
        target_col = f'Scaled_{target_col}'
    
    # Handle missing values
    data = data.dropna().reset_index(drop=True)
    
    return data, target_col, scaler

def create_enhanced_fev_dataset(data, target_col, context_lengths=[30, 60, 90, 120]):
    """
    Create multiple datasets with different context lengths for optimization
    """
    datasets = {}
    
    for context_length in context_lengths:
        records = []
        
        for i in range(context_length, len(data)):
            historical_data = data[target_col].iloc[i-context_length:i]
            target = data[target_col].iloc[i]
            
            record = {
                'unique_id': f'gold_futures_{i}',
                'ds': data['Date'].iloc[i].strftime('%Y-%m-%d'),
                'y': target,
                'historical_data': historical_data.values.tolist(),
                'context_length': context_length,
                'prediction_length': 1
            }
            records.append(record)
        
        datasets[context_length] = records
    
    return datasets