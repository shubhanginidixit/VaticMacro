import pandas as pd
import numpy as np


def create_features(df):
    """
    Create engineered features for inflation forecasting.
    
    Args:
        df: DataFrame with economic indicators
    
    Returns:
        DataFrame with engineered features and renamed CPI column
    """
    df = df.copy()
    
    # Standardize date column name
    if 'observation_date' in df.columns:
        df = df.rename(columns={'observation_date': 'Date'})
    
    # Rename CPI column for consistency
    if 'INDCPIALLMINMEI' in df.columns:
        df = df.rename(columns={'INDCPIALLMINMEI': 'CPI'})
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Lag features (previous 1, 3, 6, 12 months ~ 30, 90, 180, 360 days)
    for col in df.columns:
        if col not in ['Date', 'CPI']:
            for lag in [30, 90, 180, 360]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Rolling averages
    for col in df.columns:
        if col not in ['Date', 'CPI'] and '_lag_' not in col:
            for window in [30, 90, 180]:
                df[f'{col}_rolling_avg_{window}'] = df[col].rolling(window=window).mean()
    
    # Rate of change
    for col in df.columns:
        if col not in ['Date', 'CPI'] and '_lag_' not in col and '_rolling_' not in col:
            df[f'{col}_pct_change'] = df[col].pct_change() * 100
    
    # Fill NaN values from feature engineering (use modern syntax)
    df = df.ffill().bfill()
    
    return df
