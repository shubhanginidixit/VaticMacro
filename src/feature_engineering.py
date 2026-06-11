import pandas as pd
import numpy as np


def create_features(df):
    """
    Create engineered features using PERCENTAGE CHANGES and RATIOS.
    This makes features scale-independent so model generalizes across years.
    
    Key insight: Train on 2000-2022 with percentage changes → predicts 2026+ correctly
    
    Args:
        df: DataFrame with economic indicators
    
    Returns:
        DataFrame with engineered features (percentage changes, ratios, standardized)
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
    
    # Add calendar features (seasonality)
    df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
    
    # Add inflation lag features (autoregressive)
    if 'CPI' in df.columns:
        cpi_pct = df['CPI'].pct_change(12) * 100
        df['inflation_lag_1m'] = cpi_pct.shift(1)
        df['inflation_lag_3m'] = cpi_pct.shift(3)
        df['inflation_lag_6m'] = cpi_pct.shift(6)
        df['inflation_lag_12m'] = cpi_pct.shift(12)
    
    # **KEY**: Use PERCENTAGE CHANGES instead of raw values
    # This makes features comparable across different time periods
    indicator_cols = [col for col in df.columns if col not in ['Date', 'CPI']]

    # Create percentage change features (how much did each indicator change?)
    # Since data is now strictly monthly, 1 row = 1 month
    for col in indicator_cols:
        df[f'{col}_pct_change_1m'] = df[col].pct_change(1) * 100
        df[f'{col}_pct_change_3m'] = df[col].pct_change(3) * 100
        df[f'{col}_pct_change_12m'] = df[col].pct_change(12) * 100
        
        # Rolling statistics on the 1m pct change (momentum & volatility)
        df[f'{col}_pct_1m_rolling_mean_6m'] = df[f'{col}_pct_change_1m'].rolling(6).mean()
        df[f'{col}_pct_1m_rolling_std_6m'] = df[f'{col}_pct_change_1m'].rolling(6).std()
    
    # Create ratios between key indicators (normalized relationships)
    if 'WPIATT01INM661N' in df.columns and 'CPI' in df.columns:
        df['WPI_to_CPI_ratio'] = (df['WPIATT01INM661N'] / df['CPI']) * 100
        df['WPI_to_CPI_pct_change_1m'] = df['WPI_to_CPI_ratio'].pct_change(1) * 100
    
    if 'DEXINUS' in df.columns and 'Average of DCOILBRENTEU' in df.columns:
        df['oil_to_inr_ratio'] = df['Average of DCOILBRENTEU'] / df['DEXINUS']
        df['oil_to_inr_pct_change_1m'] = df['oil_to_inr_ratio'].pct_change(1) * 100
    
    # Drop rows with NaN from shifts/rolling
    # Clip extreme percentage-change values to reduce outlier influence
    pct_cols = [c for c in df.columns if 'pct_change' in c or 'lag_' in c or 'rolling_' in c]
    if pct_cols:
        # Use tighter clipping to reduce influence of extreme policy or market shocks
        df[pct_cols] = df[pct_cols].clip(lower=-25.0, upper=25.0)

    df = df.dropna().reset_index(drop=True)

    # Drop absolute raw indicator columns to rely strictly on percentage changes and ratios.
    # This prevents tree-based models from failing when future data exceeds historical absolute limits.
    df = df.drop(columns=indicator_cols, errors='ignore')
    
    return df
