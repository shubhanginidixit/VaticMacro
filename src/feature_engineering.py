import pandas as pd


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
    
    # **KEY**: Use PERCENTAGE CHANGES instead of raw values
    # This makes features comparable across different time periods
    indicator_cols = [col for col in df.columns if col not in ['Date', 'CPI']]

    # Create percentage change features (how much did each indicator change?)
    for col in indicator_cols:
        # Different timeframes for momentum
        df[f'{col}_pct_change_30'] = df[col].pct_change(30) * 100    # 30-day %change
        df[f'{col}_pct_change_90'] = df[col].pct_change(90) * 100    # 90-day %change
        df[f'{col}_pct_change_180'] = df[col].pct_change(180) * 100  # 180-day %change

    # Create lag features on percentage changes (recent volatility)
    for col in indicator_cols:
        df[f'{col}_lag_pct_30'] = df[f'{col}_pct_change_30'].shift(30)
        df[f'{col}_lag_pct_90'] = df[f'{col}_pct_change_90'].shift(90)

    # Create rolling averages of percentage changes (average momentum)
    for col in indicator_cols:
        df[f'{col}_rolling_pct_avg_30'] = df[f'{col}_pct_change_30'].rolling(window=30).mean()
        df[f'{col}_rolling_pct_avg_90'] = df[f'{col}_pct_change_90'].rolling(window=90).mean()

    # Create ratios between key indicators (normalized relationships)
    if 'WPIATT01INM661N' in df.columns and 'CPI' in df.columns:
        df['WPI_to_CPI_ratio'] = (df['WPIATT01INM661N'] / df['CPI']) * 100
        df['WPI_to_CPI_pct_change'] = df['WPI_to_CPI_ratio'].pct_change(30) * 100

    if 'DEXINUS' in df.columns and 'Average of DCOILBRENTEU' in df.columns:
        df['oil_to_inr_ratio'] = df['Average of DCOILBRENTEU'] / df['DEXINUS']
        df['oil_to_inr_pct_change'] = df['oil_to_inr_ratio'].pct_change(30) * 100

    # Add CPI lag features (direct momentum of target variable)
    df['CPI_lag_30'] = df['CPI'].shift(30)
    df['CPI_lag_90'] = df['CPI'].shift(90)
    df['CPI_lag_365'] = df['CPI'].shift(365)
    df['CPI_pct_change_30'] = df['CPI'].pct_change(30) * 100
    df['CPI_pct_change_90'] = df['CPI'].pct_change(90) * 100

    # Handle extreme values and negative numbers safely
    # Clip extreme percentage-change values to reduce outlier influence
    pct_cols = [c for c in df.columns if 'pct_change' in c or 'lag_pct' in c or 'rolling_pct_avg' in c]
    if pct_cols:
        # Replace inf/-inf with NaN first, then clip
        df[pct_cols] = df[pct_cols].replace([float('inf'), float('-inf')], float('nan'))
        # Clip to reasonable range: -20 to +20 percentage points
        df[pct_cols] = df[pct_cols].clip(lower=-20.0, upper=20.0)
        # Fill remaining NaN with 0 (no change)
        df[pct_cols] = df[pct_cols].fillna(0)

    df = df.dropna().reset_index(drop=True)

    # Remove original raw indicator columns (keep CPI and engineered features)
    # Keep CPI so train_model can compute year-ago baseline
    raw_cols = [col for col in df.columns if col in indicator_cols]
    if raw_cols:
        df = df.drop(columns=[c for c in raw_cols if c in df.columns])

    # Drop extremely volatile oil-related change features which often spike (e.g., COVID shock)
    volatile_patterns = ['datafilenew(india basket crude oil)', 'DCOILBRENTEU', 'oil_to_inr']
    drop_cols = [c for c in df.columns if any(pat in c for pat in volatile_patterns)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    return df
