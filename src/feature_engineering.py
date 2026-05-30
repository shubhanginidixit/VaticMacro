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
    
    # Drop rows with NaN from shifts/rolling
    # Clip extreme percentage-change values to reduce outlier influence
    pct_cols = [c for c in df.columns if 'pct_change' in c or 'lag_pct' in c or 'rolling_pct_avg' in c]
    if pct_cols:
        # Use tighter clipping to reduce influence of extreme policy or market shocks
        df[pct_cols] = df[pct_cols].clip(lower=-25.0, upper=25.0)

    df = df.dropna().reset_index(drop=True)

    # Remove original raw indicator columns (keep only engineered percent-change and ratio features)
    # This prevents large-magnitude raw features (e.g., GDP levels) from dominating the model
    raw_cols = [col for col in df.columns if col in indicator_cols]
    if raw_cols:
        df = df.drop(columns=raw_cols)

    # Drop extremely volatile oil-related change features which often spike (e.g., COVID shock)
    volatile_patterns = ['datafilenew(india basket crude oil)', 'DCOILBRENTEU', 'oil_to_inr']
    drop_cols = [c for c in df.columns if any(pat in c for pat in volatile_patterns)]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    return df
