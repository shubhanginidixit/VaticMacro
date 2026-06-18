import pandas as pd
import numpy as np


# Only keep columns that have real monthly variation (no flat/synthetic data)
KEEP_COLUMNS = [
    'INDCPIALLMINMEI',   # CPI (will be renamed to CPI)
    'WPIATT01INM661N',   # WPI
    'INTDSRINM193N',     # Interest Rate
    'DEXINUS',           # USD/INR
    'Average of DCOILBRENTEU',  # Brent Crude
]


def create_features(df):
    """
    Create engineered features using percentage changes, ratios, and z-scores.
    ~28 features from 7 indicators (5 base + 2 FRED) + calendar + AR lags.
    """
    df = df.copy()

    if 'observation_date' in df.columns:
        df = df.rename(columns={'observation_date': 'Date'})

    if 'INDCPIALLMINMEI' in df.columns:
        df = df.rename(columns={'INDCPIALLMINMEI': 'CPI'})

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Calendar features
    df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)

    # YoY inflation and lags (autoregressive features)
    if 'CPI' in df.columns:
        cpi_pct = df['CPI'].pct_change(12) * 100
        df['inflation_lag_1m'] = cpi_pct.shift(1)
        df['inflation_lag_3m'] = cpi_pct.shift(3)
        df['inflation_lag_6m'] = cpi_pct.shift(6)
        df['inflation_lag_12m'] = cpi_pct.shift(12)
        df['inflation_lag_2m'] = cpi_pct.shift(2)
        df['inflation_lag_9m'] = cpi_pct.shift(9)
        df['inflation_acceleration'] = cpi_pct - cpi_pct.shift(1)

    # Percentage changes for key indicators (1m and 12m only)
    indicator_cols = [col for col in KEEP_COLUMNS if col in df.columns and col != 'CPI']

    for col in indicator_cols:
        df[f'{col}_pct_1m'] = df[col].pct_change(1) * 100
        df[f'{col}_pct_12m'] = df[col].pct_change(12) * 100
        df[f'{col}_rolling_6m'] = df[f'{col}_pct_1m'].rolling(6).mean()

    # Ratios between key indicators
    if 'WPIATT01INM661N' in df.columns and 'CPI' in df.columns:
        df['WPI_CPI_ratio'] = (df['WPIATT01INM661N'] / df['CPI']) * 100
        df['WPI_CPI_pct_1m'] = df['WPI_CPI_ratio'].pct_change(1) * 100

    if 'DEXINUS' in df.columns and 'Average of DCOILBRENTEU' in df.columns:
        df['oil_inr_ratio'] = df['Average of DCOILBRENTEU'] / df['DEXINUS']
        df['oil_inr_pct_1m'] = df['oil_inr_ratio'].pct_change(1) * 100
        # Brent crude volatility: 12m rolling std of monthly pct changes
        df['brent_vol_12m'] = df['Average of DCOILBRENTEU'].pct_change(1).rolling(12).std() * 100
        # INR momentum: 3m change in exchange rate
        df['inr_momentum_3m'] = df['DEXINUS'].diff(3)

    # WPI-CPI spread (divergence between wholesale and consumer prices)
    if 'WPIATT01INM661N' in df.columns and 'CPI' in df.columns:
        df['wpi_cpi_spread'] = df['WPIATT01INM661N'] - df['CPI']

    # Interest rate direction: 3m change in policy rate
    if 'INTDSRINM193N' in df.columns:
        df['rate_direction_3m'] = df['INTDSRINM193N'].diff(3)

    # Industrial Production: z-score (deviation from 24m trend) and 6m smoothed level
    if 'INDPRINTO01GYSAM' in df.columns:
        iip = df['INDPRINTO01GYSAM']
        iip_mean_24m = iip.rolling(24, min_periods=12).mean()
        iip_std_24m = iip.rolling(24, min_periods=12).std()
        df['iip_zscore_24m'] = (iip - iip_mean_24m) / iip_std_24m.clip(lower=1.0)
        df['iip_smooth_6m'] = iip.rolling(6).mean()

    # Trade Balance: pct change of 3m moving average (smooth out monthly noise)
    if 'XTNTVA01INM667N' in df.columns:
        tb = df['XTNTVA01INM667N']
        tb_ma3 = tb.rolling(3).mean()
        df['trade_ma3_pct_12m'] = tb_ma3.pct_change(12) * 100
        df['trade_ma3_pct_1m'] = tb_ma3.pct_change(1) * 100

    # Clip extreme values (wider range to preserve crisis-period signal)
    pct_cols = [c for c in df.columns if 'pct' in c or 'lag_' in c or 'rolling_' in c]
    if pct_cols:
        df[pct_cols] = df[pct_cols].clip(lower=-50.0, upper=50.0)

    # Current YoY inflation (the target basis)
    if 'CPI' in df.columns:
        df['current_inflation_yoy'] = df['CPI'].pct_change(12) * 100

    df = df.dropna().reset_index(drop=True)

    # Drop raw indicator columns (keep only engineered features + Date + target)
    known_raw = ['INDCPIALLMINMEI', 'WPIATT01INM661N', 'INTDSRINM193N', 'DEXINUS',
                 'Average of DCOILBRENTEU', 'INDPRINTO01GYSAM', 'XTNTVA01INM667N',
                 'INDIRLTLT01STM', 'INDLOLITOAASTSAM', 'MABMM301INM189N']
    df = df.drop(columns=[c for c in known_raw if c in df.columns], errors='ignore')

    return df
