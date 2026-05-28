import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import create_features

MODEL_PATH = 'models/best_model.pkl'
DATA_PATH = 'data/inflation_dataset.csv'

artifact = joblib.load(MODEL_PATH)
if isinstance(artifact, dict):
    pipeline = artifact.get('pipeline')
    FEATURE_COLUMNS = artifact.get('feature_columns')
else:
    pipeline = artifact
    FEATURE_COLUMNS = None

COLUMN_MAP = {
    'cpi': 'INDCPIALLMINMEI',
    'wpi': 'WPIATT01INM661N',
    'interest_rate': 'INTDSRINM193N',
    'usd_inr': 'DEXINUS',
    'brent_crude': 'Average of DCOILBRENTEU',
    'gdp_proxy': 'MKTGDPINA646NWDB'
}

# User-provided scenario
s_date = pd.to_datetime('2023-07-01')
val_wpi = 136.3
val_ir = 6.5
val_usd = 83.42
val_brent = 80.92
val_gdp = 3500.0

# Load data

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
last_row = df.iloc[-1]

# Build appended scenario row (based on last_row values)
new_row = last_row.copy()
new_row['Date'] = s_date
new_row[COLUMN_MAP['brent_crude']] = val_brent
new_row[COLUMN_MAP['usd_inr']] = val_usd
new_row[COLUMN_MAP['interest_rate']] = val_ir
new_row[COLUMN_MAP['gdp_proxy']] = val_gdp
new_row[COLUMN_MAP['wpi']] = val_wpi

# Create features on concatenated DF
from pandas import DataFrame

DF = pd.concat([df, DataFrame([new_row])], ignore_index=True)
featured = create_features(DF)

# Select the appended scenario row (prefer last occurrence)
scenario_row = featured.loc[featured['Date'] == s_date].tail(1)
if scenario_row.empty:
    scenario_row = featured.iloc[[-1]]

pred_X = scenario_row.drop(['Date', 'CPI'], axis=1, errors='ignore')
if FEATURE_COLUMNS:
    pred_X = pred_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)

print('\n=== Debug Scenario 2023-07-01 ===')
print('pred_X shape:', pred_X.shape)
print('Any NaNs:', pred_X.isna().any().any())
print('Sample features (first 20):')
print(pred_X.iloc[0,:20].to_dict())

pred = pipeline.predict(pred_X)[0]
print('\nPredicted YoY Inflation (%):', round(float(pred), 2))
