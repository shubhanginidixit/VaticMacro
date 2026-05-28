import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import create_features

MODEL_PATH = 'models/best_model.pkl'
DATA_PATH = 'data/inflation_dataset.csv'

print('Workspace:', os.getcwd())

artifact = joblib.load(MODEL_PATH)
print('Loaded artifact type:', type(artifact))
if isinstance(artifact, dict):
    pipeline = artifact.get('pipeline')
    FEATURE_COLUMNS = artifact.get('feature_columns')
    print('Model name:', artifact.get('model_name'))
else:
    pipeline = artifact
    FEATURE_COLUMNS = None

print('Num feature columns:', len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 'None')

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

COLUMN_MAP = {
    'cpi': 'INDCPIALLMINMEI',
    'wpi': 'WPIATT01INM661N',
    'interest_rate': 'INTDSRINM193N',
    'usd_inr': 'DEXINUS',
    'brent_crude': 'Average of DCOILBRENTEU',
    'gdp_proxy': 'MKTGDPINA646NWDB'
}

last_row = df.iloc[-1]
last_date = df['Date'].max()

scenarios = [
    {'name':'default','date':None,'wpi':136.3,'ir':6.5,'usd':83.42,'brent':80.92,'gdp':3500.0},
    {'name':'low-wpi','date':None,'wpi':50.0,'ir':6.5,'usd':83.42,'brent':80.92,'gdp':3500.0},
    {'name':'high-wpi','date':None,'wpi':300.0,'ir':6.5,'usd':83.42,'brent':80.92,'gdp':3500.0},
    {'name':'low-gdp','date':None,'wpi':136.3,'ir':6.5,'usd':83.42,'brent':80.92,'gdp':100.0},
    {'name':'future-date','date':pd.to_datetime('2026-06-01'),'wpi':150.0,'ir':6.0,'usd':82.0,'brent':90.0,'gdp':3600.0}
]

for s in scenarios:
    new_row = last_row.copy()
    new_date = s['date'] if s['date'] is not None else (last_date + pd.Timedelta(days=30))
    new_row['Date'] = new_date
    new_row[COLUMN_MAP['brent_crude']] = s['brent']
    new_row[COLUMN_MAP['usd_inr']] = s['usd']
    new_row[COLUMN_MAP['interest_rate']] = s['ir']
    new_row[COLUMN_MAP['gdp_proxy']] = s['gdp']
    new_row[COLUMN_MAP['wpi']] = s['wpi']

    df_pred = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    featured = create_features(df_pred)

    # locate scenario row
    scenario_row = featured.loc[featured['Date'] == new_date]
    if scenario_row.empty:
        scenario_row = featured.iloc[[-1]]

    pred_X = scenario_row.drop(['Date','CPI'], axis=1, errors='ignore')
    if FEATURE_COLUMNS:
        pred_X = pred_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # check for NaNs and extremes
    nan_counts = pred_X.isna().sum().sum()
    max_abs = pred_X.abs().values.max() if pred_X.size>0 else None

    try:
        pred_inflation = pipeline.predict(pred_X)[0]
    except Exception as e:
        pred_inflation = f'ERROR: {e}'

    print('\n--- Scenario:', s['name'], '---')
    print('scenario date:', new_date)
    print('nan count in pred_X:', nan_counts)
    print('max abs in pred_X:', max_abs)
    print('pred_X sample (first 10 cols):')
    print(pred_X.iloc[0,:10].to_dict() if pred_X.shape[1]>0 else {})
    print('pred_inflation (YoY %):', pred_inflation)

    # show any huge pct-change features
    huge = pred_X.columns[(pred_X.abs()>1000).any()].tolist()
    if huge:
        print('Huge features (>1000):', huge)

print('\nDone')
