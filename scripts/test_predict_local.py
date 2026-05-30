import joblib
import pandas as pd
from pathlib import Path
from src.feature_engineering import create_features

MODEL='models/best_model.pkl'
DATA='data/inflation_dataset.csv'

print('MODEL exists', Path(MODEL).exists())
print('DATA exists', Path(DATA).exists())

m = joblib.load(MODEL)
if isinstance(m, dict):
    feat_cols = m.get('feature_columns')
    pipe = m.get('pipeline')
else:
    feat_cols = None
    pipe = m
print('pipeline type', type(pipe))

# load data
df = pd.read_csv(DATA)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
# construct scenario row same as app
last = df.iloc[-1].copy()
last['Date'] = last['Date'] + pd.Timedelta(days=30)
df_pred = pd.concat([df, pd.DataFrame([last])], ignore_index=True)
featured = create_features(df_pred)
scenario = featured.loc[featured['Date'] == last['Date']].tail(1)
if scenario.empty:
    scenario = featured.iloc[[-1]]
pred_X = scenario.drop(['Date', 'CPI'], axis=1, errors='ignore')
if feat_cols:
    pred_X = pred_X.reindex(columns=feat_cols, fill_value=0)
print('pred_X cols', pred_X.columns[:10])
print('pred_X values', pred_X.iloc[0].to_dict())
preds = pipe.predict(pred_X)
print('prediction', preds[0])
