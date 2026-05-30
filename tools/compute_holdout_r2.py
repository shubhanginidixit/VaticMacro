"""
Compute R^2 for saved models against models/holdout.csv and print results.
"""
import json, os
from joblib import load
import pandas as pd
from sklearn.metrics import r2_score
import sys

root = os.path.dirname(os.path.dirname(__file__)) if __file__.endswith('tools\\compute_holdout_r2.py') else os.getcwd()
# ensure repo root is on path so package `src` can be imported by unpickling
sys.path.insert(0, root)
metrics_path = os.path.join(root, 'models', 'metrics.json')
if not os.path.exists(metrics_path):
    print('metrics.json not found at', metrics_path)
    raise SystemExit(1)
with open(metrics_path) as f:
    metrics = json.load(f)
models_list = metrics.get('saved_models', [])
holdout = os.path.join(root, 'models', 'holdout.csv')
if not os.path.exists(holdout):
    print('holdout.csv not found at', holdout)
    raise SystemExit(1)
df = pd.read_csv(holdout)
# pick a sensible target
if 'CPI' in df.columns:
    TARGET = 'CPI'
elif 'INDCPIALLMINMEI' in df.columns:
    TARGET = 'INDCPIALLMINMEI'
else:
    TARGET = df.columns[-1]
print('Using TARGET=', TARGET)
X = df.drop(columns=[TARGET, 'Date'], errors='ignore')
y = df[TARGET]

for mpath in models_list:
    full = os.path.join(root, mpath)
    if not os.path.exists(full):
        print('missing model', full)
        continue
    try:
        obj = load(full)
        pipeline = obj.get('pipeline', obj) if isinstance(obj, dict) else obj
        yhat = pipeline.predict(X)
        r2 = r2_score(y, yhat)
        print(mpath, 'r2 =', r2)
    except Exception as e:
        print('error testing', mpath, e)

# helper: compute YoY target series (percent change) aligned to model prediction rows
def compute_yoy_series(df, value_col='CPI'):
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2 = df2.sort_values('Date').reset_index(drop=True)
    y_true = []
    rows = []
    for idx, r in df2.iterrows():
        d = r['Date']
        y_ago = d - pd.Timedelta(days=365)
        # find closest prior
        diffs = (df2['Date'] - y_ago).abs()
        if diffs.empty:
            continue
        closest_idx = diffs.idxmin()
        past_val = df2.loc[closest_idx, value_col]
        if past_val:
            yoy = ((r[value_col] - past_val) / past_val) * 100
            y_true.append(yoy)
            rows.append(idx)
    return pd.Series(y_true), rows

# Now compute YoY-aligned R2 for each model (if possible)
print('\nComputing YoY-aligned R2 (model predictions should be YoY % if trained accordingly)')
_yoy_series, _rows = compute_yoy_series(df)
if len(_rows) == 0:
    print('No YoY-aligned rows found in holdout')
else:
    X_yoy = X.iloc[_rows]
    y_yoy = _yoy_series
    for mpath in models_list:
        full = os.path.join(root, mpath)
        if not os.path.exists(full):
            continue
        try:
            obj = load(full)
            pipeline = obj.get('pipeline', obj) if isinstance(obj, dict) else obj
            yhat = pipeline.predict(X_yoy)
            r2 = r2_score(y_yoy, yhat)
            print(mpath, 'r2_yoy =', r2)
        except Exception as e:
            print('error testing (yoy)', mpath, e)
# also try best_model.pkl
best = os.path.join(root, 'models', 'best_model.pkl')
if os.path.exists(best):
    try:
        obj = load(best)
        pipeline = obj.get('pipeline', obj) if isinstance(obj, dict) else obj
        yhat = pipeline.predict(X)
        print('best_model.pkl r2 =', r2_score(y, yhat))
    except Exception as e:
        print('error testing best_model.pkl', e)

print('Done')
