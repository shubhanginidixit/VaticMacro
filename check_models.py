import os, json, joblib
import pandas as pd
import numpy as np

print('Checking models and metrics...')
print('models dir exists:', os.path.exists('models'))
print('files:', os.listdir('models'))

metrics = None
if os.path.exists('models/metrics.json'):
    with open('models/metrics.json') as f:
        metrics = json.load(f)
        print('\nLoaded models/metrics.json')
else:
    print('\nNo models/metrics.json found')

# load all pkl files and run predictions on holdout if available
holdout = None
if os.path.exists('models/holdout.csv'):
    holdout = pd.read_csv('models/holdout.csv')
    if 'Date' in holdout.columns:
        holdout['Date'] = pd.to_datetime(holdout['Date'])

pkl_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
report = {}
for p in pkl_files:
    path = os.path.join('models', p)
    try:
        obj = joblib.load(path)
    except Exception as e:
        report[p] = {'error': str(e)}
        continue
    if isinstance(obj, dict) and 'pipeline' in obj:
        pipe = obj['pipeline']
        cols = obj.get('feature_columns')
        report[p] = {'loaded': True, 'type': str(type(pipe))}
        if holdout is not None and cols is not None and set(cols).issubset(holdout.columns):
            X = holdout[cols].iloc[:50]
            try:
                preds = pipe.predict(X)
                preds = np.array(preds, dtype=float)
                report[p]['pred_min'] = float(np.min(preds))
                report[p]['pred_max'] = float(np.max(preds))
                report[p]['pred_negative_count'] = int(np.sum(preds < 0))
                report[p]['out_of_rbi_range_count'] = int(np.sum((preds < 2.0) | (preds > 6.0)))
            except Exception as e:
                report[p]['predict_error'] = str(e)
        else:
            report[p]['predict_skipped'] = True
    else:
        report[p] = {'loaded': True, 'object_type': str(type(obj))}

print('\nReport:')
print(json.dumps(report, indent=2))

# Quick scan metrics for negatives in MAE/RMSE (shouldn't be negative)
if metrics:
    for k,v in metrics.items():
        if isinstance(v, dict):
            for m in ['mae','rmse']:
                if m in v:
                    arr = v[m]
                    if any([x < 0 for x in arr]):
                        print(f'Warning: negative values in {k}.{m}:', arr)

print('\nDone')
