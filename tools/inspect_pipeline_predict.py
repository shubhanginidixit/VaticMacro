import joblib
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
import sys
# ensure project root is importable so `import src` works during unpickle
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    # Prefer package import (project root on sys.path)
    from src.feature_engineering import create_features
except Exception:
    # Fallback to direct import if src is on sys.path
    from feature_engineering import create_features

MODEL_PATH = Path('models') / 'best_model.pkl'
DATA_PATH = Path('data') / 'inflation_dataset.csv'

print('Loading model:', MODEL_PATH)
loaded = joblib.load(MODEL_PATH)
if isinstance(loaded, dict):
    pipeline = loaded.get('pipeline')
    FEATURE_COLUMNS = loaded.get('feature_columns')
    MODEL_NAME = loaded.get('model_name')
else:
    pipeline = loaded
    FEATURE_COLUMNS = None
    MODEL_NAME = getattr(pipeline, '__class__', str(type(pipeline)))

print('Pipeline type:', type(pipeline))
print('Feature columns count:', 0 if not FEATURE_COLUMNS else len(FEATURE_COLUMNS))
print('Feature columns sample:', (FEATURE_COLUMNS[:10] if FEATURE_COLUMNS else None))

print('Loading data...')
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

for d in ['2012-01-01','2017-01-01','2022-01-01']:
    print('\n--- Scenario date:', d)
    try:
        last = df.iloc[-1].copy()
        new_row = last.copy()
        new_row['Date'] = pd.to_datetime(d)
        # set some example indicator overrides if present
        mapping = {
            'Average of DCOILBRENTEU': 120.0,
            'DEXINUS': 90.0,
            'INTDSRINM193N': 7.5,
            'WPIATT01INM661N': 150.0,
            'MKTGDPINA646NWDB': last.get('MKTGDPINA646NWDB', np.nan)
        }
        for k,v in mapping.items():
            if k in new_row.index:
                new_row[k] = v
        df_pred = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        featured = create_features(df_pred)
        scenario = featured.loc[featured['Date'] == pd.to_datetime(d)].tail(1)
        if scenario.empty:
            scenario = featured.iloc[[-1]]
        pred_X = scenario.drop(['Date', 'CPI'], axis=1, errors='ignore')
        print('Engineered feature columns (sample 20):', list(pred_X.columns)[:20])
        print('Engineered sample values (first 20):')
        print(pred_X.iloc[0].to_dict())
        if FEATURE_COLUMNS:
            pred_X = pred_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
            print('Reindexed to FEATURE_COLUMNS, sample keys:', list(pred_X.columns)[:20])
            print('Reindexed sample values (first 20):')
            print(pred_X.iloc[0].to_dict())
        # predict
        preds = pipeline.predict(pred_X)
        print('Pipeline prediction:', preds[0])
        # inspect final estimator if possible
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(pipeline, Pipeline):
                final = pipeline.named_steps.get('model') or pipeline.steps[-1][1]
                est = getattr(final, 'estimator_', final)
                print('Final estimator type:', type(est))
                if hasattr(est, 'intercept_'):
                    print('Estimator intercept:', float(getattr(est, 'intercept_')))
        except Exception:
            pass
    except Exception:
        print('Error for date', d)
        traceback.print_exc()

print('\nDone.')
