import json
import joblib
import numpy as np
import pandas as pd

art = joblib.load('models/best_model.pkl')
print('artifact_keys:', list(art.keys()))
print('model_name:', art.get('model_name'))
print('best_choice:', art.get('best_model_choice'))
feat = art.get('feature_columns') or []
print('feature_count:', len(feat))

df = pd.read_csv('data/inflation_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
if 'INDCPIALLMINMEI' in df.columns and 'CPI' not in df.columns:
    df = df.rename(columns={'INDCPIALLMINMEI': 'CPI'})

missing = [c for c in feat if c not in df.columns]
print('missing_features_count:', len(missing))
print('missing_features_sample:', missing[:10])

if not missing and feat:
    X = df[feat].head(5)
    preds = np.array(art['pipeline'].predict(X), dtype=float)
    print('preds:', preds.tolist())
    print('pred_min:', float(preds.min()))
    print('pred_max:', float(preds.max()))
    print('negative_count:', int((preds < 0).sum()))
    print('out_of_rbi_count:', int(((preds < 2.0) | (preds > 6.0)).sum()))
else:
    print('prediction_test_skipped: feature mismatch')

if __name__ == '__main__':
    pass
