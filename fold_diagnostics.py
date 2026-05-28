import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import create_features

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH='data/inflation_dataset.csv'

print('Loading data')
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
feat = create_features(df)

left = feat[['Date','CPI']].sort_values('Date').copy()
left['base_date'] = left['Date'] - pd.Timedelta(days=365)
right = feat[['Date','CPI']].sort_values('Date').copy()
merged = pd.merge_asof(left, right, left_on='base_date', right_on='Date', direction='nearest', suffixes=('','_base'))
feat = feat.loc[merged.index].copy()
feat['CPI_base'] = merged['CPI_base'].values
feat['inflation_yoy'] = ((feat['CPI'] - feat['CPI_base']) / feat['CPI_base']) * 100
feat = feat.dropna(subset=['inflation_yoy']).reset_index(drop=True)

X = feat.drop(['Date','CPI','CPI_base','inflation_yoy'], axis=1)
y = feat['inflation_yoy']

ts = TimeSeriesSplit(n_splits=5)

print('\nPer-fold diagnostics:')
for i, (train_idx, test_idx) in enumerate(ts.split(X), 1):
    test_dates = feat.loc[test_idx, 'Date']
    y_test = y.iloc[test_idx]
    print(f"\nFold {i}: test rows {test_idx[0]}-{test_idx[-1]} | dates {test_dates.min().date()} - {test_dates.max().date()}")
    print('y_test describe:', y_test.describe().to_dict())

# Fit the final pipeline and compute predictions per fold to find worst residuals
pipeline = Pipeline([('scaler', RobustScaler()), ('model', RidgeCV(alphas=[0.01,0.1,1.0,3.0,10.0], cv=ts))])

for i, (train_idx, test_idx) in enumerate(ts.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"Fold {i} performance: MAE={mae:.4f}, R2={r2:.4f}")

print('\nDone')
