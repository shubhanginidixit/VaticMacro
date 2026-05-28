import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import create_features

from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer, r2_score, mean_absolute_error

DATA_PATH='data/inflation_dataset.csv'

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
scoring = {'r2': make_scorer(r2_score), 'mae': make_scorer(mean_absolute_error)}

ridge_pipe = Pipeline([('scaler', RobustScaler()), ('model', RidgeCV(alphas=[0.01,0.1,1.0,3.0,10.0], cv=ts))])
rf_pipe = Pipeline([('scaler', RobustScaler()), ('model', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1))])

print('Evaluating RidgeCV...')
ridge_res = cross_validate(ridge_pipe, X, y, cv=ts, scoring=scoring)
print('Ridge R2 per-fold:', ridge_res['test_r2'])
print('Ridge MAE per-fold:', -ridge_res['test_mae'])

print('\nEvaluating RandomForest...')
rf_res = cross_validate(rf_pipe, X, y, cv=ts, scoring=scoring)
print('RF R2 per-fold:', rf_res['test_r2'])
print('RF MAE per-fold:', -rf_res['test_mae'])

print('\nDone')
