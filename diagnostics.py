import pandas as pd
import numpy as np
import os
import sys
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import create_features

DATA_PATH='data/inflation_dataset.csv'

print('Loading data')
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# create features
feat = create_features(df)
print('Features shape:', feat.shape)

# compute YoY inflation target
left = feat[['Date','CPI']].sort_values('Date').copy()
left['base_date'] = left['Date'] - pd.Timedelta(days=365)
right = feat[['Date','CPI']].sort_values('Date').copy()
merged = pd.merge_asof(left, right, left_on='base_date', right_on='Date', direction='nearest', suffixes=('','_base'))
feat = feat.loc[merged.index].copy()
feat['CPI_base'] = merged['CPI_base'].values
feat['inflation_yoy'] = ((feat['CPI'] - feat['CPI_base']) / feat['CPI_base']) * 100
feat = feat.dropna(subset=['inflation_yoy'])

print('\nTarget distribution (YoY inflation %):')
print(feat['inflation_yoy'].describe())

X = feat.drop(['Date','CPI','CPI_base','inflation_yoy'], axis=1)
y = feat['inflation_yoy']

# zero variance
zero_var = X.columns[X.std()==0].tolist()
print('\nZero-variance columns:', zero_var)

# huge magnitude features
max_abs = X.abs().max()
huge = max_abs[max_abs>100].sort_values(ascending=False)
print('\nFeatures with abs(max) > 100:')
print(huge.head(20))

# quick CV check using KFold and TimeSeriesSplit
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error

scoring = {'r2': make_scorer(r2_score), 'mae': make_scorer(mean_absolute_error)}

pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=3.0))])

print('\nRunning KFold CV (5)')
kf = KFold(n_splits=5, shuffle=False)
res_kf = cross_validate(pipe, X, y, cv=kf, scoring=scoring, return_train_score=False)
print('KF R2 per-fold:', res_kf['test_r2'])
print('KF mean R2:', res_kf['test_r2'].mean())

print('\nRunning TimeSeriesSplit CV (5)')
ts = TimeSeriesSplit(n_splits=5)
res_ts = cross_validate(pipe, X, y, cv=ts, scoring=scoring, return_train_score=False)
print('TS R2 per-fold:', res_ts['test_r2'])
print('TS mean R2:', res_ts['test_r2'].mean())

print('\nDone')
