import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import create_features

DATA_PATH='data/inflation_dataset.csv'

print('Loading and feature-engineering')
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import RidgeCV

ts = TimeSeriesSplit(n_splits=5)

for i, (train_idx, test_idx) in enumerate(ts.split(X), 1):
    if i==5:
        X_test = X.iloc[test_idx]
        dates = feat.loc[test_idx,'Date']
        print('Fold 5 date range:', dates.min(), dates.max())
        # show columns with abs max > 10
        max_abs = X_test.abs().max()
        big = max_abs[max_abs>20].sort_values(ascending=False)
        print('\nColumns with abs(max)>20 in test set:')
        print(big)
        # show sample rows with biggest values
        row_max = X_test.abs().max(axis=1).nlargest(10)
        print('\nTop 10 rows by max-abs feature value:')
        for idx in row_max.index[:10]:
            print(idx, feat.loc[idx,'Date'], X_test.loc[idx].abs().nlargest(5).to_dict())
        break

print('Done')
