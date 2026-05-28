import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from src.preprocessing import load_and_clean_data
from src.feature_engineering import create_features

SHOCK_START = pd.Timestamp('2017-10-06')
SHOCK_END = pd.Timestamp('2022-01-17')
RECENT_START = pd.Timestamp('2010-01-01')

def main():
    print('Loading data')
    df = load_and_clean_data('data/inflation_dataset.csv')
    df = create_features(df)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f'Removing shock window {SHOCK_START.date()} -> {SHOCK_END.date()}')
    df_clean = df[~((df['Date'] >= SHOCK_START) & (df['Date'] <= SHOCK_END))].copy().reset_index(drop=True)

    # Training window (recent)
    training_df = df_clean[(df_clean['Date'] >= RECENT_START) & (df_clean['Date'] < '2023-01-01')].copy().reset_index(drop=True)
    if len(training_df) < 100:
        training_df = df_clean[df_clean['Date'] < '2023-01-01'].copy().reset_index(drop=True)

    # year-ago CPI for target
    left = training_df[['Date', 'CPI']].sort_values('Date').copy()
    left['base_date'] = left['Date'] - pd.Timedelta(days=365)
    right = df_clean[['Date', 'CPI']].sort_values('Date').copy()
    merged = pd.merge_asof(left, right, left_on='base_date', right_on='Date', direction='nearest', suffixes=('', '_base'))
    training_df['CPI_base'] = merged['CPI_base'].values
    training_df['inflation_yoy'] = ((training_df['CPI'] - training_df['CPI_base']) / training_df['CPI_base']) * 100
    training_df = training_df.dropna(subset=['inflation_yoy']).reset_index(drop=True)

    X = training_df.drop(['Date', 'CPI', 'CPI_base', 'inflation_yoy'], axis=1)
    y = training_df['inflation_yoy']

    print('Training rows:', len(X), 'features:', X.shape[1])

    tscv = TimeSeriesSplit(n_splits=5)
    scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error'}

    candidates = {
        'RandomForest': Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=1))
        ]),
        'XGBoost': Pipeline([
            ('scaler', RobustScaler()),
            ('model', XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbosity=0))
        ])
    }

    results = {}
    for name, pipe in candidates.items():
        print(f'Evaluating {name}...')
        res = cross_validate(pipe, X, y, cv=tscv, scoring=scoring)
        r2 = res['test_r2']
        mae = -res['test_mae']
        results[name] = {
            'r2': [float(x) for x in r2],
            'mae': [float(x) for x in mae],
            'r2_mean': float(np.mean(r2)),
            'mae_mean': float(np.mean(mae))
        }
        print(name, 'r2_mean', results[name]['r2_mean'], 'mae_mean', results[name]['mae_mean'])

    # choose best by mean r2
    chosen = max(results.keys(), key=lambda k: results[k]['r2_mean'])
    print('Chosen model after exclusion:', chosen)

    # fit chosen model on full training set
    best_pipeline = candidates[chosen]
    best_pipeline.fit(X, y)

    # save artifact
    os.makedirs('models', exist_ok=True)
    artifact = {
        'pipeline': best_pipeline,
        'feature_columns': list(X.columns),
        'best_model_choice': chosen,
        'model_name': chosen
    }
    joblib.dump(artifact, 'models/best_model.pkl')

    # save metrics.json including results and an entry noting exclusion
    metrics = {'best_model': chosen, 'excluded_window': [str(SHOCK_START.date()), str(SHOCK_END.date())]}
    for k, v in results.items():
        metrics[k.lower()] = {'r2': v['r2'], 'mae': v['mae']}

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print('Saved models/best_model.pkl and models/metrics.json')
    print('Results:', json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
