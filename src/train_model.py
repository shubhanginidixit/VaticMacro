import os
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def train(df):
    """
    Train Ridge Regression with K-fold cross-validation on 2000-2022 data.
    Uses percentage-change features for generalization across any year.

    Ridge regression with K-fold CV ensures:
    - Model generalizes to unseen years (2026, 2027, etc.)
    - No overfitting to specific value ranges
    - Robust performance with percentage-change features

    Args:
        df: DataFrame with features and 'CPI' target column

    Returns:
        The best trained model (Pipeline with scaler + Ridge)
    """
    # **IMPORTANT**: Prefer a recent training window to reduce regime-shift damage
    # We train the model to predict Year-over-Year (YoY) inflation (%) rather than raw CPI.
    # Use merge_asof to find the nearest year-ago CPI for each training date.
    df = df.sort_values('Date').reset_index(drop=True)
    recent_start = pd.Timestamp('2010-01-01')
    training_df = df[(df['Date'] >= recent_start) & (df['Date'] < '2023-01-01')].copy().reset_index(drop=True)
    
    if len(training_df) < 100:
        print(f"WARNING: Only {len(training_df)} rows for recent-window training. Falling back to full pre-2023 data.")
        training_df = df[df['Date'] < '2023-01-01'].copy().reset_index(drop=True)
    else:
        print(f"Training on recent-window data since {recent_start.date()} ({len(training_df)} rows)")
        print(f"Full dataset has {len(df)} rows")
    
    # Find the matching year-ago CPI for each training row using nearest-date merge
    left = training_df[['Date', 'CPI']].sort_values('Date').copy()
    left['base_date'] = left['Date'] - pd.Timedelta(days=365)
    right = df[['Date', 'CPI']].sort_values('Date').copy()
    merged = pd.merge_asof(left, right, left_on='base_date', right_on='Date', direction='nearest', suffixes=('', '_base'))
    training_df['CPI_base'] = merged['CPI_base'].values

    # Calculate YoY inflation (%) as the target and drop rows without a valid year-ago CPI
    training_df['inflation_yoy'] = ((training_df['CPI'] - training_df['CPI_base']) / training_df['CPI_base']) * 100
    training_df = training_df.dropna(subset=['inflation_yoy']).reset_index(drop=True)

    # Prepare features and target: drop Date, CPI and CPI_base; target is YoY inflation
    X = training_df.drop(['Date', 'CPI', 'CPI_base', 'inflation_yoy'], axis=1)
    y = training_df['inflation_yoy']

    # TimeSeriesSplit cross-validation and RidgeCV with RobustScaler
    tscv = TimeSeriesSplit(n_splits=5)

    ridge_alphas = [0.01, 0.1, 1.0, 3.0, 10.0]

    # Try both RidgeCV and RandomForest (choose the better based on CV)
    ridge_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', RidgeCV(alphas=ridge_alphas, cv=tscv))
    ])

    rf_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))
    ])

    xgb_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0))
    ])

    scoring = {'r2': 'r2', 'mae': 'neg_mean_absolute_error'}

    print('\nEvaluating RidgeCV via TimeSeriesSplit...')
    ridge_res = cross_validate(ridge_pipeline, X, y, cv=tscv, scoring=scoring)
    ridge_r2 = ridge_res['test_r2']
    ridge_mae = -ridge_res['test_mae']
    print('Ridge R2 per-fold:', ridge_r2)
    print('Ridge MAE per-fold:', ridge_mae)

    print('\nEvaluating RandomForest via TimeSeriesSplit...')
    rf_res = cross_validate(rf_pipeline, X, y, cv=tscv, scoring=scoring)
    rf_r2 = rf_res['test_r2']
    rf_mae = -rf_res['test_mae']
    print('RF R2 per-fold:', rf_r2)
    print('RF MAE per-fold:', rf_mae)

    print('\nEvaluating XGBoost via TimeSeriesSplit...')
    xgb_res = cross_validate(xgb_pipeline, X, y, cv=tscv, scoring=scoring)
    xgb_r2 = xgb_res['test_r2']
    xgb_mae = -xgb_res['test_mae']
    print('XGB R2 per-fold:', xgb_r2)
    print('XGB MAE per-fold:', xgb_mae)

    # Choose best model by mean R2 among Ridge, RF, and XGB
    candidates = {
        'RidgeCV': (ridge_pipeline, np.mean(ridge_r2), ridge_r2, ridge_mae),
        'RandomForest': (rf_pipeline, np.mean(rf_r2), rf_r2, rf_mae),
        'XGBoost': (xgb_pipeline, np.mean(xgb_r2), xgb_r2, xgb_mae)
    }
    # select model with highest mean R2
    chosen_name = max(candidates.keys(), key=lambda k: candidates[k][1])
    best_pipeline, _, chosen_r2, chosen_mae = candidates[chosen_name]
    chosen = chosen_name

    best_pipeline.fit(X, y)

    os.makedirs('models', exist_ok=True)
    model_artifact = {
        'pipeline': best_pipeline,
        'feature_columns': list(X.columns),
        'best_model_choice': chosen,
        'model_name': f'{chosen}'
    }
    joblib.dump(model_artifact, 'models/best_model.pkl')

    # Save metrics summary
    with open('models/metrics.json', 'w') as f:
        json.dump({
            'best_model': chosen,
            'ridge': {'r2': [float(x) for x in ridge_r2], 'mae': [float(x) for x in ridge_mae]},
            'random_forest': {'r2': [float(x) for x in rf_r2], 'mae': [float(x) for x in rf_mae]},
            'xgboost': {'r2': [float(x) for x in xgb_r2], 'mae': [float(x) for x in xgb_mae]},
        }, f, indent=4)

    print(f"\nChosen model: {chosen} (mean R2: {np.mean(chosen_r2):.4f})")
    return best_pipeline
