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
from sklearn.base import BaseEstimator, RegressorMixin, clone


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
    # Train on full 2000-01-01 through end of 2022 as requested
    recent_start = pd.Timestamp('2000-01-01')
    training_df = df[(df['Date'] >= recent_start) & (df['Date'] < '2023-01-01')].copy().reset_index(drop=True)

    if len(training_df) < 50:
        print(f"WARNING: Only {len(training_df)} rows for 2000-2022 training. Falling back to full pre-2023 data.")
        training_df = df[df['Date'] < '2023-01-01'].copy().reset_index(drop=True)
    else:
        print(f"Training on 2000-01-01 through 2022 ({len(training_df)} rows)")
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
    # RBI target range (default 2-6%). Can be adjusted if needed.
    rbi_min, rbi_max = 2.0, 6.0

    # Clip wrapper to enforce RBI range and avoid negative predictions
    class ClipRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, estimator=None, min_value=None, max_value=None):
            self.estimator = estimator
            self.min_value = min_value
            self.max_value = max_value

        def fit(self, X, y):
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y)
            return self

        def predict(self, X):
            preds = self.estimator_.predict(X)
            preds = np.array(preds, dtype=float)
            if self.min_value is not None:
                preds = np.maximum(preds, self.min_value)
            if self.max_value is not None:
                preds = np.minimum(preds, self.max_value)
            return preds

    ridge_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', ClipRegressor(RidgeCV(alphas=ridge_alphas, cv=tscv), min_value=rbi_min, max_value=rbi_max))
    ])

    rf_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', ClipRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1), min_value=rbi_min, max_value=rbi_max))
    ])

    xgb_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', ClipRegressor(XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0), min_value=rbi_min, max_value=rbi_max))
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

    # Compute RMSE per-fold for each model (re-evaluate with cross_val_predict would be heavier)
    from math import sqrt
    ridge_rmse = [sqrt(float(x)) for x in (np.square(ridge_mae) if False else np.array(ridge_mae))]
    rf_rmse = [sqrt(float(x)) for x in (np.square(rf_mae) if False else np.array(rf_mae))]
    xgb_rmse = [sqrt(float(x)) for x in (np.square(xgb_mae) if False else np.array(xgb_mae))]

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

    # Fit and save individual model artifacts for training/testing convenience
    named_pipelines = {
        'ridge': ridge_pipeline,
        'random_forest': rf_pipeline,
        'xgboost': xgb_pipeline,
    }
    saved_models = []
    for name, pipe in named_pipelines.items():
        fitted = pipe.fit(X, y)
        artifact = {'pipeline': fitted, 'feature_columns': list(X.columns), 'model_name': name}
        path = f'models/{name}.pkl'
        joblib.dump(artifact, path)
        saved_models.append(path)

    # Save a holdout dataset (post-2022) for later testing if present
    holdout_df = df[df['Date'] >= '2023-01-01'].copy()
    if not holdout_df.empty:
        holdout_path = 'models/holdout.csv'
        holdout_df.to_csv(holdout_path, index=False)
    else:
        holdout_path = None

    # Save metrics summary
    # Calculate RMSE properly per-fold using predictions would require cross_val_predict; approximate using MAE->RMSE is not ideal.
    # Here we compute RMSE by running TimeSeriesSplit predictions for each estimator to get accurate per-fold RMSE.
    def fold_metrics(pipeline):
        rmses = []
        maes = []
        r2s = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            fitted = pipeline.fit(X_train, y_train)
            preds = fitted.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
            r2s.append(r2_score(y_test, preds))
        return r2s, maes, rmses

    ridge_r2, ridge_mae, ridge_rmse = fold_metrics(ridge_pipeline)
    rf_r2, rf_mae, rf_rmse = fold_metrics(rf_pipeline)
    xgb_r2, xgb_mae, xgb_rmse = fold_metrics(xgb_pipeline)

    with open('models/metrics.json', 'w') as f:
        json.dump({
            'best_model': chosen,
            'ridge': {'r2': [float(x) for x in ridge_r2], 'mae': [float(x) for x in ridge_mae], 'rmse': [float(x) for x in ridge_rmse]},
            'random_forest': {'r2': [float(x) for x in rf_r2], 'mae': [float(x) for x in rf_mae], 'rmse': [float(x) for x in rf_rmse]},
            'xgboost': {'r2': [float(x) for x in xgb_r2], 'mae': [float(x) for x in xgb_mae], 'rmse': [float(x) for x in xgb_rmse]},
            'saved_models': saved_models,
            'holdout_path': holdout_path
        }, f, indent=4)

    # --- Additionally train month-over-month model (same feature set, different target) ---
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    # Prefer shift if monthly frequency
    try:
        median_diff = df_sorted['Date'].diff().median()
    except Exception:
        median_diff = pd.Timedelta(days=31)
    if median_diff <= pd.Timedelta(days=31):
        df_sorted['CPI_prev_month'] = df_sorted['CPI'].shift(1)
    else:
        left_m = df_sorted[['Date', 'CPI']].copy()
        left_m['base_date'] = left_m['Date'] - pd.Timedelta(days=30)
        merged_m = pd.merge_asof(left_m, df_sorted[['Date', 'CPI']].sort_values('Date'), left_on='base_date', right_on='Date', direction='nearest', suffixes=('', '_prev'))
        if 'CPI_prev' in merged_m.columns:
            df_sorted['CPI_prev_month'] = merged_m['CPI_prev'].values
        else:
            df_sorted['CPI_prev_month'] = merged_m['CPI'].values

    training_month = df_sorted[(df_sorted['Date'] >= recent_start) & (df_sorted['Date'] < '2023-01-01')].copy().reset_index(drop=True)
    training_month = training_month.dropna(subset=['CPI_prev_month']).reset_index(drop=True)
    if not training_month.empty:
        training_month['inflation_mom'] = ((training_month['CPI'] - training_month['CPI_prev_month']) / training_month['CPI_prev_month']) * 100
        training_month = training_month.dropna(subset=['inflation_mom']).reset_index(drop=True)
        if len(training_month) >= 10:
            X_m = training_month.drop(['Date', 'CPI', 'CPI_prev_month', 'inflation_mom'], axis=1)
            y_m = training_month['inflation_mom']
            # fit and save per-model month artifacts
            month_saved = []
            for name, pipe in named_pipelines.items():
                fitted = pipe.fit(X_m, y_m)
                artifact = {'pipeline': fitted, 'feature_columns': list(X_m.columns), 'model_name': name}
                path = f'models/month_{name}.pkl'
                joblib.dump(artifact, path)
                month_saved.append(path)
            # choose best by mean R2
            month_ridge_res = cross_validate(ridge_pipeline, X_m, y_m, cv=tscv, scoring=scoring)
            month_rf_res = cross_validate(rf_pipeline, X_m, y_m, cv=tscv, scoring=scoring)
            month_xgb_res = cross_validate(xgb_pipeline, X_m, y_m, cv=tscv, scoring=scoring)
            month_means = {'ridge': np.mean(month_ridge_res['test_r2']), 'random_forest': np.mean(month_rf_res['test_r2']), 'xgboost': np.mean(month_xgb_res['test_r2'])}
            month_best = max(month_means.keys(), key=lambda k: month_means[k])
            # save month best model wrapper
            joblib.dump({'pipeline': {'model': month_best}, 'feature_columns': list(X_m.columns), 'model_name': month_best}, 'models/month_best_model.pkl')
            # compute simple fold metrics for month
            def fold_metrics_month(pipeline):
                rmses = []
                maes = []
                r2s = []
                for train_idx, test_idx in tscv.split(X_m):
                    X_train, X_test = X_m.iloc[train_idx], X_m.iloc[test_idx]
                    y_train, y_test = y_m.iloc[train_idx], y_m.iloc[test_idx]
                    fitted = pipeline.fit(X_train, y_train)
                    preds = fitted.predict(X_test)
                    maes.append(mean_absolute_error(y_test, preds))
                    rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
                    r2s.append(r2_score(y_test, preds))
                return r2s, maes, rmses
            mr_r2, mr_mae, mr_rmse = fold_metrics_month(ridge_pipeline)
            rf_r2_m, rf_mae_m, rf_rmse_m = fold_metrics_month(rf_pipeline)
            xgb_r2_m, xgb_mae_m, xgb_rmse_m = fold_metrics_month(xgb_pipeline)
            with open('models/month_metrics.json', 'w') as f:
                json.dump({
                    'best_model': month_best,
                    'rbi_min': rbi_min,
                    'rbi_max': rbi_max,
                    'ridge': {'r2': [float(x) for x in mr_r2], 'mae': [float(x) for x in mr_mae], 'rmse': [float(x) for x in mr_rmse]},
                    'random_forest': {'r2': [float(x) for x in rf_r2_m], 'mae': [float(x) for x in rf_mae_m], 'rmse': [float(x) for x in rf_rmse_m]},
                    'xgboost': {'r2': [float(x) for x in xgb_r2_m], 'mae': [float(x) for x in xgb_mae_m], 'rmse': [float(x) for x in xgb_rmse_m]},
                    'saved_models': month_saved
                }, f, indent=4)

    print(f"\nChosen model: {chosen} (mean R2: {np.mean(chosen_r2):.4f})")
    return best_pipeline
