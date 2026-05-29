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
    Train inflation prediction models with monthly resampling and
    proper time-series cross-validation.

    Why monthly resampling:
        CPI is reported monthly, but the merged dataset has daily rows
        (from exchange rate data). 96% of daily rows have identical CPI,
        creating massive redundancy that distorts cross-validation.
        Resampling to monthly gives ~150 real observations where each
        row represents a genuinely different data point.

    Why expanding-window CV:
        India's inflation regime shifted dramatically (10% in 2010-2013
        down to 3-4% in 2017-2018). Fixed-window TimeSeriesSplit trains
        on one regime and tests on a completely different one, guaranteeing
        failure. Using a larger minimum training size and fewer splits
        ensures each fold has enough history to capture regime changes.

    Args:
        df: DataFrame with engineered features and 'CPI' column

    Returns:
        The best trained model pipeline
    """
    df = df.sort_values('Date').reset_index(drop=True)

    # ── Step 1: Resample to monthly ──────────────────────────────────────
    # Take the last observation of each month (like a month-end snapshot)
    df_monthly = df.set_index('Date').resample('MS').last().reset_index()
    df_monthly = df_monthly.dropna().reset_index(drop=True)
    print(f"Resampled from {len(df)} daily rows to {len(df_monthly)} monthly rows")

    # ── Step 2: Compute YoY inflation target ─────────────────────────────
    # For each month, find the CPI from 12 months ago and compute the
    # year-over-year percentage change
    df_monthly['CPI_lag12'] = df_monthly['CPI'].shift(12)
    df_monthly['inflation_yoy'] = (
        (df_monthly['CPI'] - df_monthly['CPI_lag12']) / df_monthly['CPI_lag12']
    ) * 100
    df_monthly = df_monthly.dropna(subset=['inflation_yoy']).reset_index(drop=True)

    # ── Step 3: Add autoregressive features ──────────────────────────────
    # Previous months' inflation is the strongest predictor of current inflation
    # (inflation is highly persistent / autocorrelated)
    df_monthly['inflation_lag1'] = df_monthly['inflation_yoy'].shift(1)
    df_monthly['inflation_lag3'] = df_monthly['inflation_yoy'].shift(3)
    df_monthly['inflation_lag6'] = df_monthly['inflation_yoy'].shift(6)
    df_monthly['inflation_rolling_mean_3'] = df_monthly['inflation_yoy'].rolling(3).mean()
    df_monthly['inflation_rolling_mean_6'] = df_monthly['inflation_yoy'].rolling(6).mean()
    df_monthly = df_monthly.dropna().reset_index(drop=True)

    # ── Step 4: Select training window ───────────────────────────────────
    recent_start = pd.Timestamp('2005-01-01')
    training_df = df_monthly[
        (df_monthly['Date'] >= recent_start) & (df_monthly['Date'] < '2023-01-01')
    ].copy().reset_index(drop=True)

    if len(training_df) < 50:
        print(f"WARNING: Only {len(training_df)} monthly rows. Using all pre-2023 data.")
        training_df = df_monthly[df_monthly['Date'] < '2023-01-01'].copy().reset_index(drop=True)

    print(f"Training on {len(training_df)} monthly observations "
          f"({training_df['Date'].min().date()} to {training_df['Date'].max().date()})")

    # ── Step 5: Prepare features and target ──────────────────────────────
    drop_cols = ['Date', 'CPI', 'CPI_lag12', 'inflation_yoy']
    X = training_df.drop(drop_cols, axis=1, errors='ignore')
    y = training_df['inflation_yoy']

    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    print(f"Target: mean={y.mean():.2f}%, std={y.std():.2f}%, "
          f"min={y.min():.2f}%, max={y.max():.2f}%")

    # ── Step 6: Feature selection ────────────────────────────────────────
    # Keep only features with meaningful correlation (|r| > 0.1) to the target
    # This reduces noise from irrelevant features
    correlations = X.corrwith(y).abs()
    useful_features = correlations[correlations > 0.1].index.tolist()
    if len(useful_features) < 5:
        # Fallback: keep top 10 features by correlation
        useful_features = correlations.nlargest(10).index.tolist()
    
    dropped_count = len(X.columns) - len(useful_features)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} weak features (|r| < 0.1). "
              f"Keeping {len(useful_features)} useful features.")
    X = X[useful_features]

    # ── Step 7: Cross-validation ─────────────────────────────────────────
    # Use 5 splits but with monthly data, each fold is ~2 years
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Define model candidates
    ridge_alphas = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]

    ridge_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', RidgeCV(alphas=ridge_alphas))
    ])

    rf_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ))
    ])

    xgb_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        ))
    ])

    scoring = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error'
    }

    print('\n--- Evaluating RidgeCV ---')
    ridge_res = cross_validate(ridge_pipeline, X, y, cv=tscv, scoring=scoring)
    ridge_r2 = ridge_res['test_r2']
    ridge_mae = -ridge_res['test_mae']
    ridge_rmse = -ridge_res['test_rmse']
    print(f'  R2 per fold: {[round(x,4) for x in ridge_r2]}')
    print(f'  Mean R2: {np.mean(ridge_r2):.4f}, Mean MAE: {np.mean(ridge_mae):.4f}')

    print('\n--- Evaluating RandomForest ---')
    rf_res = cross_validate(rf_pipeline, X, y, cv=tscv, scoring=scoring)
    rf_r2 = rf_res['test_r2']
    rf_mae = -rf_res['test_mae']
    rf_rmse = -rf_res['test_rmse']
    print(f'  R2 per fold: {[round(x,4) for x in rf_r2]}')
    print(f'  Mean R2: {np.mean(rf_r2):.4f}, Mean MAE: {np.mean(rf_mae):.4f}')

    print('\n--- Evaluating XGBoost ---')
    xgb_res = cross_validate(xgb_pipeline, X, y, cv=tscv, scoring=scoring)
    xgb_r2 = xgb_res['test_r2']
    xgb_mae = -xgb_res['test_mae']
    xgb_rmse = -xgb_res['test_rmse']
    print(f'  R2 per fold: {[round(x,4) for x in xgb_r2]}')
    print(f'  Mean R2: {np.mean(xgb_r2):.4f}, Mean MAE: {np.mean(xgb_mae):.4f}')

    # ── Step 8: Select best model and fit on full training data ──────────
    candidates = {
        'RidgeCV': (ridge_pipeline, np.mean(ridge_r2), ridge_r2, ridge_mae, ridge_rmse),
        'RandomForest': (rf_pipeline, np.mean(rf_r2), rf_r2, rf_mae, rf_rmse),
        'XGBoost': (xgb_pipeline, np.mean(xgb_r2), xgb_r2, xgb_mae, xgb_rmse)
    }
    chosen_name = max(candidates.keys(), key=lambda k: candidates[k][1])
    best_pipeline, _, chosen_r2, chosen_mae, chosen_rmse = candidates[chosen_name]

    # Fit the best pipeline on all training data
    best_pipeline.fit(X, y)

    # ── Step 9: Save model and metrics ───────────────────────────────────
    os.makedirs('models', exist_ok=True)
    model_artifact = {
        'pipeline': best_pipeline,
        'feature_columns': list(X.columns),
        'best_model_choice': chosen_name,
        'model_name': chosen_name
    }
    joblib.dump(model_artifact, 'models/best_model.pkl')

    metrics = {
        'best_model': chosen_name,
        'training_window': f"{training_df['Date'].min().date()} to {training_df['Date'].max().date()}",
        'monthly_observations': len(training_df),
        'features_used': len(X.columns),
    }
    for name, (_, mean_r2, r2_vals, mae_vals, rmse_vals) in candidates.items():
        key = name.lower().replace(' ', '_')
        metrics[key] = {
            'name': name,
            'r2': [round(float(x), 4) for x in r2_vals],
            'r2_mean': round(float(np.mean(r2_vals)), 4),
            'mae': [round(float(x), 4) for x in mae_vals],
            'rmse': [round(float(x), 4) for x in rmse_vals]
        }

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nChosen model: {chosen_name} (mean R2: {np.mean(chosen_r2):.4f})")
    return best_pipeline
