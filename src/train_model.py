import os
import sys
import platform
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin, clone


def _capture_environment():
    """Snapshot current library versions for cross-machine reproducibility checks."""
    import sklearn
    import xgboost as xgb
    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
    }


def train(df):
    """
    Train models to forecast inflation 1 month ahead.
    """
    print(f"Initial dataset has {len(df)} rows")
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Target creation: We want to predict YoY inflation 1 month from now.
    # First, calculate current YoY inflation for all rows.
    # Since data is strictly monthly, YoY inflation is just a 12-month percentage change.
    df['current_inflation_yoy'] = df['CPI'].pct_change(12) * 100

    # FUTURE TARGET: Shift the current_inflation_yoy backward by 1 month
    # This means for a row today, the target is the inflation 1 month in the future
    forecast_horizon = 1
    df['target_future_inflation'] = df['current_inflation_yoy'].shift(-forecast_horizon)
    
    # Drop rows where target is NaN (from the shift or from pct_change)
    df = df.dropna().reset_index(drop=True)
    
    # Remove the flat synthetic CPI data (anything after May 2026 is synthetic in this dataset)
    df = df[df['Date'] < '2026-06-01'].copy()
    print(f"After handling target and dropping synthetic data, {len(df)} rows remain")

    # Use 2000-2024 H1 for training, keep 2024 H2 - 2026 for holdout
    training_df = df[df['Date'] < '2024-07-01'].copy().reset_index(drop=True)
    holdout_df = df[df['Date'] >= '2024-07-01'].copy().reset_index(drop=True)
    
    print(f"Training on 2000-2024 ({len(training_df)} rows)")
    
    X = training_df.drop(['Date', 'CPI', 'target_future_inflation'], axis=1)
    y = training_df['target_future_inflation']

    tscv = TimeSeriesSplit(n_splits=5)

    # 1. Ridge
    print('\nTuning Ridge...')
    ridge_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', Ridge())
    ])
    ridge_param_grid = {'model__alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge_grid = GridSearchCV(ridge_pipe, ridge_param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    ridge_grid.fit(X, y)
    best_ridge = ridge_grid.best_estimator_
    
    cv_results = cross_validate(
        best_ridge, X, y,
        cv=tscv,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
        return_train_score=False
    )
    ridge_r2 = cv_results['test_r2'].mean()
    ridge_mae = -cv_results['test_neg_mean_absolute_error'].mean()
    ridge_rmse = -cv_results['test_neg_root_mean_squared_error'].mean()
    print(f"Best Ridge R2: {ridge_r2:.4f} (alpha={ridge_grid.best_params_['model__alpha']})")

    # 2. RandomForest
    print('\nTuning RandomForest...')
    rf_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])
    rf_param_grid = {
        'model__n_estimators': [100],
        'model__max_depth': [3, 5, 10],
        'model__min_samples_leaf': [10, 20]
    }
    rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    rf_grid.fit(X, y)
    best_rf = rf_grid.best_estimator_
    
    cv_results = cross_validate(
        best_rf, X, y,
        cv=tscv,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
        return_train_score=False
    )
    rf_r2 = cv_results['test_r2'].mean()
    rf_mae = -cv_results['test_neg_mean_absolute_error'].mean()
    rf_rmse = -cv_results['test_neg_root_mean_squared_error'].mean()
    print(f"Best RF R2: {rf_r2:.4f}")

    # 3. XGBoost
    print('\nTuning XGBoost...')
    xgb_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(random_state=42, verbosity=0))
    ])
    xgb_param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5],
        'model__subsample': [0.8],
        'model__reg_alpha': [0, 0.1]
    }
    xgb_grid = GridSearchCV(xgb_pipe, xgb_param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    xgb_grid.fit(X, y)
    best_xgb = xgb_grid.best_estimator_
    
    cv_results = cross_validate(
        best_xgb, X, y,
        cv=tscv,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
        return_train_score=False
    )
    xgb_r2 = cv_results['test_r2'].mean()
    xgb_mae = -cv_results['test_neg_mean_absolute_error'].mean()
    xgb_rmse = -cv_results['test_neg_root_mean_squared_error'].mean()
    print(f"Best XGB R2: {xgb_r2:.4f}")

    # 4. LightGBM
    print('\nTuning LightGBM...')
    from lightgbm import LGBMRegressor
    lgbm_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', LGBMRegressor(random_state=42, verbosity=-1))
    ])
    lgbm_param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5],
        'model__num_leaves': [15, 31],
        'model__reg_alpha': [0, 0.1]
    }
    lgbm_grid = GridSearchCV(lgbm_pipe, lgbm_param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    lgbm_grid.fit(X, y)
    best_lgbm = lgbm_grid.best_estimator_
    
    cv_results = cross_validate(
        best_lgbm, X, y,
        cv=tscv,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
        return_train_score=False
    )
    lgbm_r2 = cv_results['test_r2'].mean()
    lgbm_mae = -cv_results['test_neg_mean_absolute_error'].mean()
    lgbm_rmse = -cv_results['test_neg_root_mean_squared_error'].mean()
    print(f"Best LightGBM R2: {lgbm_r2:.4f}")

    # Choose best model
    candidates = {
        'Ridge': (best_ridge, ridge_r2),
        'RandomForest': (best_rf, rf_r2),
        'XGBoost': (best_xgb, xgb_r2),
        'LightGBM': (best_lgbm, lgbm_r2)
    }
    chosen_name = max(candidates.keys(), key=lambda k: candidates[k][1])
    best_pipeline, chosen_r2 = candidates[chosen_name]
    
    print(f"\nWINNER: {chosen_name} with cross-validated R2 of {chosen_r2:.4f}")

    import json
    
    # We will use simple metrics for the JSON output based on the best scores
    metrics_output = {
        "best_model": chosen_name,
        "metrics": [
            {
                "name": "Ridge",
                "r2_mean": ridge_r2,
                "mae": ridge_mae,
                "rmse": ridge_rmse
            },
            {
                "name": "Random Forest",
                "r2_mean": rf_r2,
                "mae": rf_mae,
                "rmse": rf_rmse
            },
            {
                "name": "XGBoost",
                "r2_mean": xgb_r2,
                "mae": xgb_mae,
                "rmse": xgb_rmse
            },
            {
                "name": "LightGBM",
                "r2_mean": lgbm_r2,
                "mae": lgbm_mae,
                "rmse": lgbm_rmse
            }
        ]
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=4)

    # Save Models
    os.makedirs('models', exist_ok=True)
    
    def save_artifact(pipe, name, best=False):
        artifact = {
            'pipeline': pipe,
            'feature_columns': list(X.columns),
            'model_name': name,
            'environment': _capture_environment(),
        }
        if best:
            artifact['best_model_choice'] = name
            joblib.dump(artifact, 'models/best_model.pkl')
        else:
            name_file = name.lower().replace(' ', '_')
            joblib.dump(artifact, f'models/{name_file}.pkl')

    save_artifact(best_pipeline, chosen_name, best=True)
    save_artifact(best_ridge, 'Ridge')
    save_artifact(best_rf, 'Random_Forest')
    save_artifact(best_xgb, 'XGBoost')
    save_artifact(best_lgbm, 'LightGBM')

    # Save holdout
    if not holdout_df.empty:
        # Evaluate Best Model on Holdout
        X_test = holdout_df.drop(['Date', 'CPI', 'target_future_inflation'], axis=1)
        y_test = holdout_df['target_future_inflation']
        y_pred = best_pipeline.predict(X_test)
        holdout_r2 = r2_score(y_test, y_pred)
        holdout_mae = mean_absolute_error(y_test, y_pred)
        print(f"\nHoldout (2023-2024) Performance of {chosen_name}:")
        print(f"Holdout R2: {holdout_r2:.4f}")
        print(f"Holdout MAE: {holdout_mae:.4f}")
        
        # Save actual vs predicted for evaluation
        holdout_df['Predicted_Inflation'] = y_pred
        holdout_df.to_csv('models/holdout.csv', index=False)
    
    return best_pipeline
if __name__ == "__main__":
    print("Run via run_train.py")

