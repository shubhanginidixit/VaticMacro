import os
import sys
import platform
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression
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
    # current_inflation_yoy is already provided by feature_engineering.py

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
    
    X = training_df.drop(['Date', 'CPI', 'target_future_inflation'], axis=1, errors='ignore')
    y = training_df['target_future_inflation']

    tscv = TimeSeriesSplit(n_splits=5)

    # 1. Ridge
    print('\nTuning Ridge...')
    ridge_pipe = Pipeline([
        ('variance', VarianceThreshold(threshold=0.01)),
        ('select', SelectKBest(f_regression)),
        ('scaler', RobustScaler()),
        ('model', Ridge())
    ])
    ridge_param_grid = {
        'model__alpha': [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0],
        'select__k': [5, 8, 10, 12, 15, 18, 20, 25, 'all']
    }

    # 1b. Ridge with mutual_info scoring
    ridge_mi_pipe = Pipeline([
        ('variance', VarianceThreshold(threshold=0.01)),
        ('select', SelectKBest(mutual_info_regression)),
        ('scaler', RobustScaler()),
        ('model', Ridge())
    ])
    ridge_mi_param_grid = {
        'model__alpha': [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0],
        'select__k': [10, 15, 20, 25, 'all']
    }
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

    # 1b. Ridge with mutual_info scoring
    ridge_mi_grid = GridSearchCV(ridge_mi_pipe, ridge_mi_param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    ridge_mi_grid.fit(X, y)
    best_ridge_mi = ridge_mi_grid.best_estimator_
    
    cv_results_mi = cross_validate(
        best_ridge_mi, X, y,
        cv=tscv,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
        return_train_score=False
    )
    ridge_mi_r2 = cv_results_mi['test_r2'].mean()
    ridge_mi_mae = -cv_results_mi['test_neg_mean_absolute_error'].mean()
    ridge_mi_rmse = -cv_results_mi['test_neg_root_mean_squared_error'].mean()
    print(f"Best Ridge(MI) R2: {ridge_mi_r2:.4f} (alpha={ridge_mi_grid.best_params_['model__alpha']}, k={ridge_mi_grid.best_params_['select__k']})")

    # 1c. ElasticNet
    en_pipe = Pipeline([
        ('variance', VarianceThreshold(threshold=0.01)),
        ('select', SelectKBest(f_regression)),
        ('scaler', RobustScaler()),
        ('model', ElasticNet(max_iter=10000, random_state=42))
    ])
    en_param_grid = {
        'model__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'select__k': [10, 15, 20, 'all']
    }
    en_grid = GridSearchCV(en_pipe, en_param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    en_grid.fit(X, y)
    best_en = en_grid.best_estimator_
    
    cv_results_en = cross_validate(
        best_en, X, y,
        cv=tscv,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
        return_train_score=False
    )
    en_r2 = cv_results_en['test_r2'].mean()
    en_mae = -cv_results_en['test_neg_mean_absolute_error'].mean()
    en_rmse = -cv_results_en['test_neg_root_mean_squared_error'].mean()
    print(f"Best ElasticNet R2: {en_r2:.4f} (alpha={en_grid.best_params_['model__alpha']}, l1_ratio={en_grid.best_params_['model__l1_ratio']})")

    # 2. RandomForest
    print('\nTuning RandomForest...')
    rf_pipe = Pipeline([
        ('variance', VarianceThreshold(threshold=0.01)),
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
        ('variance', VarianceThreshold(threshold=0.01)),
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(random_state=42))
    ])
    xgb_param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5, 7]
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
    try:
        from lightgbm import LGBMRegressor
        lgbm_pipe = Pipeline([
            ('variance', VarianceThreshold(threshold=0.01)),
            ('scaler', RobustScaler()),
            ('model', LGBMRegressor(random_state=42, verbose=-1))
        ])
        lgbm_param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__num_leaves': [15, 31],
            'model__max_depth': [-1, 5, 10]
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
    except ImportError:
        print("LightGBM not installed, skipping.")
        best_lgbm = best_rf
        lgbm_r2 = -99.0
        lgbm_mae = 99.0
        lgbm_rmse = 99.0

    # Choose best model
    candidates = {
        'Ridge': (best_ridge, ridge_r2),
        'Ridge(MI)': (best_ridge_mi, ridge_mi_r2),
        'ElasticNet': (best_en, en_r2),
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
                "name": "Ridge(MI)",
                "r2_mean": ridge_mi_r2,
                "mae": ridge_mi_mae,
                "rmse": ridge_mi_rmse
            },
            {
                "name": "ElasticNet",
                "r2_mean": en_r2,
                "mae": en_mae,
                "rmse": en_rmse
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
    save_artifact(best_ridge_mi, 'Ridge_MI')
    save_artifact(best_en, 'ElasticNet')
    save_artifact(best_rf, 'Random_Forest')
    save_artifact(best_xgb, 'XGBoost')
    save_artifact(best_lgbm, 'LightGBM')

    # Save holdout
    if not holdout_df.empty:
        # Evaluate Best Model on Holdout
        X_test = holdout_df.drop(['Date', 'CPI', 'target_future_inflation'], axis=1, errors='ignore')
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

