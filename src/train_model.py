import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import json


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
    # **IMPORTANT**: Only train on 2000-2022 data for generalization
    # Features are percentage changes, so they're comparable across any year
    training_df = df[df['Date'] < '2023-01-01'].copy()
    
    if len(training_df) < 100:
        print(f"WARNING: Only {len(training_df)} rows for training. Using all data.")
        training_df = df.copy()
    else:
        print(f"Training on 2000-2022 data ({len(training_df)} rows)")
        print(f"Full dataset has {len(df)} rows")
    
    # Prepare features and target
    X = training_df.drop(['Date', 'CPI'], axis=1)
    y = training_df['CPI']

    # K-fold cross-validation setup
    kfold = KFold(n_splits=5, shuffle=False)
    
    # Models: Ridge with K-fold CV is the priority, Linear Regression for comparison
    models = {
        'Ridge (K-fold CV)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=42))  # Default alpha=1.0, L2 regularization
        ]),
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
    }

    results = []
    best_r2 = -float('inf')
    best_model_name = ""
    best_model = None
    all_fold_predictions = []
    all_fold_actual = []

    print("\nTraining and evaluating models with K-fold Cross-Validation...")
    print("-" * 70)

    for model_name, pipeline in models.items():
        fold_scores = []
        fold_mae = []
        fold_rmse = []
        all_preds = np.array([])
        all_actual = np.array([])
        
        print(f"\n{model_name}:")
        print(f"{'Fold':<6} {'MAE':<12} {'RMSE':<12} {'R2':<12}")
        print("-" * 42)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train pipeline
            pipeline_copy = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0, random_state=42) if 'Ridge' in model_name 
                         else LinearRegression())
            ])
            pipeline_copy.fit(X_train, y_train)

            # Predict
            preds = pipeline_copy.predict(X_test)

            # Evaluate
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)

            fold_scores.append(r2)
            fold_mae.append(mae)
            fold_rmse.append(rmse)
            
            print(f"{fold:<6} {mae:<12.4f} {rmse:<12.4f} {r2:<12.4f}")

            # Collect predictions for overall evaluation
            all_preds = np.append(all_preds, preds)
            all_actual = np.append(all_actual, y_test.values)

        # Average across folds
        avg_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)
        avg_mae = np.mean(fold_mae)
        avg_rmse = np.mean(fold_rmse)

        print("-" * 42)
        print(f"{'Avg:':<6} {avg_mae:<12.4f} {avg_rmse:<12.4f} {avg_r2:<12.4f} (±{std_r2:.4f})\n")

        results.append({
            'name': model_name,
            'mae': round(avg_mae, 4),
            'rmse': round(avg_rmse, 4),
            'r2_mean': round(avg_r2, 4),
            'r2_std': round(std_r2, 4),
            'folds': [round(score, 4) for score in fold_scores]
        })

        # Track best model by average R2 score
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model_name = model_name
            best_model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0, random_state=42) if 'Ridge' in model_name
                         else LinearRegression())
            ])
            # Re-train on full training set for final model
            best_model.fit(X, y)
            all_fold_predictions = all_preds
            all_fold_actual = all_actual

    print("=" * 70)
    print(f"✓ Best Model: {best_model_name} (R2: {best_r2:.4f})")
    print("=" * 70)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save the best model pipeline (includes the scaler)
    joblib.dump(best_model, "models/best_model.pkl")

    # Build prediction chart data from cross-validation results
    chart_sample = min(24, len(all_fold_predictions))
    indices = np.linspace(0, len(all_fold_predictions) - 1, chart_sample, dtype=int)
    
    chart_actual = [round(float(all_fold_actual[i]), 2) for i in indices]
    chart_preds = [round(float(all_fold_predictions[i]), 2) for i in indices]
    chart_dates = [f"CV_{i+1}" for i in range(chart_sample)]

    # Save evaluation metrics + prediction chart data to JSON
    with open("models/metrics.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "method": "K-Fold Cross-Validation",
            "features": "Percentage Changes (% changes instead of absolute values)",
            "training_data": "2000-2022 only for generalization to future years",
            "metrics": results,
            "prediction_chart": {
                "dates": chart_dates,
                "actual": chart_actual,
                "best_model_predictions": chart_preds
            }
        }, f, indent=4)

    return best_model
