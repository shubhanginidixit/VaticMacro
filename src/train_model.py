import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json


def train(df):
    """
    Train multiple models on inflation data, evaluate them,
    and save the best performing model.

    Each model is wrapped in a Pipeline with StandardScaler so that
    feature magnitudes don't bias linear models and the comparison is fair.

    Args:
        df: DataFrame with features and 'CPI' target column

    Returns:
        The best trained model (as a Pipeline)
    """
    # Prepare features and target
    X = df.drop(['Date', 'CPI'], axis=1)
    y = df['CPI']

    # 80/20 train-test chronological split — never shuffle time-series data
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Why these hyperparameters? ---
    # Random Forest: limit max_depth to prevent memorizing training patterns.
    #   min_samples_leaf=10 forces each leaf to represent at least 10 real data
    #   points, reducing overfit noise from tight leaf splits.
    # XGBoost: lower learning_rate + fewer estimators + reg_lambda (L2)
    #   slows down fitting and penalizes large weights, which generalizes better
    #   across different inflation regimes (pre- and post-COVID).
    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                random_state=42
            ))
        ])
    }

    results = []
    best_r2 = -float('inf')
    best_model_name = ""
    best_model = None
    best_preds = None

    print("\nTraining and evaluating models...")
    print("-" * 50)

    for name, pipeline in models.items():
        # Train
        pipeline.fit(X_train, y_train)

        # Predict on test set
        preds = pipeline.predict(X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        print(f"{name}:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}\n")

        results.append({
            'name': name,
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mse': round(mse, 4),
            'r2': round(r2, 4)
        })

        # Track the best model by R2 score
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = pipeline
            best_preds = preds

    print("-" * 50)
    print(f"Best Model: {best_model_name} (R2: {best_r2:.4f})")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save the best model pipeline (includes the scaler)
    joblib.dump(best_model, "models/best_model.pkl")

    # Build prediction chart data for the last 24 test points so the UI
    # can display actual vs predicted curves with real numbers
    test_dates = df.iloc[split:]['Date'].reset_index(drop=True)
    actual_cpi = y_test.reset_index(drop=True)
    chart_sample = min(24, len(test_dates))

    # Sample evenly across the test period for a clean chart
    indices = np.linspace(0, len(test_dates) - 1, chart_sample, dtype=int)
    chart_dates = [test_dates.iloc[i].strftime('%b %Y') for i in indices]
    chart_actual = [round(float(actual_cpi.iloc[i]), 2) for i in indices]
    chart_preds = [round(float(best_preds[i]), 2) for i in indices]

    # Save evaluation metrics + prediction chart data to JSON
    with open("models/metrics.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "metrics": results,
            "prediction_chart": {
                "dates": chart_dates,
                "actual": chart_actual,
                "best_model_predictions": chart_preds
            }
        }, f, indent=4)

    return best_model
