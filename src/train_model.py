import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(df):
    """
    Train a Random Forest model on inflation data.
    
    Args:
        df: DataFrame with features and 'CPI' target column
    
    Returns:
        Trained RandomForestRegressor model
    """
    # Prepare features and target
    X = df.drop(['Date', 'CPI'], axis=1)
    y = df['CPI']

    # 80/20 train-test split
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Generate predictions
    preds = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    # Save model
    joblib.dump(model, "models/random_forest.pkl")

    return model
