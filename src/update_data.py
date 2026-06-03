import os
import pandas as pd
import joblib

def update_dataset_with_forecasts():
    data_path = 'data/inflation_dataset.csv'
    model_path = 'models/best_model.pkl'
    
    print("Loading dataset and model...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    loaded = joblib.load(model_path)
    pipeline = loaded['pipeline']
    
    # We need features to run the prediction
    import sys
    sys.path.append('src')
    from feature_engineering import create_features
    
    # Create features for the entire dataset
    df_feat = create_features(df.copy())
    
    # Calculate current_inflation_yoy as it's required by the model
    left = df[['Date', 'INDCPIALLMINMEI']].copy()
    left['base_date'] = left['Date'] - pd.Timedelta(days=365)
    merged = pd.merge_asof(left, df[['Date', 'INDCPIALLMINMEI']], left_on='base_date', right_on='Date', direction='nearest', suffixes=('', '_base'))
    df_feat['current_inflation_yoy'] = ((df['INDCPIALLMINMEI'] - merged['INDCPIALLMINMEI_base']) / merged['INDCPIALLMINMEI_base']) * 100
    
    # Make sure we have the exact columns the model expects
    expected_cols = loaded['feature_columns']
    
    # We will predict the inflation for all rows
    # Note: Our model predicts 180 days in the future, but since the independent variables 
    # in 2025-2026 are completely flat, the prediction will just act as a stabilized baseline forecast.
    
    print("Generating ML forecasts for padded 2025-2026 data...")
    updates_made = 0
    for idx, row in df.iterrows():
        if row['Date'] >= pd.to_datetime('2025-01-01'):
            # Find the corresponding feature row
            try:
                feat_row = df_feat[df_feat['Date'] == row['Date']].iloc[0]
            except IndexError:
                continue
                
            X_pred = pd.DataFrame([feat_row])[expected_cols]
            
            # Predict YoY inflation
            predicted_inflation = pipeline.predict(X_pred)[0]
            
            # Find CPI 365 days ago to back-calculate the new synthetic CPI
            y_ago = row['Date'] - pd.Timedelta(days=365)
            closest_idx = (df['Date'] - y_ago).abs().idxmin()
            past_cpi = df.loc[closest_idx, 'INDCPIALLMINMEI']
            
            # Back-calculate CPI: Inflation = (Current - Past) / Past * 100
            # Current = Past * (1 + Inflation / 100)
            new_cpi = past_cpi * (1 + predicted_inflation / 100.0)
            
            # Update the dataframe
            df.loc[idx, 'INDCPIALLMINMEI'] = new_cpi
            updates_made += 1
            
    print(f"Updated {updates_made} rows with smart ML forecasts.")
    df.to_csv(data_path, index=False)
    print("Successfully saved data/inflation_dataset.csv!")

if __name__ == '__main__':
    update_dataset_with_forecasts()
