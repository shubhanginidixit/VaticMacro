from flask import Flask, render_template, request, redirect, url_for
import json
import joblib
import numpy as np
import pandas as pd
import os
import traceback

from src.feature_engineering import create_features
from src.config import COLUMN_MAP, MODEL_PATH, METRICS_PATH, DATA_PATH

app = Flask(__name__, template_folder='app/templates')

# ── Model & Metrics Loading ──────────────────────────────────────────────────

FEATURE_COLUMNS = None
MODEL_R2 = None
MODEL_NAME = "Unknown"
try:
    best_model = joblib.load(MODEL_PATH)
    if isinstance(best_model, dict):
        FEATURE_COLUMNS = best_model.get("feature_columns")
        MODEL_NAME = best_model.get("model_name", MODEL_NAME)
        best_model = best_model.get("pipeline")
except Exception as e:
    print(f"Warning: Could not load model at {MODEL_PATH}. Error: {e}")
    best_model = None

if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
        # Handle both list-format and dict-format metrics files
        best_name = metrics_data.get("best_model", "")
        for key, val in metrics_data.items():
            if not isinstance(val, dict):
                continue
            if key.replace("_", "").lower() == best_name.replace("_", "").replace(" ", "").lower():
                # Prefer pre-computed r2_mean if available, else compute from list
                if "r2_mean" in val:
                    MODEL_R2 = float(val["r2_mean"])
                else:
                    r2_vals = val.get("r2", [])
                    if isinstance(r2_vals, list) and len(r2_vals) > 0:
                        MODEL_R2 = float(sum(r2_vals) / len(r2_vals))
                    elif r2_vals is not None:
                        MODEL_R2 = float(r2_vals)
                break
    except Exception as e:
        print(f"Warning: Could not load metrics at {METRICS_PATH}. Error: {e}")


def _extract_feature_importances():
    """
    Extract real feature importances from the saved model.
    For tree-based models (RF, XGBoost): uses feature_importances_
    For linear models (Ridge): uses absolute coefficient values
    Returns a sorted list of {name, value, percentage} dicts.
    """
    if best_model is None or FEATURE_COLUMNS is None:
        return []

    try:
        model_step = best_model.named_steps.get('model')
        if model_step is None:
            return []

        # Get raw importance values
        if hasattr(model_step, 'feature_importances_'):
            raw_importances = model_step.feature_importances_
        elif hasattr(model_step, 'coef_'):
            raw_importances = np.abs(model_step.coef_)
        else:
            return []

        # Pair with column names and sort by importance (descending)
        paired = list(zip(FEATURE_COLUMNS, raw_importances))
        paired.sort(key=lambda x: x[1], reverse=True)

        # Take top 8 features and convert to percentages
        top_features = paired[:8]
        total = sum(v for _, v in top_features)
        if total == 0:
            return []

        # Map technical column names to human-readable labels
        label_map = {
            'WPIATT01INM661N': 'WPI',
            'DEXINUS': 'USD/INR',
            'INTDSRINM193N': 'Interest Rate',
            'MKTGDPINA646NWDB': 'GDP',
            'INDLORSGPNOSTSAM': 'GDP Proxy',
            'Unemployment Rate Annually': 'Unemployment',
            'WPI_to_CPI': 'WPI/CPI Ratio',
        }

        result = []
        for col_name, importance in top_features:
            # Build a human-readable label from the column name
            readable = col_name
            for key, label in label_map.items():
                if key in col_name:
                    suffix = col_name.replace(key, '').strip('_')
                    readable = f"{label} {suffix}" if suffix else label
                    break
            pct = round((importance / total) * 100)
            result.append({
                "name": readable,
                "value": f"{importance:.3f}",
                "percentage": max(pct, 1)
            })

        return result
    except Exception as e:
        print(f"Warning: Could not extract feature importances: {e}")
        return []


def _compute_holdout_predictions():
    """
    Compute real actual vs predicted values on recent data
    for the models page chart. Uses the last 12 months of
    available data as a holdout window.
    Resamples to monthly and adds autoregressive features
    to match the training pipeline.
    """
    if best_model is None or not os.path.exists(DATA_PATH):
        return {"dates": [], "actual": [], "best_model_predictions": []}

    try:
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        featured_df = create_features(df)

        # Resample to monthly (matching training pipeline)
        monthly = featured_df.set_index('Date').resample('MS').last().reset_index()
        monthly = monthly.dropna().reset_index(drop=True)

        # Compute YoY inflation target
        monthly['CPI_lag12'] = monthly['CPI'].shift(12)
        monthly['inflation_yoy'] = (
            (monthly['CPI'] - monthly['CPI_lag12']) / monthly['CPI_lag12']
        ) * 100

        # Add autoregressive features (same as train_model.py)
        monthly['inflation_lag1'] = monthly['inflation_yoy'].shift(1)
        monthly['inflation_lag3'] = monthly['inflation_yoy'].shift(3)
        monthly['inflation_lag6'] = monthly['inflation_yoy'].shift(6)
        monthly['inflation_rolling_mean_3'] = monthly['inflation_yoy'].rolling(3).mean()
        monthly['inflation_rolling_mean_6'] = monthly['inflation_yoy'].rolling(6).mean()
        monthly = monthly.dropna().reset_index(drop=True)

        # Filter to only months with real CPI data (not forward-filled)
        monthly['cpi_changed'] = monthly['CPI'].diff().abs() > 0.001
        monthly.loc[0, 'cpi_changed'] = True
        real_months = monthly[monthly['cpi_changed']].copy()

        recent = real_months.tail(12).copy()

        if len(recent) < 3:
            return {"dates": [], "actual": [], "best_model_predictions": []}

        X_holdout = recent.drop(
            ['Date', 'CPI', 'CPI_lag12', 'inflation_yoy', 'cpi_changed'],
            axis=1, errors='ignore'
        )
        if FEATURE_COLUMNS:
            X_holdout = X_holdout.reindex(columns=FEATURE_COLUMNS, fill_value=0)

        predictions = best_model.predict(X_holdout)

        dates = [d.strftime('%b %Y') for d in recent['Date']]
        actual = [round(float(v), 2) for v in recent['inflation_yoy']]
        predicted = [round(float(v), 2) for v in predictions]

        return {"dates": dates, "actual": actual, "best_model_predictions": predicted}
    except Exception as e:
        print(f"Warning: Could not compute holdout predictions: {e}")
        traceback.print_exc()
        return {"dates": [], "actual": [], "best_model_predictions": []}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('home.html', active_page='dashboard')

@app.route('/dashboard')
def dashboard():
    # Load real data
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    cpi_col = COLUMN_MAP['cpi']

    # Find the last row where CPI actually changed (not just forward-filled)
    # This gives us the most recent REAL observation date
    cpi_diff = df[cpi_col].diff().abs()
    real_data_mask = cpi_diff > 0.001
    if real_data_mask.any():
        last_real_idx = real_data_mask[real_data_mask].index[-1]
        # Use the last row that shares this CPI value (the real observation)
        latest = df.loc[last_real_idx]
    else:
        latest = df.iloc[-1]
    
    # Resample to monthly for clean comparisons (avoids daily noise)
    df_monthly = df.set_index('Date').resample('MS').last().reset_index()
    
    # Find the latest month with real (non-stale) CPI data
    # Drop months where CPI is identical to the previous month (forward-filled)
    df_monthly['cpi_changed'] = df_monthly[cpi_col].diff().abs() > 0.001
    df_monthly.loc[0, 'cpi_changed'] = True  # First row is always real
    real_months = df_monthly[df_monthly['cpi_changed']].copy()
    
    if len(real_months) >= 2:
        latest_month = real_months.iloc[-1]
        prev_month = real_months.iloc[-2]
    else:
        latest_month = df_monthly.iloc[-1]
        prev_month = df_monthly.iloc[-2] if len(df_monthly) >= 2 else latest_month
    
    # Get row from ~365 days ago for YoY inflation
    past_year_date = latest_month['Date'] - pd.Timedelta(days=365)
    idx_365 = (df_monthly['Date'] - past_year_date).abs().idxmin()
    past_year_row = df_monthly.loc[idx_365]
    
    # Get row from 2 months ago for month-over-month comparison
    two_months_ago_date = prev_month['Date'] - pd.Timedelta(days=365)
    idx_prev_year = (df_monthly['Date'] - two_months_ago_date).abs().idxmin()
    prev_year_row = df_monthly.loc[idx_prev_year]
    
    # Current YoY Inflation
    current_cpi = latest_month[cpi_col]
    past_year_cpi = past_year_row[cpi_col]
    inflation_rate = ((current_cpi - past_year_cpi) / past_year_cpi) * 100 if past_year_cpi else 0
    
    # Previous month's YoY inflation (to compute change)
    prev_cpi = prev_month[cpi_col]
    prev_year_cpi = prev_year_row[cpi_col]
    prev_inflation = ((prev_cpi - prev_year_cpi) / prev_year_cpi) * 100 if prev_year_cpi else 0
    inflation_change = inflation_rate - prev_inflation

    def pct_change(current, past):
        if past == 0:
            return 0
        return ((current - past) / past) * 100

    # Chart data: last 24 months with real CPI data
    chart_months = real_months.tail(24).copy()
    chart_dates = []
    chart_values = []
    for _, row in chart_months.iterrows():
        d = row['Date']
        y_ago = d - pd.Timedelta(days=365)
        closest_y_idx = (df_monthly['Date'] - y_ago).abs().idxmin()
        y_ago_cpi = df_monthly.loc[closest_y_idx, cpi_col]
        val = ((row[cpi_col] - y_ago_cpi) / y_ago_cpi) * 100 if y_ago_cpi else 0
        chart_dates.append(d.strftime('%b %Y'))
        chart_values.append(round(val, 2))

    # Latest indicator values (from the truly latest row for non-CPI indicators)
    latest_raw = df.iloc[-1]
    latest_date = latest_month['Date']

    dashboard_data = {
        'active_page': 'dashboard',
        'inflation_rate': round(inflation_rate, 2),
        'inflation_change': round(inflation_change, 2),
        'cpi_value': round(current_cpi, 2),
        'cpi_change': round(pct_change(current_cpi, prev_cpi), 2),
        'wpi_value': round(latest_month[COLUMN_MAP['wpi']], 2),
        'wpi_change': round(pct_change(latest_month[COLUMN_MAP['wpi']], prev_month[COLUMN_MAP['wpi']]), 2),
        'interest_rate': round(latest_raw[COLUMN_MAP['interest_rate']], 2),
        'usdinr_value': round(latest_raw[COLUMN_MAP['usd_inr']], 2),
        'usdinr_change': round(pct_change(latest_raw[COLUMN_MAP['usd_inr']], prev_month[COLUMN_MAP['usd_inr']]), 2),
        'brent_value': round(latest_raw[COLUMN_MAP['brent_crude']], 2),
        'brent_change': round(pct_change(latest_raw[COLUMN_MAP['brent_crude']], prev_month[COLUMN_MAP['brent_crude']]), 2),
        'gdp_growth': round(pct_change(latest_raw[COLUMN_MAP['gdp_proxy']], past_year_row[COLUMN_MAP['gdp_proxy']]), 2),
        'avg_inflation': round(sum(chart_values)/len(chart_values), 2) if chart_values else 0,
        'peak_inflation': max(chart_values) if chart_values else 0,
        'peak_date': chart_dates[chart_values.index(max(chart_values))] if chart_values else 'N/A',
        'low_inflation': min(chart_values) if chart_values else 0,
        'low_date': chart_dates[chart_values.index(min(chart_values))] if chart_values else 'N/A',
        'trend': 'Rising' if inflation_change > 0 else 'Declining',
        'num_features': 8,
        'num_observations': len(df),
        'date_range': f"{df['Date'].dt.year.min()} - {latest_date.year}",
        'inflation_history': {
            'dates': chart_dates,
            'values': chart_values
        }
    }
    return render_template('dashboard.html', **dashboard_data)

@app.route('/analysis')
def analysis():
    df = pd.read_csv(DATA_PATH)
    
    # Calculate Correlation Matrix
    corr_cols = {
        COLUMN_MAP['cpi']: 'CPI',
        COLUMN_MAP['wpi']: 'WPI',
        COLUMN_MAP['interest_rate']: 'Interest Rate',
        COLUMN_MAP['usd_inr']: 'USD/INR',
        COLUMN_MAP['brent_crude']: 'Brent Crude',
        COLUMN_MAP['gdp_proxy']: 'GDP Proxy'
    }
    corr_df = df[list(corr_cols.keys())].rename(columns=corr_cols).corr().round(2)
    corr_matrix = corr_df.to_dict('index')

    # Generate data-driven key findings from the correlation matrix
    cpi_corr = corr_df['CPI'].drop('CPI').sort_values(ascending=False)
    key_findings = []
    
    # Find the strongest positive correlations with CPI
    top_positive = cpi_corr[cpi_corr > 0.5]
    if len(top_positive) > 0:
        names = ', '.join(top_positive.index[:3])
        key_findings.append(f"CPI is strongly correlated with {names} (r > 0.5).")
    
    # Find negative correlations with CPI
    negatives = cpi_corr[cpi_corr < -0.1]
    if len(negatives) > 0:
        for name, val in negatives.items():
            key_findings.append(f"{name} has an inverse relationship with CPI (r = {val}).")
    
    # Find the strongest non-CPI correlation pair
    corr_upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    corr_upper = corr_upper.drop('CPI', axis=0, errors='ignore').drop('CPI', axis=1, errors='ignore')
    max_pair = corr_upper.stack().abs().idxmax()
    max_val = corr_df.loc[max_pair[0], max_pair[1]]
    key_findings.append(f"Strongest non-CPI pair: {max_pair[0]} and {max_pair[1]} (r = {max_val}).")
    
    if not key_findings:
        key_findings = ["No significant correlations found in the dataset."]

    analysis_context = {
        'active_page': 'analysis',
        'key_findings': key_findings,
        'feature_names': list(corr_cols.values())
    }
    
    analysis_data_dict = {}
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df_monthly = df.set_index('Date').resample('MS').last().reset_index()
    dates_str = df_monthly['Date'].dt.strftime('%b %Y').tolist()
    
    for col_key, col_name in corr_cols.items():
        vals = df_monthly[col_key].dropna().tolist()
        counts, bins = np.histogram(df[col_key].dropna(), bins=20)
        analysis_data_dict[col_name] = {
            'dates': dates_str,
            'values': vals,
            'hist_counts': counts.tolist(),
            'hist_labels': [f"{round(bins[i], 1)}" for i in range(len(bins)-1)]
        }
        
    analysis_context['analysis_data'] = analysis_data_dict
    analysis_context['corr_matrix'] = corr_matrix
    analysis_context['features'] = list(corr_cols.values())

    return render_template('analysis_page.html', **analysis_context)

@app.route('/models')
def models():
    # Load metrics to display dynamically
    metrics_data = {"best_model": "Unknown", "metrics": {}}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            
    best_model_name = metrics_data.get("best_model", "Unknown")

    # Parse metrics from either list-format or dict-format
    models_metrics = []

    if isinstance(metrics_data.get("metrics"), list):
        models_metrics = metrics_data.get("metrics", [])
    else:
        for key, val in metrics_data.items():
            if key in ("best_model", "excluded_window"):
                continue
            if not isinstance(val, dict):
                continue
            name = val.get("name") or key.replace("_", " ").title()

            def _mean_or_value(x):
                try:
                    if isinstance(x, list) and len(x) > 0:
                        return round(float(sum(x) / len(x)), 4)
                    if x is None:
                        return 0.0
                    return round(float(x), 4)
                except Exception:
                    return 0.0

            r2_mean = _mean_or_value(val.get("r2") or val.get("r2_mean"))
            mae_mean = _mean_or_value(val.get("mae"))
            rmse_mean = _mean_or_value(val.get("rmse"))

            models_metrics.append({
                "name": name,
                "r2_mean": r2_mean,
                "mae": mae_mean,
                "rmse": rmse_mean
            })

    best_model_r2 = 0
    best_model_mae = 0
    best_model_rmse = 0

    # Show honest R² values — do NOT hide negative scores
    for m in models_metrics:
        r2_val = m.get("r2_mean", m.get("r2", 0))
        try:
            r2_val = float(r2_val)
        except Exception:
            r2_val = 0.0
        m["r2_mean"] = round(r2_val, 4)

        if m.get("name", "").lower().replace(" ", "") == best_model_name.lower().replace(" ", ""):
            best_model_r2 = m.get("r2_mean", 0)
            best_model_mae = m.get("mae", 0)
            best_model_rmse = m.get("rmse", 0)

    # Extract REAL feature importances from the trained model
    feature_importances = _extract_feature_importances()

    # Compute REAL holdout predictions for the chart
    prediction_chart_data = _compute_holdout_predictions()
            
    return render_template('models_page.html', 
                           active_page='models',
                           best_model_name=best_model_name,
                           models_metrics=models_metrics,
                           best_model_r2=best_model_r2,
                           best_model_mae=best_model_mae,
                           best_model_rmse=best_model_rmse,
                           feature_importances=feature_importances,
                           prediction_chart_data=prediction_chart_data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Default scenario baseline based on latest real data averages roughly
    current_values = {
        'wpi_index': 136.30,
        'interest_rate': 6.5,
        'usd_inr': 83.42,
        'brent_crude': 80.92,
        'gdp_proxy': 3500.0
    }
    
    prediction = None
    display_prediction = None
    interpretation_text = ""
    interpretation_color = "border-l-primary"
    model_used = MODEL_NAME
    
    if request.method == 'POST':
        try:
            raw_date = request.form.get('scenario_date')

            # Extract values from form safely
            raw_wpi = request.form.get('wpi_index')
            val_wpi = float(raw_wpi) if raw_wpi else current_values['wpi_index']
            
            raw_ir = request.form.get('interest_rate')
            val_ir = float(raw_ir) if raw_ir else current_values['interest_rate']
            
            raw_usd = request.form.get('usd_inr')
            val_usd = float(raw_usd) if raw_usd else current_values['usd_inr']
            
            raw_brent = request.form.get('brent_crude')
            val_brent = float(raw_brent) if raw_brent else current_values['brent_crude']
            
            raw_gdp = request.form.get('gdp_proxy')
            val_gdp = float(raw_gdp) if raw_gdp else current_values['gdp_proxy']
            
            # Update current values to preserve state in UI
            current_values = {
                'wpi_index': val_wpi,
                'interest_rate': val_ir,
                'usd_inr': val_usd,
                'brent_crude': val_brent,
                'gdp_proxy': val_gdp
            }

            scenario_date = pd.to_datetime(raw_date) if raw_date else None
            
            if best_model and os.path.exists(DATA_PATH):
                # Load the raw dataset
                df = pd.read_csv(DATA_PATH)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                
                cpi_col = COLUMN_MAP['cpi']
                
                # Create a scenario row with user inputs
                last_date = df['Date'].max()
                new_date = scenario_date if scenario_date is not None else last_date + pd.Timedelta(days=30)
                
                new_row = df.iloc[-1].copy()
                new_row['Date'] = new_date
                new_row[COLUMN_MAP['brent_crude']] = val_brent
                new_row[COLUMN_MAP['usd_inr']] = val_usd
                new_row[COLUMN_MAP['interest_rate']] = val_ir
                new_row[COLUMN_MAP['gdp_proxy']] = val_gdp
                new_row[COLUMN_MAP['wpi']] = val_wpi
                
                # Build full monthly feature table from historical data
                featured_df = create_features(df)
                featured_monthly = featured_df.set_index('Date').resample('MS').last().reset_index()
                featured_monthly = featured_monthly.dropna().reset_index(drop=True)
                
                # Compute YoY inflation for all months (for autoregressive features)
                featured_monthly['CPI_lag12'] = featured_monthly['CPI'].shift(12)
                featured_monthly['inflation_yoy'] = (
                    (featured_monthly['CPI'] - featured_monthly['CPI_lag12']) / featured_monthly['CPI_lag12']
                ) * 100
                
                # Add autoregressive features from real inflation history
                featured_monthly['inflation_lag1'] = featured_monthly['inflation_yoy'].shift(1)
                featured_monthly['inflation_lag3'] = featured_monthly['inflation_yoy'].shift(3)
                featured_monthly['inflation_lag6'] = featured_monthly['inflation_yoy'].shift(6)
                featured_monthly['inflation_rolling_mean_3'] = featured_monthly['inflation_yoy'].rolling(3).mean()
                featured_monthly['inflation_rolling_mean_6'] = featured_monthly['inflation_yoy'].rolling(6).mean()
                featured_monthly = featured_monthly.dropna().reset_index(drop=True)
                
                # Take the last row as our prediction base
                # This has real autoregressive context from inflation history
                scenario_row = featured_monthly.iloc[[-1]].copy()
                
                # Now overlay the user's scenario economic indicators
                # The model uses percentage-change features, so we adjust the raw
                # features and recompute pct changes where possible
                # For simplicity, we modify the pct_change features directly
                # based on user input vs current values
                indicator_map = {
                    'WPIATT01INM661N': val_wpi,
                    'DEXINUS': val_usd,
                    'INTDSRINM193N': val_ir,
                    'MKTGDPINA646NWDB': val_gdp,
                }
                # Get current values from the last row of raw data
                latest_raw = df.iloc[-1]
                for col_name, user_val in indicator_map.items():
                    current_val = latest_raw.get(col_name, user_val)
                    if current_val and current_val != 0:
                        pct_diff = ((user_val - current_val) / current_val) * 100
                    else:
                        pct_diff = 0
                    # Apply the scenario change to pct_change features
                    for suffix in ['_pct_change_30', '_pct_change_90', '_pct_change_180']:
                        feat_name = f"{col_name}{suffix}"
                        if feat_name in scenario_row.columns:
                            scenario_row[feat_name] = scenario_row[feat_name].values[0] + pct_diff

                # Align with the training feature order
                pred_X = scenario_row.drop(
                    ['Date', 'CPI', 'CPI_lag12', 'inflation_yoy'], axis=1, errors='ignore'
                )
                if FEATURE_COLUMNS:
                    pred_X = pred_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                
                # Model predicts Year-over-Year inflation (%) directly
                pred_inflation = best_model.predict(pred_X)[0]

                # Compute a recent historical fallback (median YoY inflation over last 12 months)
                df_dates = df.copy()
                df_dates['Date'] = pd.to_datetime(df_dates['Date'])
                latest_date_hist = df_dates['Date'].max()
                one_year_ago = latest_date_hist - pd.Timedelta(days=365)
                recent = df_dates[df_dates['Date'] >= one_year_ago].copy()
                yoy_vals = []
                for idx, r in recent.iterrows():
                    d = r['Date']
                    y_ago = d - pd.Timedelta(days=365)
                    closest_idx = (df_dates['Date'] - y_ago).abs().idxmin()
                    past_cpi = df_dates.loc[closest_idx, COLUMN_MAP['cpi']]
                    cur_cpi = r[COLUMN_MAP['cpi']]
                    if past_cpi:
                        yoy_vals.append(((cur_cpi - past_cpi) / past_cpi) * 100)
                fallback_median = float(pd.Series(yoy_vals).median()) if len(yoy_vals) > 0 else 0.0

                prediction_raw = float(pred_inflation)
                # If the model predicts an extreme value, fallback to recent median
                if pd.isna(prediction_raw) or prediction_raw < 0 or abs(prediction_raw) > 25:
                    prediction = round(fallback_median, 2)
                    interpretation_text = f"Model produced an extreme prediction ({round(prediction_raw,2)}%). Using recent median fallback {prediction}% to keep outputs stable."
                    interpretation_color = "border-l-[#ffb4ab]"
                else:
                    prediction = round(prediction_raw, 2)

                display_prediction = prediction

                print(f"\n=== PREDICTION ({MODEL_NAME}) ===")
                print(f"Scenario Date: {new_date.date()}")
                print(f"Predicted YoY Inflation (%): {prediction}%")
                
                # Interpret Results
                if prediction < 2:
                    interpretation_color = "border-l-[#4d8eff]"
                    interpretation_text = f"Model predicts inflation at <strong class='text-[#adc6ff]'>{prediction}%</strong>, below RBI's 2-6% target band."
                elif prediction <= 6:
                    interpretation_color = "border-l-[#4edea3]"
                    interpretation_text = f"Model predicts inflation at <strong class='text-[#4edea3]'>{prediction}%</strong>, within RBI's 2-6% target band."
                else:
                    interpretation_color = "border-l-[#ff5449]"
                    interpretation_text = f"Warning: Model predicts inflation at <strong class='text-[#ffb4ab]'>{prediction}%</strong>, exceeding RBI's 6% upper limit."
            else:
                interpretation_text = "Model or dataset missing. Cannot run prediction."

        except Exception as e:
            print("Error in prediction:")
            traceback.print_exc()
            interpretation_text = "An error occurred during calculation."
            
    return render_template('predict_page.html',
                           active_page='predict',
                           current_values=current_values,
                           scenario_date=(request.form.get('scenario_date') if request.method == 'POST' else ""),
                           prediction=prediction,
                           display_prediction=display_prediction,
                           interpretation_text=interpretation_text,
                           interpretation_color=interpretation_color,
                           model_used=model_used,
                           model_r2=MODEL_R2 if MODEL_R2 is not None else 0)

@app.route('/forecast')
def forecast():
    try:
        # Load real data for forecasting
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        cpi_col = COLUMN_MAP['cpi']
        
        # Resample to monthly to get one bar per month (not per day)
        df_monthly = df.set_index('Date').resample('MS').last().reset_index()
        
        # Filter to only months with real CPI changes (not forward-filled copies)
        df_monthly['cpi_changed'] = df_monthly[cpi_col].diff().abs() > 0.001
        df_monthly.loc[0, 'cpi_changed'] = True
        real_months = df_monthly[df_monthly['cpi_changed']].copy()
        
        # Get last 12 months of real data
        recent_12 = real_months.tail(12).copy()
        
        # Calculate YoY inflation for each month
        months = []
        inflation_rates = []
        for _, row in recent_12.iterrows():
            d = row['Date']
            y_ago = d - pd.Timedelta(days=365)
            closest_idx = (df_monthly['Date'] - y_ago).abs().idxmin()
            past_cpi = df_monthly.loc[closest_idx, cpi_col]
            current_cpi_val = row[cpi_col]
            inf_rate = ((current_cpi_val - past_cpi) / past_cpi) * 100 if past_cpi else 0
            months.append(d.strftime('%b %Y'))
            inflation_rates.append(round(inf_rate, 2))
        
        # Current indicators from the last real month
        latest_real = real_months.iloc[-1]
        current_cpi = latest_real[cpi_col]
        current_wpi = latest_real[COLUMN_MAP['wpi']]
        current_rate = df.iloc[-1][COLUMN_MAP['interest_rate']]  # Latest for rates
        
        # Current inflation (last real month vs year-ago)
        latest_date = latest_real['Date']
        past_year_date = latest_date - pd.Timedelta(days=365)
        past_idx = (df_monthly['Date'] - past_year_date).abs().idxmin()
        past_cpi = df_monthly.loc[past_idx, cpi_col]
        current_inflation = ((current_cpi - past_cpi) / past_cpi) * 100 if past_cpi else 0
        
        forecast_data = {
            'active_page': 'forecast',
            'months': months,
            'inflation_rates': inflation_rates,
            'current_cpi': round(current_cpi, 2),
            'current_wpi': round(current_wpi, 2),
            'current_rate': round(current_rate, 2),
            'current_inflation': round(current_inflation, 2),
            'latest_date': latest_date.strftime('%B %d, %Y'),
            'forecast_model': 'Historical Trend Analysis',
            'model_order': 'N/A (no time-series model)',
            'data_points': len(df)
        }
        
        return render_template('forecast_page.html', **forecast_data)
    except Exception as e:
        print(f"Error in forecast: {e}")
        traceback.print_exc()
        return f"<h1 style='color:red; font-family:sans-serif;'>Error: {str(e)}</h1>"

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')


if __name__ == '__main__':
    print("Starting VaticMacro Flask Server...")
    app.run(debug=True, host='127.0.0.1', port=5000)
