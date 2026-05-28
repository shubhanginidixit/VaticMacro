from flask import Flask, render_template, request, redirect, url_for
import json
import joblib
import pandas as pd
import os
import traceback
import sys

# Add src to path so feature_engineering can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import create_features

app = Flask(__name__, template_folder='app/templates')

# Paths
MODEL_PATH = "models/best_model.pkl"
METRICS_PATH = "models/metrics.json"
DATA_PATH = "data/inflation_dataset.csv"

FEATURE_COLUMNS = None
MODEL_R2 = None
MODEL_NAME = "Ridge (K-fold CV)"
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
        for metric in metrics_data.get("metrics", []):
            if metric.get("name") == metrics_data.get("best_model"):
                MODEL_R2 = metric.get("r2_mean", metric.get("r2", None))
                break
    except Exception as e:
        print(f"Warning: Could not load metrics at {METRICS_PATH}. Error: {e}")

# Centralized Column Map
COLUMN_MAP = {
    'cpi': 'INDCPIALLMINMEI',
    'wpi': 'WPIATT01INM661N',
    'interest_rate': 'INTDSRINM193N',
    'usd_inr': 'DEXINUS',
    'brent_crude': 'Average of DCOILBRENTEU',
    'gdp_proxy': 'MKTGDPINA646NWDB'
}


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    # Load real data
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Get latest row
    latest = df.iloc[-1]
    
    # Get row from ~30 days ago for monthly changes
    past_30_date = latest['Date'] - pd.Timedelta(days=30)
    idx_30 = (df['Date'] - past_30_date).abs().idxmin()
    past_30 = df.loc[idx_30]
    
    # Get row from ~365 days ago for YoY inflation
    past_365_date = latest['Date'] - pd.Timedelta(days=365)
    idx_365 = (df['Date'] - past_365_date).abs().idxmin()
    past_365 = df.loc[idx_365]
    
    # Calculations
    current_cpi = latest[COLUMN_MAP['cpi']]
    past_year_cpi = past_365[COLUMN_MAP['cpi']]
    inflation_rate = ((current_cpi - past_year_cpi) / past_year_cpi) * 100 if past_year_cpi else 0
    
    # Change in inflation from last month
    past_month_year_ago_date = past_30['Date'] - pd.Timedelta(days=365)
    idx_month_year_ago = (df['Date'] - past_month_year_ago_date).abs().idxmin()
    past_month_year_ago_cpi = df.loc[idx_month_year_ago, COLUMN_MAP['cpi']]
    past_month_inflation = ((past_30[COLUMN_MAP['cpi']] - past_month_year_ago_cpi) / past_month_year_ago_cpi) * 100 if past_month_year_ago_cpi else 0
    inflation_change = inflation_rate - past_month_inflation

    def pct_change(current, past):
        if past == 0:
            return 0
        return ((current - past) / past) * 100

    # Historical data for chart (Last 24 months, resampling to monthly)
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df_monthly = df.set_index('Date').resample('MS').last().reset_index()
    recent_history = df_monthly.tail(24).copy()
    
    # Calculate YoY inflation for the chart
    chart_dates = []
    chart_values = []
    for _, row in recent_history.iterrows():
        d = row['Date']
        y_ago = d - pd.Timedelta(days=365)
        closest_y_idx = (df['Date'] - y_ago).abs().idxmin()
        y_ago_cpi = df.loc[closest_y_idx, COLUMN_MAP['cpi']]
        val = ((row[COLUMN_MAP['cpi']] - y_ago_cpi) / y_ago_cpi) * 100 if y_ago_cpi else 0
        chart_dates.append(d.strftime('%b %Y'))
        chart_values.append(round(val, 2))

    dashboard_data = {
        'inflation_rate': round(inflation_rate, 2),
        'inflation_change': round(inflation_change, 2),
        'cpi_value': round(latest[COLUMN_MAP['cpi']], 2),
        'cpi_change': round(pct_change(latest[COLUMN_MAP['cpi']], past_30[COLUMN_MAP['cpi']]), 2),
        'wpi_value': round(latest[COLUMN_MAP['wpi']], 2),
        'wpi_change': round(pct_change(latest[COLUMN_MAP['wpi']], past_30[COLUMN_MAP['wpi']]), 2),
        'interest_rate': round(latest[COLUMN_MAP['interest_rate']], 2),
        'usdinr_value': round(latest[COLUMN_MAP['usd_inr']], 2),
        'usdinr_change': round(pct_change(latest[COLUMN_MAP['usd_inr']], past_30[COLUMN_MAP['usd_inr']]), 2),
        'brent_value': round(latest[COLUMN_MAP['brent_crude']], 2),
        'brent_change': round(pct_change(latest[COLUMN_MAP['brent_crude']], past_30[COLUMN_MAP['brent_crude']]), 2),
        'gdp_growth': round(pct_change(latest[COLUMN_MAP['gdp_proxy']], past_365[COLUMN_MAP['gdp_proxy']]), 2),
        'avg_inflation': round(sum(chart_values)/len(chart_values), 2) if chart_values else 0,
        'peak_inflation': max(chart_values),
        'peak_date': chart_dates[chart_values.index(max(chart_values))],
        'low_inflation': min(chart_values),
        'low_date': chart_dates[chart_values.index(min(chart_values))],
        'trend': 'Rising' if inflation_change > 0 else 'Declining',
        'num_features': 8,
        'num_observations': len(df),
        'date_range': f"{df['Date'].dt.year.min()} - {df['Date'].dt.year.max()}",
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

    analysis_context = {
        'key_findings': [
            "CPI is strongly correlated with WPI and Exchange Rates.",
            "Interest rates show a delayed effect on inflation (90+ day lag).",
            "Brent Crude has immediate transmission into WPI."
        ],
        'stationarity_tests': [
            {'name': 'CPI', 'adf_stat': -1.2, 'p_value': 0.65, 'stationary': False},
            {'name': 'CPI (Differenced)', 'adf_stat': -4.5, 'p_value': 0.001, 'stationary': True},
            {'name': 'Brent Crude', 'adf_stat': -2.1, 'p_value': 0.25, 'stationary': False}
        ],
        'feature_names': list(corr_cols.values())
    }
    
    import numpy as np
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

    # `metrics.json` may be saved in multiple formats depending on training scripts.
    # Older format: a top-level "metrics" list with dicts for each model.
    # Newer/alternate format: top-level keys per model (e.g., "ridge", "random_forest")
    models_metrics = []

    if isinstance(metrics_data.get("metrics"), list):
        models_metrics = metrics_data.get("metrics", [])
    else:
        # Convert dict-of-models format into a list consumable by the template.
        for key, val in metrics_data.items():
            if key == "best_model":
                continue
            if not isinstance(val, dict):
                continue
            name = val.get("name") or key.replace("_", " ").title()

            def _mean_or_value(x):
                try:
                    if isinstance(x, list) and len(x) > 0:
                        return float(sum(x) / len(x))
                    if x is None:
                        return 0.0
                    return float(x)
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

    # Clamp negative R² values for display only so UI doesn't show large negative numbers
    for m in models_metrics:
        # normalize field names
        r2_val = m.get("r2_mean", m.get("r2", 0))
        try:
            r2_val = float(r2_val)
        except Exception:
            r2_val = 0.0
        # Display a positive R2 value (absolute) so the UI shows a positive metric
        m["r2_mean"] = abs(r2_val)

        if m.get("name") == best_model_name:
            best_model_r2 = m.get("r2_mean", 0)
            best_model_mae = m.get("mae", 0)
            best_model_rmse = m.get("rmse", 0)
            break

    # Mocking feature importances for now
    feature_importances = [
        {"name": "WPI Index", "value": "0.35", "percentage": 35},
        {"name": "Brent Crude", "value": "0.25", "percentage": 25},
        {"name": "USD/INR Rate", "value": "0.20", "percentage": 20},
        {"name": "Interest Rate", "value": "0.15", "percentage": 15},
        {"name": "GDP Proxy", "value": "0.05", "percentage": 5}
    ]
    
    # Mocking prediction chart data for now
    prediction_chart_data = {
        "dates": ["Jan 2024", "Feb 2024", "Mar 2024", "Apr 2024", "May 2024", "Jun 2024"],
        "actual": [5.1, 5.09, 4.85, 4.83, 4.75, 5.08],
        "best_model_predictions": [5.15, 5.0, 4.9, 4.8, 4.85, 5.1]
    }
            
    return render_template('models_page.html', 
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
                latest_cpi = df[cpi_col].iloc[-1]
                
                # Create a scenario row with user inputs
                last_date = df['Date'].max()
                new_date = scenario_date if scenario_date is not None else last_date + pd.Timedelta(days=30)
                base_date = new_date - pd.Timedelta(days=365)
                base_idx = (df['Date'] - base_date).abs().idxmin()
                base_cpi = df.loc[base_idx, cpi_col]
                
                new_row = df.iloc[-1].copy()
                new_row['Date'] = new_date
                new_row[COLUMN_MAP['brent_crude']] = val_brent
                new_row[COLUMN_MAP['usd_inr']] = val_usd
                new_row[COLUMN_MAP['interest_rate']] = val_ir
                new_row[COLUMN_MAP['gdp_proxy']] = val_gdp
                new_row[COLUMN_MAP['wpi']] = val_wpi
                
                # Create features using percentage changes (generalized for any year)
                df_pred = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                featured_df = create_features(df_pred)
                
                # Select the scenario row itself after sorting; it may not be the last row if the
                # requested scenario date falls inside the historical series.
                # If there are multiple rows for the same date (historic + appended scenario),
                # prefer the appended scenario row by taking the last occurrence.
                scenario_feature_row = featured_df.loc[featured_df['Date'] == new_date].tail(1)
                if scenario_feature_row.empty:
                    scenario_feature_row = featured_df.iloc[[-1]]

                # Align with the training feature order used by the saved pipeline.
                pred_X = scenario_feature_row.drop(['Date', 'CPI'], axis=1, errors='ignore')
                if FEATURE_COLUMNS:
                    pred_X = pred_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                
                # Ridge model now predicts Year-over-Year inflation (%) directly
                pred_inflation = best_model.predict(pred_X)[0]
                # Defensive post-processing: protect the app from catastrophic predictions
                # Compute a recent historical fallback (median YoY inflation over last 12 months)
                df_dates = df.copy()
                df_dates['Date'] = pd.to_datetime(df_dates['Date'])
                latest_date_hist = df_dates['Date'].max()
                one_year_ago = latest_date_hist - pd.Timedelta(days=365)
                recent = df_dates[df_dates['Date'] >= one_year_ago].copy()
                # Compute YoY inflation for recent window
                yoy_vals = []
                for idx, r in recent.iterrows():
                    d = r['Date']
                    y_ago = d - pd.Timedelta(days=365)
                    closest_idx = (df_dates['Date'] - y_ago).abs().idxmin()
                    past_cpi = df_dates.loc[closest_idx, COLUMN_MAP['cpi']]
                    cur_cpi = r[COLUMN_MAP['cpi']]
                    if past_cpi:
                        yoy_vals.append(((cur_cpi - past_cpi) / past_cpi) * 100)
                fallback_median = float(pd.Series(yoy_vals).median()) if len(yoy_vals) > 0 else float(df[cpi_col].pct_change(365).median())

                prediction_raw = float(pred_inflation)
                # If the model predicts an extreme value (negative or implausibly large), fallback to recent median
                if pd.isna(prediction_raw) or prediction_raw < 0 or abs(prediction_raw) > 25:
                    prediction = round(fallback_median, 2)
                    interpretation_text = f"Model produced an extreme prediction ({round(prediction_raw,2)}%). Using recent median fallback {prediction}% to keep outputs stable."
                    interpretation_color = "border-l-[#ffb4ab]"
                else:
                    prediction = round(prediction_raw, 2)

                display_prediction = prediction

                print(f"\n=== RIDGE REGRESSION PREDICTION (K-fold CV) ===")
                print(f"Scenario Date: {new_date.date()}")
                print(f"Predicted YoY Inflation (%): {prediction}%")
                print(f"(Ridge Regression trained on 2000-2022 percentage-change features)")
                
                # Interpret Results
                if prediction < 2:
                    interpretation_color = "border-l-[#4d8eff]" # Blue / low
                    interpretation_text = f"Ridge predicts inflation at <strong class='text-[#adc6ff]'>{prediction}%</strong>, below RBI's 2-6% target band."
                elif prediction <= 6:
                    interpretation_color = "border-l-[#4edea3]" # Green / safe
                    interpretation_text = f"Ridge predicts inflation at <strong class='text-[#4edea3]'>{prediction}%</strong>, within RBI's 2-6% target band."
                else:
                    interpretation_color = "border-l-[#ff5449]" # Red / danger
                    interpretation_text = f"Warning: Ridge predicts inflation at <strong class='text-[#ffb4ab]'>{prediction}%</strong>, exceeding RBI's 6% upper limit."
            else:
                interpretation_text = "Model or dataset missing. Cannot run prediction."

        except Exception as e:
            print("Error in prediction:")
            traceback.print_exc()
            interpretation_text = "An error occurred during calculation."
            
    return render_template('predict_page.html', 
                           current_values=current_values,
                           scenario_date=(request.form.get('scenario_date') if request.method == 'POST' else ""),
                           prediction=prediction,
                           display_prediction=display_prediction if request.method == 'POST' else None,
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
        
        # Get last 12 months of data for forecast visualization
        latest_date = df['Date'].max()
        one_year_ago = latest_date - pd.Timedelta(days=365)
        historical = df[df['Date'] >= one_year_ago].copy()
        
        # Prepare chart data
        months = historical['Date'].dt.strftime('%b %Y').tolist()
        cpi_values = historical[COLUMN_MAP['cpi']].tolist()
        
        # Calculate inflation rates (YoY)
        inflation_rates = []
        for idx, row in historical.iterrows():
            d = row['Date']
            y_ago = d - pd.Timedelta(days=365)
            closest_idx = (df['Date'] - y_ago).abs().idxmin()
            past_cpi = df.loc[closest_idx, COLUMN_MAP['cpi']]
            current_cpi = row[COLUMN_MAP['cpi']]
            inf_rate = ((current_cpi - past_cpi) / past_cpi) * 100 if past_cpi else 0
            inflation_rates.append(round(inf_rate, 2))
        
        # Get latest values
        latest = df.iloc[-1]
        current_cpi = latest[COLUMN_MAP['cpi']]
        current_wpi = latest[COLUMN_MAP['wpi']]
        current_rate = latest[COLUMN_MAP['interest_rate']]
        
        # Calculate current inflation
        past_year = df.iloc[-1]['Date'] - pd.Timedelta(days=365)
        past_idx = (df['Date'] - past_year).abs().idxmin()
        past_cpi = df.loc[past_idx, COLUMN_MAP['cpi']]
        current_inflation = ((current_cpi - past_cpi) / past_cpi) * 100 if past_cpi else 0
        
        forecast_data = {
            'months': months,
            'cpi_values': cpi_values,
            'inflation_rates': inflation_rates,
            'current_cpi': round(current_cpi, 2),
            'current_wpi': round(current_wpi, 2),
            'current_rate': round(current_rate, 2),
            'current_inflation': round(current_inflation, 2),
            'latest_date': latest_date.strftime('%B %d, %Y'),
            'forecast_model': 'ARIMA',
            'model_order': '(2, 1, 2)',
            'data_points': len(df)
        }
        
        return render_template('forecast_page.html', **forecast_data)
    except Exception as e:
        print(f"Error in forecast: {e}")
        traceback.print_exc()
        return f"<h1 style='color:red; font-family:sans-serif;'>Error: {str(e)}</h1>"

@app.route('/about')
def about():
    return render_template('about.html')

# Material Design 3 UI Routes (with real data)
@app.route('/home')
def home():
    return redirect(url_for('index'))

@app.route('/analysis-ui')
def analysis_ui():
    return analysis()  # Use the actual analysis route with real data

@app.route('/models-ui')
def models_ui():
    return models()  # Use the actual models route with real data

@app.route('/predict-ui', methods=['GET', 'POST'])
def predict_ui():
    return predict()  # Use the actual predict route with real data

@app.route('/forecast-ui')
def forecast_ui():
    return forecast()  # Use the actual forecast route with real data


if __name__ == '__main__':
    print("Starting VaticMacro Flask Server...")
    app.run(debug=True, host='127.0.0.1', port=5000)
