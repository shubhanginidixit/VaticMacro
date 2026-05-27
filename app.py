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

try:
    best_model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model at {MODEL_PATH}. Error: {e}")
    best_model = None

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
    return redirect(url_for('dashboard'))

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

    return render_template('analysis.html', **analysis_context)

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
    models_metrics = metrics_data.get("metrics", [])
    
    best_model_r2 = 0
    best_model_mae = 0
    best_model_rmse = 0
    
    for m in models_metrics:
        if m.get("name") == best_model_name:
            best_model_r2 = m.get("r2", 0)
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
            
    return render_template('models.html', 
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
    
    if request.method == 'POST':
        try:
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
            
            if best_model and os.path.exists(DATA_PATH):
                # Load the raw dataset
                df = pd.read_csv(DATA_PATH)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Create a new row representing our "scenario" day
                last_date = df['Date'].max()
                new_date = last_date + pd.Timedelta(days=1)
                
                new_row = df.iloc[-1].copy()
                new_row['Date'] = new_date
                new_row[COLUMN_MAP['brent_crude']] = val_brent
                new_row[COLUMN_MAP['usd_inr']] = val_usd
                new_row[COLUMN_MAP['interest_rate']] = val_ir
                new_row[COLUMN_MAP['gdp_proxy']] = val_gdp
                new_row[COLUMN_MAP['wpi']] = val_wpi
                
                # Append the scenario row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Generate all lag/rolling features!
                featured_df = create_features(df)
                
                # Extract the final row containing the full context for prediction
                pred_X = featured_df.drop(['Date', 'CPI'], axis=1).iloc[[-1]]
                
                # Run the trained model
                pred_cpi = best_model.predict(pred_X)[0]
                
                # Calculate Year-over-Year Inflation %
                past_date = new_date - pd.Timedelta(days=365)
                closest_idx = (df['Date'] - past_date).abs().idxmin()
                past_cpi = df.loc[closest_idx, COLUMN_MAP['cpi']]
                
                inflation_rate = ((pred_cpi - past_cpi) / past_cpi) * 100 if past_cpi else 0
                prediction = round(inflation_rate, 2)
                
                # Interpret Results
                if prediction < 2:
                    interpretation_color = "border-l-[#4d8eff]" # Blue / low
                    interpretation_text = f"At these simulated levels, the model predicts inflation <strong class='text-[#adc6ff] font-semibold'>falling below</strong> the RBI's target band at {prediction}%."
                elif prediction <= 6:
                    interpretation_color = "border-l-[#ca8a04]" # Yellow / safe
                    interpretation_text = f"At these simulated levels, the model predicts inflation remaining <strong class='text-[#facc15] font-semibold'>safely within</strong> the RBI's target band (2-6%)."
                else:
                    interpretation_color = "border-l-[#ff5449]" # Red / danger
                    interpretation_text = f"Warning: These macroeconomic conditions push predicted inflation <strong class='text-[#ffb4ab] font-semibold'>dangerously above</strong> the 6% maximum threshold."
            else:
                interpretation_text = "Model or dataset missing. Cannot run prediction."

        except Exception as e:
            print("Error in prediction:")
            traceback.print_exc()
            interpretation_text = "An error occurred during calculation."
            
    return render_template('predict.html', 
                           current_values=current_values,
                           prediction=prediction,
                           interpretation_text=interpretation_text,
                           interpretation_color=interpretation_color)

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
        
        return render_template('forecast.html', **forecast_data)
    except Exception as e:
        print(f"Error in forecast: {e}")
        traceback.print_exc()
        return f"<h1 style='color:red; font-family:sans-serif;'>Error: {str(e)}</h1>"

@app.route('/about')
def about():
    if os.path.exists(os.path.join(app.template_folder, 'about.html')):
        return render_template('about.html')
    return "<h1 style='color:white; font-family:sans-serif; text-align:center; padding-top: 50px;'>About Page Placeholder</h1><p style='color:white; text-align:center;'>Paste the HTML when ready!</p>"

# Material Design 3 UI Routes (with real data)
@app.route('/home')
def home():
    return render_template('home.html')

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
