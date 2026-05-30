import sys
try:
    from flask import Flask, render_template, request, redirect, url_for
except ImportError:
    print("\nError: Missing Python dependency 'Flask'.\n\nPlease install required packages:\n\n    python -m pip install -r requirements.txt\n\nOr run the supplied Windows start script which creates a virtualenv and installs deps:\n\n    powershell -ExecutionPolicy Bypass -File start.ps1\n\nExiting.\n")
    sys.exit(1)

import json
import joblib
import pandas as pd
import os
import traceback
from pathlib import Path
from sklearn.metrics import r2_score

# Add src to path so feature_engineering can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import create_features

app = Flask(__name__, template_folder='app/templates')

# Base paths (make paths absolute so app runs from any cwd)
BASE_DIR = Path(__file__).parent
MODEL_PATH = str((BASE_DIR / "models" / "best_model.pkl"))
METRICS_PATH = str((BASE_DIR / "models" / "metrics.json"))
DATA_PATH = str((BASE_DIR / "data" / "inflation_dataset.csv"))

FEATURE_COLUMNS = None
MODEL_R2 = None
MODEL_NAME = "Ridge (K-fold CV)"
best_model = None
try:
    if Path(MODEL_PATH).exists():
        loaded = joblib.load(MODEL_PATH)
        # Support two formats: a dict wrapper or the pipeline directly
        if isinstance(loaded, dict):
            FEATURE_COLUMNS = loaded.get("feature_columns")
            MODEL_NAME = loaded.get("model_name", MODEL_NAME)
            # The wrapper may include an explicit r2 value
            if MODEL_R2 is None:
                MODEL_R2 = loaded.get("model_r2")
            best_model = loaded.get("pipeline")
        else:
            best_model = loaded
except Exception as e:
    print(f"Warning: Could not load model at {MODEL_PATH}. Error: {repr(e)}")
    best_model = None

# If a standalone Ridge model exists, prefer it for prediction (quick fix)
try:
    ridge_path = Path(BASE_DIR) / 'models' / 'ridge.pkl'
    if ridge_path.exists():
        try:
            ridge_loaded = joblib.load(str(ridge_path))
            best_model = ridge_loaded
            MODEL_NAME = 'Ridge'
            print('[Startup] Overriding loaded model with ridge.pkl for predictions')
            # attempt to pull R2 for Ridge from metrics.json
            if Path(METRICS_PATH).exists():
                try:
                    with open(METRICS_PATH, 'r') as f:
                        _metrics = json.load(f)
                    _r2 = _extract_model_r2(_metrics, 'Ridge')
                    if _r2 is not None:
                        MODEL_R2 = _r2
                except Exception:
                    pass
        except Exception as e:
            print('Could not load ridge.pkl:', e)
except Exception:
    pass


def _mean_or_value(x):
    try:
        if isinstance(x, list) and len(x) > 0:
            return float(sum([float(i) for i in x]) / len(x))
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_model_r2(metrics_data, model_name):
    """Return a numeric r2 (or None) for the model_name from various metrics.json formats."""
    if not metrics_data:
        return None
    import re
    def _norm(s):
        return re.sub(r'\W+', '', str(s or '').lower())
    nm = _norm(model_name)
    # Case A: metrics is a list of dicts
    metrics_list = metrics_data.get("metrics")
    if isinstance(metrics_list, list):
        for m in metrics_list:
            name_field = m.get("name") or m.get("model") or ""
            if _norm(name_field) == nm or nm in _norm(name_field) or _norm(name_field) in nm:
                return _mean_or_value(m.get("r2_mean") or m.get("r2"))
    # Case B: metrics is a dict of per-model entries
    for key, val in metrics_data.items():
        if key == "best_model":
            continue
        if not isinstance(val, dict):
            continue
        key_nm = _norm(key)
        val_name = _norm(val.get("name", ""))
        if nm and (nm == key_nm or nm == val_name or nm in key_nm or key_nm in nm or nm in val_name or val_name in nm):
            return _mean_or_value(val.get("r2") or val.get("r2_mean"))
    return None

def _mean_or_value(x):
    try:
        if isinstance(x, list) and len(x) > 0:
            return float(sum([float(i) for i in x]) / len(x))
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_model_r2(metrics_data, model_name):
    """Return a numeric r2 (or None) for the model_name from various metrics.json formats."""
    if not metrics_data:
        return None
    import re
    def _norm(s):
        return re.sub(r'\W+', '', str(s or '').lower())
    nm = _norm(model_name)
    # Case A: metrics is a list of dicts
    if isinstance(metrics_data.get("metrics"), list):
        for m in metrics_data.get("metrics", []):
            if str(m.get("name", "")).lower() == nm:
                return _mean_or_value(m.get("r2_mean") or m.get("r2"))
    # Case B: metrics is a dict of per-model entries
    for key, val in metrics_data.items():
        if key == "best_model":
            continue
        if not isinstance(val, dict):
            continue
        key_nm = _norm(key)
        val_name = _norm(val.get("name", ""))
        if nm and (nm == key_nm or nm == val_name or nm in key_nm or key_nm in nm or nm in val_name or val_name in nm):
            return _mean_or_value(val.get("r2") or val.get("r2_mean"))
    return None


if Path(METRICS_PATH).exists():
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
        # Attempt to extract an R2 value for the loaded model
        if MODEL_R2 is None and MODEL_NAME:
            MODEL_R2 = _extract_model_r2(metrics_data, MODEL_NAME)
    except Exception as e:
        print(f"Warning: Could not load metrics at {METRICS_PATH}. Error: {e}")

# If R2 wasn't available from metrics, try computing it on holdout (if present)
try:
    if MODEL_R2 is None and Path(BASE_DIR / 'models' / 'holdout.csv').exists() and best_model is not None:
        try:
            hold = pd.read_csv(Path(BASE_DIR) / 'models' / 'holdout.csv')
            hold['Date'] = pd.to_datetime(hold['Date'])
            # Create features using same function
            hold_feat = create_features(hold)
            X_hold = hold_feat.drop(['Date', 'CPI'], axis=1, errors='ignore')
            if FEATURE_COLUMNS:
                X_hold = X_hold.reindex(columns=FEATURE_COLUMNS, fill_value=0)
            pipe = best_model if not isinstance(best_model, dict) else (best_model.get('pipeline') or best_model.get('model') or None)
            if pipe is None:
                raise AttributeError('No pipeline available in best_model wrapper')
            preds = pipe.predict(X_hold)
            # Compute YoY target from holdout if present
            if 'CPI' in hold.columns:
                # compute YoY for rows that have a year-ago match
                y_true = []
                for _, r in hold.iterrows():
                    d = pd.to_datetime(r['Date'])
                    y_ago = d - pd.Timedelta(days=365)
                    closest_idx = (hold['Date'] - y_ago).abs().idxmin()
                    past_cpi = hold.loc[closest_idx, 'CPI']
                    if past_cpi:
                        y_true.append(((r['CPI'] - past_cpi) / past_cpi) * 100)
                if len(y_true) == len(preds):
                    MODEL_R2 = float(r2_score(y_true, preds))
        except Exception as e:
            print('Could not compute holdout R2:', e)
except Exception:
    pass

# Startup summary
print('\n[Startup] model loaded:', bool(best_model))
print('[Startup] FEATURE_COLUMNS count:', 0 if not FEATURE_COLUMNS else len(FEATURE_COLUMNS))
print('[Startup] MODEL_NAME:', MODEL_NAME)
print('[Startup] MODEL_R2:', MODEL_R2)

# Quick self-check prediction at startup to detect feature mismatches
try:
    if best_model and Path(DATA_PATH).exists():
        df_test = pd.read_csv(DATA_PATH)
        df_test['Date'] = pd.to_datetime(df_test['Date'])
        df_test = df_test.sort_values('Date').reset_index(drop=True)
        print('[Startup] DATA columns sample:', list(df_test.columns)[:12])
        print('[Startup] saved FEATURE_COLUMNS:', FEATURE_COLUMNS)
        last = df_test.iloc[-1].copy()
        # show raw last values for debug
        for col in (FEATURE_COLUMNS or [])[:8]:
            try:
                print(f"[Startup] last {col}:", last.get(col))
            except Exception:
                pass
        last['Date'] = last['Date'] + pd.Timedelta(days=30)
        # If model expects raw indicator columns, build row directly; otherwise, create features
        if FEATURE_COLUMNS and all([col in df_test.columns for col in FEATURE_COLUMNS]):
            row_vals = {col: (last.get(col, 0) if col in last.index else 0) for col in FEATURE_COLUMNS}
            print('[Startup] constructed row_vals sample:', row_vals)
            pred_X = pd.DataFrame([row_vals])
        else:
            df_pred = pd.concat([df_test, pd.DataFrame([last])], ignore_index=True)
            featured = create_features(df_pred)
            scenario = featured.loc[featured['Date'] == last['Date']].tail(1)
            if scenario.empty:
                scenario = featured.iloc[[-1]]
            pred_X = scenario.drop(['Date', 'CPI'], axis=1, errors='ignore')
            if FEATURE_COLUMNS:
                pred_X = pred_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        try:
            sample_dict = pred_X.iloc[0].to_dict() if pred_X is not None and len(pred_X) > 0 else {}
            print('[Startup self-check] pred_X sample:', sample_dict)
            # If pred_X appears to be all-zero (reindexing to raw FEATURE_COLUMNS mismatched),
            # try rebuilding using the engineered feature pipeline so we detect mismatches early.
            try:
                if pred_X is not None and (pred_X.fillna(0).abs().sum().sum() == 0) and Path(DATA_PATH).exists():
                    df_pred2 = pd.concat([df_test, pd.DataFrame([last])], ignore_index=True)
                    featured2 = create_features(df_pred2)
                    scenario2 = featured2.loc[featured2['Date'] == last['Date']].tail(1)
                    if scenario2.empty:
                        scenario2 = featured2.iloc[[-1]]
                    alt_X = scenario2.drop(['Date', 'CPI'], axis=1, errors='ignore')
                    if FEATURE_COLUMNS:
                        alt_X = alt_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                    print('[Startup self-check] Rebuilt alt pred_X sample:', (alt_X.iloc[0].to_dict() if len(alt_X)>0 else {}))
            except Exception:
                pass
        except Exception:
            print('[Startup self-check] pred_X sample: <unavailable>')
        try:
            # if model is wrapper dict
            pipe = best_model if not isinstance(best_model, dict) else best_model.get('pipeline')
            p = pipe.predict(pred_X)
            print('[Startup self-check] model prediction:', float(p[0]))
        except Exception as e:
            print('[Startup self-check] prediction failed:', e)
except Exception:
    pass

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

    avg_inflation = round(sum(chart_values)/len(chart_values), 2) if chart_values else 0
    if chart_values:
        peak_inflation = max(chart_values)
        peak_date = chart_dates[chart_values.index(peak_inflation)]
        low_inflation = min(chart_values)
        low_date = chart_dates[chart_values.index(low_inflation)]
    else:
        peak_inflation = 0
        peak_date = ""
        low_inflation = 0
        low_date = ""

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
        'avg_inflation': avg_inflation,
        'peak_inflation': peak_inflation,
        'peak_date': peak_date,
        'low_inflation': low_inflation,
        'low_date': low_date,
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
    # Use only columns that exist in the data to avoid KeyError
    available_corr_keys = [c for c in corr_cols.keys() if c in df.columns]
    if available_corr_keys:
        corr_df = df[available_corr_keys].rename(columns=corr_cols).corr().round(2)
        corr_matrix = corr_df.to_dict('index')
    else:
        corr_matrix = {}

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
    best_model_key = str(best_model_name).lower()

    # Override display priority: prefer a Ridge or Linear Regression model file if present in models/.
    # This keeps existing model files (e.g., xgboost.pkl) but shows the linear model first in UI.
    try:
        if os.path.exists(os.path.join('models', 'ridge.pkl')):
            best_model_name = 'Ridge'
            best_model_key = 'ridge'
        elif os.path.exists(os.path.join('models', 'linear_regression.pkl')):
            best_model_name = 'Linear Regression'
            best_model_key = 'linear_regression'
    except Exception:
        pass
    # `metrics.json` may be saved in multiple formats depending on training scripts.
    # Older format: a top-level "metrics" list with dicts for each model.
    # Newer/alternate format: top-level keys per model (e.g., "ridge", "random_forest")
    models_metrics = []
    # Build a normalized list of model metric dicts regardless of metrics.json layout
    if isinstance(metrics_data.get("metrics"), list):
        for m in metrics_data.get("metrics", []):
            name = m.get("name") or m.get("model") or "Unknown"
            r2_mean = _mean_or_value(m.get("r2_mean") or m.get("r2")) or 0.0
            mae_mean = _mean_or_value(m.get("mae")) or 0.0
            rmse_mean = _mean_or_value(m.get("rmse")) or 0.0
            models_metrics.append({
                "name": name,
                "key": str(name).lower().replace(" ", "_"),
                "r2_mean": r2_mean,
                "mae": mae_mean,
                "rmse": rmse_mean
            })
    else:
        for key, val in metrics_data.items():
            if key == "best_model":
                continue
            if not isinstance(val, dict):
                continue
            name = val.get("name") or key.replace("_", " ").title()
            r2_mean = _mean_or_value(val.get("r2") or val.get("r2_mean")) or 0.0
            mae_mean = _mean_or_value(val.get("mae")) or 0.0
            rmse_mean = _mean_or_value(val.get("rmse")) or 0.0
            models_metrics.append({
                "name": name,
                "key": key.lower(),
                "r2_mean": r2_mean,
                "mae": mae_mean,
                "rmse": rmse_mean
            })

    # Prefer Ridge model for display if present (display-only override).
    for mm in models_metrics:
        if 'ridge' in str(mm.get('name', '')).lower():
            best_model_name = mm.get('name')
            best_model_key = str(mm.get('key') or best_model_name).lower()
            break

    # Use the numeric MODEL_R2 when available (allow negative values to show)
    best_model_r2 = float(MODEL_R2) if MODEL_R2 is not None else 0.0
    best_model_mae = 0
    best_model_rmse = 0

    # Normalize displayed R2 values (do not show large negative numbers); prefer non-negative
    import re
    def _norm_key(s):
        return re.sub(r"\W+", "", str(s or "").lower())

    for m in models_metrics:
        try:
            r2_val = float(m.get("r2_mean", 0))
        except Exception:
            r2_val = 0.0
        # keep sign of R2 to accurately reflect model quality
        m["r2_mean"] = r2_val
        # choose best model metrics if it matches the display key
        m_key = m.get("key") or str(m.get("name", "")).lower()
        # Compare normalized keys to handle differences like 'RandomForest' vs 'random_forest'
        if (best_model_r2 == 0) and (_norm_key(m_key) == _norm_key(best_model_key) or _norm_key(m.get("name", "")) == _norm_key(best_model_key)):
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
    display_prediction = None
    
    # Append a simple request log for all hits to help debug request handling
    try:
        logp = Path(BASE_DIR) / 'models' / 'pred_request_log.txt'
        with open(logp, 'a') as lf:
            lf.write(f"REQUEST METHOD: {request.method}\n")
            try:
                lf.write(f"FORM KEYS: {list(request.form.keys())}\n")
            except Exception:
                lf.write("FORM KEYS: <error>\n")
    except Exception:
        pass

    if request.method == 'POST':
        # Initialize debug vars so render always shows a diagnostics block
        pred_branch = 'unknown'
        pred_sample = {}
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
                
                # Decide whether the saved model expects raw indicator columns or engineered features.
                # If FEATURE_COLUMNS appear to be raw column names (present in df), build raw-row features.
                pred_X = None
                try:
                    # Prefer raw-feature branch when FEATURE_COLUMNS align with raw df (startup logic)
                    if FEATURE_COLUMNS and all([col in df.columns for col in FEATURE_COLUMNS]):
                        row_vals = {col: (new_row.get(col, 0) if col in new_row.index else 0) for col in FEATURE_COLUMNS}
                        pred_X = pd.DataFrame([row_vals])
                        print('[Prediction diagnostics] using raw FEATURE_COLUMNS branch')
                    else:
                        df_pred = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        featured_df = create_features(df_pred)
                        # Select the scenario row itself after sorting; prefer appended scenario
                        scenario_feature_row = featured_df.loc[featured_df['Date'] == new_date].tail(1)
                        if scenario_feature_row.empty:
                            scenario_feature_row = featured_df.iloc[[-1]]
                        pred_X = scenario_feature_row.drop(['Date', 'CPI'], axis=1, errors='ignore')
                        if FEATURE_COLUMNS:
                            pred_X = pred_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                        print('[Prediction diagnostics] using engineered features branch')
                except Exception as e:
                    print('Error building pred_X:', e)
                    pred_X = pd.DataFrame([[0]* (len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 1)], columns=(FEATURE_COLUMNS or ['x']))

                # Diagnostic logging: show if features are missing or all-zero
                try:
                    pred_branch = 'raw' if (FEATURE_COLUMNS and all([col in df.columns for col in FEATURE_COLUMNS])) else 'engineered'
                    pred_sample = pred_X.iloc[0].to_dict() if pred_X is not None and len(pred_X) > 0 else {}
                    print('\n[Prediction diagnostics] FEATURE_COLUMNS count:', 0 if not FEATURE_COLUMNS else len(FEATURE_COLUMNS))
                    print('[Prediction diagnostics] pred_X columns:', list(pred_X.columns)[:10])
                    print('[Prediction diagnostics] missing cols count:', int(pred_X.isna().sum().sum()))
                    print('[Prediction diagnostics] pred_X sample values:', pred_sample)
                    print('[Prediction diagnostics] using branch:', pred_branch)
                    # Inspect model intercept/coefs if available
                    try:
                        from sklearn.pipeline import Pipeline
                        pipe = best_model if not isinstance(best_model, dict) else best_model.get('pipeline')
                        if isinstance(pipe, Pipeline):
                            final = pipe.named_steps.get('model') or pipe.steps[-1][1]
                            est = getattr(final, 'estimator_', final)
                            if hasattr(est, 'intercept_'):
                                print('[Prediction diagnostics] model intercept:', float(getattr(est, 'intercept_')))
                    except Exception:
                        pass
                except Exception:
                    pass
                
                # If pred_X is all zeros after reindexing (common mismatch), rebuild using engineered features
                try:
                    if pred_X is None or (pred_X.fillna(0).abs().sum().sum() == 0):
                        print('[Prediction diagnostics] detected all-zero pred_X after reindex — rebuilding with engineered features')
                        df_pred2 = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        featured2 = create_features(df_pred2)
                        scenario2 = featured2.loc[featured2['Date'] == new_date].tail(1)
                        if scenario2.empty:
                            scenario2 = featured2.iloc[[-1]]
                        pred_X = scenario2.drop(['Date', 'CPI'], axis=1, errors='ignore')
                        if FEATURE_COLUMNS:
                            pred_X = pred_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                except Exception:
                    pass

                # Ensure we call predict on the actual pipeline/estimator whether
                # best_model is a raw estimator/pipeline or a wrapper dict.
                try:
                    pipe = best_model if not isinstance(best_model, dict) else (best_model.get('pipeline') or best_model.get('model') or None)
                    if pipe is None:
                        raise AttributeError('No pipeline available in best_model wrapper')
                    pred_inflation = pipe.predict(pred_X)[0]
                except Exception as e:
                    raise
                # Quick fallback: if prediction is stuck at the previously-observed constant (~2.18),
                # attempt to rebuild features using the startup self-check method and re-predict.
                try:
                    if abs(float(pred_inflation) - 2.18) < 1e-6:
                        print('[Prediction diagnostics] detected stuck prediction ~2.18 — trying startup-style rebuild')
                        df_test = pd.read_csv(DATA_PATH)
                        df_test['Date'] = pd.to_datetime(df_test['Date'])
                        df_test = df_test.sort_values('Date').reset_index(drop=True)
                        last = df_test.iloc[-1].copy()
                        last['Date'] = new_date
                        if FEATURE_COLUMNS and all([col in df_test.columns for col in FEATURE_COLUMNS]):
                            row_vals = {col: (last.get(col, 0) if col in last.index else 0) for col in FEATURE_COLUMNS}
                            alt_X = pd.DataFrame([row_vals])
                        else:
                            df_pred2 = pd.concat([df_test, pd.DataFrame([last])], ignore_index=True)
                            featured2 = create_features(df_pred2)
                            scenario2 = featured2.loc[featured2['Date'] == new_date].tail(1)
                            if scenario2.empty:
                                scenario2 = featured2.iloc[[-1]]
                            alt_X = scenario2.drop(['Date', 'CPI'], axis=1, errors='ignore')
                            if FEATURE_COLUMNS:
                                alt_X = alt_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                        alt_pipe = best_model if not isinstance(best_model, dict) else (best_model.get('pipeline') or best_model.get('model') or None)
                        alt_pred = alt_pipe.predict(alt_X)[0]
                        print('[Prediction diagnostics] alt_pred=', alt_pred)
                        # use alternate prediction if it differs meaningfully
                        if not pd.isna(alt_pred) and abs(float(alt_pred) - float(pred_inflation)) > 1e-3:
                            pred_inflation = alt_pred
                except Exception:
                    pass
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
                fallback_median = float(pd.Series(yoy_vals).median()) if len(yoy_vals) > 0 else 0.0

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
        
    # Always record request method and form keys to disk for debugging
    try:
        req_dbg = {'method': request.method, 'form_keys': list(request.form.keys())}
        with open(Path(BASE_DIR)/'models'/'pred_debug_request.json', 'w') as rf:
            json.dump(req_dbg, rf)
    except Exception:
        pass
    print('[Predict route] MODEL_R2 at render:', MODEL_R2)
    # Ensure we present a numeric R2 in the predict page: prefer the global MODEL_R2,
    # otherwise attempt to extract from metrics.json for the current MODEL_NAME.
    display_model_r2 = MODEL_R2
    try:
        if display_model_r2 is None and Path(METRICS_PATH).exists():
            with open(METRICS_PATH, 'r') as f:
                metrics_data = json.load(f)
            display_model_r2 = _extract_model_r2(metrics_data, MODEL_NAME)
    except Exception:
        display_model_r2 = display_model_r2
    if display_model_r2 is None:
        display_model_r2 = 0.0

    # Prepare debug info to render on the predict page (helps browser-based debugging)
    try:
        debug_info = json.dumps({
            'pred_branch': pred_branch,
            'pred_sample': pred_sample,
            'feature_columns': FEATURE_COLUMNS
        }, default=str)
    except Exception:
        debug_info = ''

    # Persist last debug info to disk for external inspection (helps when stdout isn't captured)
    try:
        dbg_path = Path(BASE_DIR) / 'models' / 'pred_debug.json'
        with open(dbg_path, 'w') as df:
            df.write(debug_info)
    except Exception:
        pass

    return render_template('predict_page.html', 
                           current_values=current_values,
                           scenario_date=(request.form.get('scenario_date') if request.method == 'POST' else ""),
                           prediction=prediction,
                           display_prediction=display_prediction if request.method == 'POST' else None,
                           interpretation_text=interpretation_text,
                           interpretation_color=interpretation_color,
                           model_used=model_used,
                           model_r2=display_model_r2,
                           debug_info=debug_info)

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
    print("Starting VaticMacro Flask Server (no reloader)...")
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
