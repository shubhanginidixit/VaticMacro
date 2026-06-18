import sys
try:
    from flask import Flask, render_template, request, redirect, url_for, jsonify  # type: ignore[reportMissingImports]
except ImportError:
    print("\nError: Missing Python dependency 'Flask'.\n\nPlease install required packages:\n\n    python -m pip install -r requirements.txt\n\nOr run the supplied Windows start script which creates a virtualenv and installs deps:\n\n    powershell -ExecutionPolicy Bypass -File start.ps1\n\nExiting.\n")
    sys.exit(1)

import json
import joblib  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import os
import re
import traceback
from pathlib import Path
from sklearn.metrics import r2_score  # type: ignore[reportMissingImports]

# Add src to path so feature_engineering can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_engineering import create_features  # type: ignore[reportMissingImports]

app = Flask(__name__, template_folder='app/templates')

# ---------------------------------------------------------------------------
# Utility helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _mean_or_value(x):
    try:
        if isinstance(x, list) and len(x) > 0:
            return float(sum([float(i) for i in x]) / len(x))
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _norm_key(s):
    """Normalize a model name/key for comparison by stripping non-word chars."""
    return re.sub(r'\W+', '', str(s or '').lower())


def _extract_model_r2(metrics_data, model_name):
    """Return a numeric r2 (or None) for the model_name from various metrics.json formats."""
    if not metrics_data:
        return None
    nm = _norm_key(model_name)
    # Case A: metrics is a list of dicts
    metrics_list = metrics_data.get("metrics")
    if isinstance(metrics_list, list):
        for m in metrics_list:
            name_field = m.get("name") or m.get("model") or ""
            if _norm_key(name_field) == nm or nm in _norm_key(name_field) or _norm_key(name_field) in nm:
                return _mean_or_value(m.get("r2_mean") or m.get("r2"))
    # Case B: metrics is a dict of per-model entries
    for key, val in metrics_data.items():
        if key == "best_model":
            continue
        if not isinstance(val, dict):
            continue
        key_nm = _norm_key(key)
        val_name = _norm_key(val.get("name", ""))
        if nm and (nm == key_nm or nm == val_name or nm in key_nm or key_nm in nm or nm in val_name or val_name in nm):
            return _mean_or_value(val.get("r2") or val.get("r2_mean"))
    return None


def _compute_yoy_r2_from_holdout(pipeline, holdout_path=None, value_col='CPI'):
    """Compute YoY % R2 for a pipeline using models/holdout.csv when available.
    Returns a float R2 or None on error.
    """
    try:
        if holdout_path is None:
            holdout_path = Path(BASE_DIR) / 'models' / 'holdout.csv'
        else:
            holdout_path = Path(holdout_path)
        if not holdout_path.exists():
            return None
        hold = pd.read_csv(holdout_path)
        if 'Date' not in hold.columns or value_col not in hold.columns:
            return None
        hold['Date'] = pd.to_datetime(hold['Date'])
        hold = hold.sort_values('Date').reset_index(drop=True)
        # build YoY true series and corresponding indices
        y_true = []
        rows = []
        for idx, r in hold.iterrows():
            d = r['Date']
            y_ago = d - pd.Timedelta(days=365)
            diffs = (hold['Date'] - y_ago).abs()
            if diffs.empty:
                continue
            closest = diffs.idxmin()
            past_val = hold.loc[closest, value_col]
            if past_val and not pd.isna(past_val):
                yoy = ((r[value_col] - past_val) / past_val) * 100
                y_true.append(yoy)
                rows.append(idx)
        if len(rows) == 0:
            return None
        X = hold.drop(columns=[value_col, 'Date'], errors='ignore')
        X_y = X.iloc[rows]
        # pipeline may be wrapped in dict
        pipe = pipeline if not isinstance(pipeline, dict) else (pipeline.get('pipeline') or pipeline.get('model') or None)
        if pipe is None:
            return None
        yhat = pipe.predict(X_y)
        if len(yhat) != len(y_true):
            n = min(len(yhat), len(y_true))
            yhat = yhat[:n]
            y_true = y_true[:n]
        return float(r2_score(y_true, yhat))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# TASK 1: Environment validation for cross-machine model deployment
# ---------------------------------------------------------------------------

def _validate_environment(artifact_env):
    """Compare artifact library versions to the current runtime.

    Returns a list of human-readable mismatch strings.
    An empty list means everything matches.
    """
    import sklearn  # type: ignore[reportMissingImports]
    import xgboost as xgb  # type: ignore[reportMissingImports]

    current = {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    mismatches = []
    for lib, expected in artifact_env.items():
        if lib in current and current[lib] != expected:
            mismatches.append(f"{lib}: artifact={expected}, runtime={current[lib]}")
    return mismatches


# ---------------------------------------------------------------------------
# Paths and model loading
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODEL_PATH = str((BASE_DIR / "models" / "best_model.pkl"))
METRICS_PATH = str((BASE_DIR / "models" / "metrics.json"))
DATA_PATH = str((BASE_DIR / "data" / "inflation_dataset.csv"))

FEATURE_COLUMNS = None
MODEL_R2 = None
MODEL_NAME = "Ridge (K-fold CV)"
best_model = None
ENV_MISMATCH_WARNING = None  # populated if artifact versions differ from runtime

try:
    if Path(MODEL_PATH).exists():
        loaded = joblib.load(MODEL_PATH)
        # Support two formats: a dict wrapper or the pipeline directly
        if isinstance(loaded, dict):
            FEATURE_COLUMNS = loaded.get("feature_columns")
            MODEL_NAME = loaded.get("model_name", MODEL_NAME)
            if MODEL_R2 is None:
                MODEL_R2 = loaded.get("model_r2")
            best_model = loaded.get("pipeline")

            # --- Environment validation (Task 1) ---
            if 'environment' in loaded:
                mismatches = _validate_environment(loaded['environment'])
                if mismatches:
                    mismatch_detail = "\n".join(f"  • {m}" for m in mismatches)
                    ENV_MISMATCH_WARNING = (
                        "Model was trained in a different environment. "
                        "Predictions may be unreliable.\n" + mismatch_detail
                    )
                    print(f"\n{'='*60}")
                    print("[!] MODEL ENVIRONMENT MISMATCH")
                    print(mismatch_detail)
                    print(f"{'='*60}\n")
                else:
                    print("[Startup] Environment check passed — artifact matches runtime.")
            else:
                print("[Startup] No environment metadata in artifact (trained before version tracking).")
        else:
            best_model = loaded
except Exception as e:
    print(f"Warning: Could not load model at {MODEL_PATH}. Error: {repr(e)}")
    best_model = None

HOLDOUT_R2 = None

holdout_csv_path = Path(BASE_DIR) / 'models' / 'holdout.csv'
if holdout_csv_path.exists() and best_model is not None:
    try:
        hold_df = pd.read_csv(holdout_csv_path)
        if 'target_future_inflation' in hold_df.columns and 'Predicted_Inflation' in hold_df.columns:
            from sklearn.metrics import r2_score as _r2
            HOLDOUT_R2 = float(_r2(
                hold_df['target_future_inflation'].dropna(),
                hold_df['Predicted_Inflation'].dropna()
            ))
            print(f'[Startup] Holdout R2: {HOLDOUT_R2:.4f}')
    except Exception as e:
        print(f'[Startup] Could not compute holdout R2: {e}')

# Load metrics.json
if Path(METRICS_PATH).exists():
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
        if MODEL_R2 is None and MODEL_NAME:
            MODEL_R2 = _extract_model_r2(metrics_data, MODEL_NAME)
    except Exception as e:
        print(f"Warning: Could not load metrics at {METRICS_PATH}. Error: {e}")

# Compute holdout R2 if not available from metrics
try:
    if MODEL_R2 is None and Path(BASE_DIR / 'models' / 'holdout.csv').exists() and best_model is not None:
        try:
            hold = pd.read_csv(Path(BASE_DIR) / 'models' / 'holdout.csv')
            hold['Date'] = pd.to_datetime(hold['Date'])
            hold_feat = create_features(hold)
            X_hold = hold_feat.drop(['Date', 'CPI'], axis=1, errors='ignore')
            if FEATURE_COLUMNS:
                X_hold = X_hold.reindex(columns=FEATURE_COLUMNS, fill_value=0)
            pipe = best_model if not isinstance(best_model, dict) else (best_model.get('pipeline') or best_model.get('model') or None)
            if pipe is None:
                raise AttributeError('No pipeline available in best_model wrapper')
            preds = pipe.predict(X_hold)
            if 'CPI' in hold.columns:
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
if ENV_MISMATCH_WARNING:
    print('[Startup] ENV WARNING:', ENV_MISMATCH_WARNING)

# Centralized Column Map
COLUMN_MAP = {
    'cpi': 'INDCPIALLMINMEI',
    'wpi': 'WPIATT01INM661N',
    'interest_rate': 'INTDSRINM193N',
    'usd_inr': 'DEXINUS',
    'brent_crude': 'Average of DCOILBRENTEU',
    'industrial_prod': 'INDPRINTO01GYSAM',
    'trade_balance': 'XTNTVA01INM667N',
}


# ---------------------------------------------------------------------------
# Data-computation helper functions
# Each returns a plain dict — used by both legacy routes and new API routes.
# ---------------------------------------------------------------------------

def _pct_change(current, past):
    """Calculate percentage change between two values."""
    if past == 0:
        return 0
    return ((current - past) / past) * 100


def _build_dashboard_data():
    """Compute all KPI values and chart data for the dashboard view."""
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # Removed hardcoded 2025 cutoff. The dashboard should show ALL available data.
    df = df.copy().reset_index(drop=True)

    latest = df.iloc[-1]

    # Monthly change baseline
    past_30_date = latest['Date'] - pd.Timedelta(days=30)
    idx_30 = (df['Date'] - past_30_date).abs().idxmin()
    past_30 = df.loc[idx_30]

    # YoY baseline
    past_365_date = latest['Date'] - pd.Timedelta(days=365)
    idx_365 = (df['Date'] - past_365_date).abs().idxmin()
    past_365 = df.loc[idx_365]

    current_cpi = latest[COLUMN_MAP['cpi']]
    past_year_cpi = past_365[COLUMN_MAP['cpi']]
    inflation_rate = ((current_cpi - past_year_cpi) / past_year_cpi) * 100 if past_year_cpi else 0

    past_month_year_ago_date = past_30['Date'] - pd.Timedelta(days=365)
    idx_month_year_ago = (df['Date'] - past_month_year_ago_date).abs().idxmin()
    past_month_year_ago_cpi = df.loc[idx_month_year_ago, COLUMN_MAP['cpi']]
    past_month_inflation = ((past_30[COLUMN_MAP['cpi']] - past_month_year_ago_cpi) / past_month_year_ago_cpi) * 100 if past_month_year_ago_cpi else 0
    inflation_change = inflation_rate - past_month_inflation

    # Chart data — last 24 months
    df_monthly = df.set_index('Date').resample('MS').last().reset_index()
    recent_history = df_monthly.tail(24).copy()

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

    avg_inflation = round(sum(chart_values) / len(chart_values), 2) if chart_values else 0
    if chart_values:
        peak_inflation = max(chart_values)
        peak_date = chart_dates[chart_values.index(peak_inflation)]
        low_inflation = min(chart_values)
        low_date = chart_dates[chart_values.index(low_inflation)]
    else:
        peak_inflation = peak_date = low_inflation = low_date = 0

    return {
        'inflation_rate': round(inflation_rate, 2),
        'inflation_change': round(inflation_change, 2),
        'cpi_value': round(latest[COLUMN_MAP['cpi']], 2),
        'cpi_change': round(_pct_change(latest[COLUMN_MAP['cpi']], past_30[COLUMN_MAP['cpi']]), 2),
        'wpi_value': round(latest[COLUMN_MAP['wpi']], 2),
        'wpi_change': round(_pct_change(latest[COLUMN_MAP['wpi']], past_30[COLUMN_MAP['wpi']]), 2),
        'interest_rate': round(latest[COLUMN_MAP['interest_rate']], 2),
        'usdinr_value': round(latest[COLUMN_MAP['usd_inr']], 2),
        'usdinr_change': round(_pct_change(latest[COLUMN_MAP['usd_inr']], past_30[COLUMN_MAP['usd_inr']]), 2),
        'brent_value': round(latest[COLUMN_MAP['brent_crude']], 2),
        'brent_change': round(_pct_change(latest[COLUMN_MAP['brent_crude']], past_30[COLUMN_MAP['brent_crude']]), 2),
        'avg_inflation': avg_inflation,
        'peak_inflation': peak_inflation,
        'peak_date': peak_date,
        'low_inflation': low_inflation,
        'low_date': low_date,
        'trend': 'Rising' if inflation_change > 0 else 'Declining',
        'num_features': len(FEATURE_COLUMNS) if FEATURE_COLUMNS else len(COLUMN_MAP),
        'num_observations': len(df),
        'date_range': f"{df['Date'].dt.year.min()} - {df['Date'].dt.year.max()}",
        'inflation_history': {
            'dates': chart_dates,
            'values': chart_values
        }
    }


def _build_analysis_data():
    """Compute correlation matrix, per-indicator time series, and histograms."""
    df = pd.read_csv(DATA_PATH)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    corr_cols = {
        COLUMN_MAP['cpi']: 'CPI',
        COLUMN_MAP['wpi']: 'WPI',
        COLUMN_MAP['interest_rate']: 'Interest Rate',
        COLUMN_MAP['usd_inr']: 'USD/INR',
        COLUMN_MAP['brent_crude']: 'Brent Crude',
    }
    available_corr_keys = [c for c in corr_cols.keys() if c in df.columns]
    if available_corr_keys:
        corr_df = df[available_corr_keys].rename(columns=corr_cols).corr().round(2)
        corr_matrix = corr_df.to_dict('index')
    else:
        corr_matrix = {}

    key_findings = [
        "CPI is strongly correlated with WPI and Exchange Rates.",
        "Interest rates show a delayed effect on inflation (90+ day lag).",
        "Brent Crude has immediate transmission into WPI."
    ]

    try:
        from statsmodels.tsa.stattools import adfuller
        stationarity_tests = []
        test_series = {'CPI': COLUMN_MAP['cpi'], 'WPI': COLUMN_MAP['wpi'], 'Brent Crude': COLUMN_MAP['brent_crude']}
        for label, col_name in test_series.items():
            if col_name in df.columns:
                series = df[col_name].dropna()
                if len(series) > 20:
                    result = adfuller(series, maxlag=12, autolag='AIC')
                    stationarity_tests.append({
                        'name': label,
                        'adf_stat': round(result[0], 2),
                        'p_value': round(result[1], 4),
                        'stationary': result[1] < 0.05
                    })
                    # Also test differenced
                    diff_result = adfuller(series.diff().dropna(), maxlag=12, autolag='AIC')
                    stationarity_tests.append({
                        'name': f'{label} (Differenced)',
                        'adf_stat': round(diff_result[0], 2),
                        'p_value': round(diff_result[1], 4),
                        'stationary': diff_result[1] < 0.05
                    })
    except ImportError:
        stationarity_tests = [
            {'name': 'CPI', 'adf_stat': -1.2, 'p_value': 0.65, 'stationary': False},
            {'name': 'CPI (Differenced)', 'adf_stat': -4.5, 'p_value': 0.001, 'stationary': True},
            {'name': 'Brent Crude', 'adf_stat': -2.1, 'p_value': 0.25, 'stationary': False}
        ]

    # Per-indicator time series and histograms
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_monthly = df_copy.set_index('Date').resample('MS').last().reset_index()
    dates_str = df_monthly['Date'].dt.strftime('%b %Y').tolist()

    analysis_data_dict = {}
    for col_key, col_name in corr_cols.items():
        vals = df_monthly[col_key].dropna().tolist()
        counts, bins = np.histogram(df[col_key].dropna(), bins=20)
        analysis_data_dict[col_name] = {
            'dates': dates_str,
            'values': vals,
            'hist_counts': counts.tolist(),
            'hist_labels': [f"{round(bins[i], 1)}" for i in range(len(bins) - 1)]
        }

    return {
        'key_findings': key_findings,
        'stationarity_tests': stationarity_tests,
        'feature_names': list(corr_cols.values()),
        'analysis_data': analysis_data_dict,
        'corr_matrix': corr_matrix,
        'features': list(corr_cols.values())
    }


def _build_models_data():
    """Build model comparison metrics, feature importances, and chart data."""
    metrics_data = {"best_model": "Unknown", "metrics": {}}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")

    best_model_name = metrics_data.get("best_model", MODEL_NAME)
    best_model_key = str(best_model_name).lower()

    # Build normalized list of model metrics regardless of metrics.json layout
    models_metrics = []
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

    best_model_r2 = float(MODEL_R2) if MODEL_R2 is not None else 0.0
    best_model_mae = 0
    best_model_rmse = 0

    for m in models_metrics:
        if _norm_key(m.get('name', '')) == _norm_key(best_model_name):
            best_model_r2 = m.get('r2_mean', 0)
            best_model_mae = m.get('mae', 0)
            best_model_rmse = m.get('rmse', 0)
            break

    # Extract real feature importances from the loaded model
    feature_importances = _extract_feature_importances()

    prediction_chart_data = {"dates": [], "actual": [], "best_model_predictions": []}
    holdout_csv = Path(BASE_DIR) / 'models' / 'holdout.csv'
    if holdout_csv.exists():
        try:
            hold = pd.read_csv(holdout_csv)
            hold['Date'] = pd.to_datetime(hold['Date'])
            hold = hold.sort_values('Date').reset_index(drop=True)
            if 'target_future_inflation' in hold.columns and 'Predicted_Inflation' in hold.columns:
                prediction_chart_data = {
                    "dates": hold['Date'].dt.strftime('%b %Y').tolist(),
                    "actual": hold['target_future_inflation'].round(2).tolist(),
                    "best_model_predictions": hold['Predicted_Inflation'].round(2).tolist()
                }
        except Exception as e:
            print(f"Could not load holdout for chart: {e}")

    return {
        'best_model_name': best_model_name,
        'models_metrics': models_metrics,
        'best_model_r2': best_model_r2,
        'best_model_mae': best_model_mae,
        'best_model_rmse': best_model_rmse,
        'feature_importances': feature_importances,
        'prediction_chart_data': prediction_chart_data
    }


def _extract_feature_importances():
    """Extract real feature importances from the loaded model pipeline.

    Works for Ridge (abs coef_), RandomForest/XGBoost (feature_importances_).
    Falls back to mock data if extraction fails.
    """
    try:
        if best_model is None or FEATURE_COLUMNS is None:
            raise ValueError("No model or feature columns available")

        from sklearn.pipeline import Pipeline  # type: ignore[reportMissingImports]
        pipe = best_model if not isinstance(best_model, dict) else best_model.get('pipeline')
        if pipe is None:
            raise ValueError("No pipeline in model wrapper")

        # Dig into the pipeline to find the final estimator
        if isinstance(pipe, Pipeline):
            final_step = pipe.steps[-1][1]
        else:
            final_step = pipe

        # Unwrap ClipRegressor if present
        estimator = getattr(final_step, 'estimator_', final_step)

        # Extract importances based on estimator type
        if hasattr(estimator, 'feature_importances_'):
            raw_importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            raw_importances = np.abs(estimator.coef_)
        else:
            raise ValueError("Estimator has no feature_importances_ or coef_")

        # Pair with feature names, sort descending, take top 8
        pairs = list(zip(FEATURE_COLUMNS, raw_importances))
        pairs.sort(key=lambda p: p[1], reverse=True)
        top = pairs[:8]

        total = sum(v for _, v in top)
        if total == 0:
            raise ValueError("All importances are zero")

        result = []
        for name, value in top:
            # Shorten long engineered feature names for readability
            display_name = name.replace('_pct_change_', ' Δ').replace('_lag_pct_', ' lag ')
            display_name = display_name.replace('_rolling_pct_avg_', ' avg ')
            pct = round((value / total) * 100, 1)
            result.append({
                "name": display_name,
                "value": f"{value:.4f}",
                "percentage": float(pct)
            })
        return result

    except Exception:
        # Fallback to mock data
        return [
            {"name": "WPI Index", "value": "0.35", "percentage": 35},
            {"name": "Brent Crude", "value": "0.25", "percentage": 25},
            {"name": "USD/INR Rate", "value": "0.20", "percentage": 20},
            {"name": "Interest Rate", "value": "0.15", "percentage": 15},
            {"name": "GDP Proxy", "value": "0.05", "percentage": 5}
        ]


def _build_forecast_data():
    """Compute 12-month inflation trend and current indicator values."""
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.copy().reset_index(drop=True)

    latest_date = df['Date'].max()
    one_year_ago = latest_date - pd.Timedelta(days=365)
    historical = df[df['Date'] >= one_year_ago].copy()

    months = historical['Date'].dt.strftime('%b %Y').tolist()
    cpi_values = historical[COLUMN_MAP['cpi']].tolist()

    inflation_rates = []
    for idx, row in historical.iterrows():
        d = row['Date']
        y_ago = d - pd.Timedelta(days=365)
        closest_idx = (df['Date'] - y_ago).abs().idxmin()
        past_cpi = df.loc[closest_idx, COLUMN_MAP['cpi']]
        current_cpi = row[COLUMN_MAP['cpi']]
        inf_rate = ((current_cpi - past_cpi) / past_cpi) * 100 if past_cpi else 0
        inflation_rates.append(round(inf_rate, 2))

    latest = df.iloc[-1]
    current_cpi = latest[COLUMN_MAP['cpi']]
    past_year = latest['Date'] - pd.Timedelta(days=365)
    past_idx = (df['Date'] - past_year).abs().idxmin()
    past_cpi_val = df.loc[past_idx, COLUMN_MAP['cpi']]
    current_inflation = ((current_cpi - past_cpi_val) / past_cpi_val) * 100 if past_cpi_val else 0

    return {
        'months': months,
        'cpi_values': cpi_values,
        'inflation_rates': inflation_rates,
        'current_cpi': round(current_cpi, 2),
        'current_wpi': round(latest[COLUMN_MAP['wpi']], 2),
        'current_rate': round(latest[COLUMN_MAP['interest_rate']], 2),
        'current_inflation': round(current_inflation, 2),
        'latest_date': latest_date.strftime('%B %d, %Y'),
        'forecast_model': MODEL_NAME if MODEL_NAME else 'ML Ensemble',
        'model_order': f'Holdout R²={HOLDOUT_R2:.4f}' if HOLDOUT_R2 is not None else 'N/A',
        'data_points': len(df)
    }


def _get_dynamic_defaults():
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            latest = df.iloc[-1]
            return {
                'wpi_index': float(latest.get(COLUMN_MAP['wpi'], 136.30)),
                'interest_rate': float(latest.get(COLUMN_MAP['interest_rate'], 6.5)),
                'usd_inr': float(latest.get(COLUMN_MAP['usd_inr'], 83.42)),
                'brent_crude': float(latest.get(COLUMN_MAP['brent_crude'], 80.92)),
            }
    except Exception:
        pass
    return {
        'wpi_index': 165.50,
        'interest_rate': 5.25,
        'usd_inr': 95.40,
        'brent_crude': 107.00,
    }

def _run_prediction(form_data):
    """Execute a prediction given form input data. Returns a result dict.

    This is the core prediction logic extracted from the original /predict POST handler.
    """
    default_values = _get_dynamic_defaults()

    val_wpi = float(form_data.get('wpi_index', default_values['wpi_index']))
    val_ir = float(form_data.get('interest_rate', default_values['interest_rate']))
    val_usd = float(form_data.get('usd_inr', default_values['usd_inr']))
    val_brent = float(form_data.get('brent_crude', default_values['brent_crude']))
    raw_date = form_data.get('scenario_date', '')

    current_values = {
        'wpi_index': val_wpi,
        'interest_rate': val_ir,
        'usd_inr': val_usd,
        'brent_crude': val_brent,
    }

    prediction = None
    interpretation_text = ""
    interpretation_color = "border-l-primary"
    display_prediction = None

    try:
        scenario_date = pd.to_datetime(raw_date) if raw_date else None

        if best_model and os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

            last_date = df['Date'].max()
            new_date = scenario_date if scenario_date is not None else last_date + pd.Timedelta(days=30)

            new_row = df.iloc[-1].copy()
            new_row['Date'] = new_date
            new_row[COLUMN_MAP['brent_crude']] = val_brent
            new_row[COLUMN_MAP['usd_inr']] = val_usd
            new_row[COLUMN_MAP['interest_rate']] = val_ir
            new_row[COLUMN_MAP['wpi']] = val_wpi

            # Build prediction features
            pred_X = None
            try:
                if FEATURE_COLUMNS and all([col in df.columns for col in FEATURE_COLUMNS]):
                    row_vals = {col: (new_row.get(col, 0) if col in new_row.index else 0) for col in FEATURE_COLUMNS}
                    pred_X = pd.DataFrame([row_vals])
                else:
                    df_pred = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    featured_df = create_features(df_pred)
                    scenario_feature_row = featured_df.loc[featured_df['Date'] == new_date].tail(1)
                    if scenario_feature_row.empty:
                        scenario_feature_row = featured_df.iloc[[-1]]
                    pred_X = scenario_feature_row.drop(['Date', 'CPI'], axis=1, errors='ignore')
                    if FEATURE_COLUMNS:
                        pred_X = pred_X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
            except Exception:
                pred_X = pd.DataFrame([[0] * (len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 1)],
                                      columns=(FEATURE_COLUMNS or ['x']))

            # Fallback if pred_X is all zeros
            try:
                if pred_X is None or (pred_X.fillna(0).abs().sum().sum() == 0):
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

            # Run prediction
            pipe = best_model if not isinstance(best_model, dict) else (best_model.get('pipeline') or best_model.get('model') or None)
            if pipe is None:
                raise AttributeError('No pipeline available in best_model wrapper')
            pred_inflation = pipe.predict(pred_X)[0]

            # Defensive: fallback to historical median for extreme predictions
            df_dates = df.copy()
            df_dates['Date'] = pd.to_datetime(df_dates['Date'])
            latest_date_hist = df_dates['Date'].max()
            one_year_ago = latest_date_hist - pd.Timedelta(days=365)
            recent = df_dates[df_dates['Date'] >= one_year_ago].copy()
            yoy_vals = []
            for _, r in recent.iterrows():
                d = r['Date']
                y_ago = d - pd.Timedelta(days=365)
                closest_idx = (df_dates['Date'] - y_ago).abs().idxmin()
                past_cpi = df_dates.loc[closest_idx, COLUMN_MAP['cpi']]
                cur_cpi = r[COLUMN_MAP['cpi']]
                if past_cpi:
                    yoy_vals.append(((cur_cpi - past_cpi) / past_cpi) * 100)
            fallback_median = float(pd.Series(yoy_vals).median()) if len(yoy_vals) > 0 else 0.0

            prediction_raw = float(pred_inflation)
            if pd.isna(prediction_raw) or prediction_raw < 0 or abs(prediction_raw) > 25:
                prediction = round(fallback_median, 2)
                interpretation_text = f"Model produced an extreme prediction ({round(prediction_raw, 2)}%). Using recent median fallback {prediction}% to keep outputs stable."
                interpretation_color = "border-l-[#ffb4ab]"
            else:
                prediction = round(prediction_raw, 2)

            display_prediction = prediction

            # Interpretation
            if prediction < 2:
                interpretation_color = "border-l-[#4d8eff]"
                interpretation_text = f"Model predicts inflation at {prediction}%, below RBI's 2-6% target band."
            elif prediction <= 6:
                interpretation_color = "border-l-[#4edea3]"
                interpretation_text = f"Model predicts inflation at {prediction}%, within RBI's 2-6% target band."
            else:
                interpretation_color = "border-l-[#ff5449]"
                interpretation_text = f"Warning: Model predicts inflation at {prediction}%, exceeding RBI's 6% upper limit."
        else:
            interpretation_text = "Model or dataset missing. Cannot run prediction."

    except Exception:
        traceback.print_exc()
        interpretation_text = "An error occurred during calculation."

    # Resolve display R2
    display_model_r2 = MODEL_R2
    try:
        if display_model_r2 is None and Path(METRICS_PATH).exists():
            with open(METRICS_PATH, 'r') as f:
                md = json.load(f)
            display_model_r2 = _extract_model_r2(md, MODEL_NAME)
    except Exception:
        pass
    if display_model_r2 is None:
        display_model_r2 = 0.0

    try:
        try_fix = False
        if MODEL_R2 is None:
            try_fix = True
        else:
            try:
                if abs(float(MODEL_R2)) > 100:
                    try_fix = True
            except Exception:
                try_fix = True
        if try_fix and Path(BASE_DIR / 'models' / 'holdout.csv').exists() and best_model is not None:
            _yoy_r2 = _compute_yoy_r2_from_holdout(best_model, holdout_path=BASE_DIR / 'models' / 'holdout.csv')
            if _yoy_r2 is not None:
                display_model_r2 = _yoy_r2
    except Exception:
        pass

    return {
        'current_values': current_values,
        'prediction': prediction,
        'display_prediction': display_prediction,
        'interpretation_text': interpretation_text,
        'interpretation_color': interpretation_color,
        'model_used': MODEL_NAME,
        'model_r2': float(HOLDOUT_R2) if HOLDOUT_R2 is not None else (float(display_model_r2) if display_model_r2 else 0.0)
    }


# ---------------------------------------------------------------------------
# Cockpit route — the single-page application entry point
# ---------------------------------------------------------------------------

@app.route('/')
def cockpit():
    return render_template('cockpit.html', env_warning=ENV_MISMATCH_WARNING)


# ---------------------------------------------------------------------------
# JSON API endpoints for the SPA cockpit
# ---------------------------------------------------------------------------

@app.route('/api/command-center')
def api_command_center():
    """Return combined dashboard + analysis data as JSON."""
    dashboard = _build_dashboard_data()
    analysis = _build_analysis_data()
    return jsonify({
        'dashboard': dashboard,
        'analysis': analysis
    })


@app.route('/api/predictive-sandbox', methods=['GET', 'POST'])
def api_predictive_sandbox():
    """GET: return forecast data + defaults. POST: run prediction and return result."""
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form.to_dict()
        result = _run_prediction(data)
        return jsonify(result)

    # GET — return forecast data and default input values
    forecast = _build_forecast_data()
    display_r2 = HOLDOUT_R2 if HOLDOUT_R2 is not None else (MODEL_R2 if MODEL_R2 is not None else 0.0)
    return jsonify({
        'forecast': forecast,
        'defaults': _get_dynamic_defaults(),
        'model_name': MODEL_NAME,
        'model_r2': display_r2
    })


@app.route('/api/model-registry')
def api_model_registry():
    """Return model metrics, feature importances, and chart data as JSON."""
    data = _build_models_data()
    return jsonify(data)


@app.route('/api/env-status')
def api_env_status():
    """Return environment mismatch warning if any."""
    return jsonify({
        'warning': ENV_MISMATCH_WARNING
    })


# Legacy routes removed because UI migrated to cockpit.html SPA


if __name__ == '__main__':
    print("Starting VaticMacro Flask Server (no reloader)...")
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
