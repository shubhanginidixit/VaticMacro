# System Architecture

## Overview

VaticMacro is a Flask-based web application that predicts Indian YoY CPI inflation using macroeconomic indicators. The system follows a clear separation between data processing, model training, and web presentation.

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐
│  Raw CSVs   │───>│  merge_data  │───>│  Merged CSV │───>│  main.py  │
│  (data/)    │    │     .py      │    │  (dataset)  │    │ (trainer) │
└─────────────┘    └──────────────┘    └──────┬──────┘    └─────┬─────┘
                                              │                 │
                                              v                 v
                                      ┌───────────────┐  ┌───────────┐
                                      │    app.py      │  │  models/  │
                                      │ (Flask routes) │<─│  .pkl     │
                                      └───────┬───────┘  └───────────┘
                                              │
                                              v
                                      ┌───────────────┐
                                      │  Browser UI   │
                                      │ (templates/)  │
                                      └───────────────┘
```

---

## Data Flow

### 1. Data Ingestion (`merge_data.py`)
- Reads 9 raw CSV files from `data/` with different formats (FRED, pivot tables, year-columns)
- Standardizes date columns and resamples to daily frequency
- Forward-fills missing values (CPI is monthly, exchange rates are daily)
- Outputs `data/inflation_dataset.csv` (7,023 rows × 10 columns)

### 2. Preprocessing (`src/preprocessing.py`)
- Loads the merged CSV
- Handles multiple date column naming conventions
- Applies forward-fill then backward-fill for missing values
- Filters to specified date range

### 3. Feature Engineering (`src/feature_engineering.py`)
- Creates **percentage-change features** at 30, 90, 180-day windows for each indicator
- Creates **lag features** on percentage changes (shifted 30 and 90 days)
- Creates **rolling average features** (30 and 90-day windows)
- Creates **ratio features** (WPI/CPI ratio, Oil/INR ratio)
- Clips extreme values to ±25% to reduce outlier influence
- Drops raw indicator columns (keeps only engineered features)
- Drops volatile oil-related features

**Output:** 44 engineered features + Date + CPI

### 4. Model Training (`src/train_model.py`)

The critical insight: CPI is released monthly, but the dataset has daily rows. Training on daily data creates 96% redundant samples. The training pipeline handles this:

1. **Monthly resampling** — Takes last observation per month (309 → 216 training rows after filtering)
2. **YoY target computation** — `(CPI_t - CPI_{t-12}) / CPI_{t-12} × 100`
3. **Autoregressive features** — Adds `inflation_lag1`, `lag3`, `lag6`, `rolling_mean_3`, `rolling_mean_6`
4. **Feature selection** — Drops features with |correlation| < 0.1 to target
5. **Cross-validation** — 5-fold TimeSeriesSplit
6. **Model comparison** — RidgeCV vs RandomForest vs XGBoost
7. **Best model saved** — Pipeline + feature columns + metadata

### 5. Web Application (`app.py`)

Seven Flask routes serving Jinja2 templates:

| Route | Method | Template | Key Data |
|-------|--------|----------|----------|
| `/` | GET | `home.html` | Landing page |
| `/dashboard` | GET | `dashboard.html` | Monthly-resampled inflation, indicator cards |
| `/analysis` | GET | `analysis_page.html` | Correlation matrix, time-series, histograms |
| `/models` | GET | `models_page.html` | R² table, feature importances, holdout chart |
| `/predict` | GET/POST | `predict_page.html` | Scenario builder with model prediction |
| `/forecast` | GET | `forecast_page.html` | 12-month inflation bar chart |
| `/about` | GET | `about.html` | Project information |

---

## Key Design Decisions

### Why monthly resampling?
CPI is released once per month. Forward-filling creates ~30 identical rows per month. Training on these gives the model false confidence (low training error) but poor generalization (negative R² on new time periods).

### Why autoregressive features?
Inflation is highly persistent (autocorrelation = 0.997 at lag-1). Last month's inflation is the single strongest predictor of this month's inflation. Not including it wastes the dominant signal.

### Why RidgeCV over XGBoost?
With only 216 monthly training observations, regularized linear regression generalizes better. Tree-based models overfit on small datasets. This is empirically validated: RidgeCV R² = 0.71 vs XGBoost R² = 0.07.

### Why percentage-change features?
Raw indicator values (e.g., GDP in trillions) have wildly different scales across decades. A model trained on 2005 GDP values wouldn't generalize to 2020 values. Percentage changes are scale-independent.

### Why forward-fill aware dashboard?
The dashboard and forecast pages detect months where CPI didn't change (forward-filled) and exclude them from inflation calculations. This prevents misleading "0% inflation" displays caused by stale data.

---

## Module Dependencies

```
app.py
├── src/config.py          (COLUMN_MAP, file paths)
├── src/feature_engineering.py  (create_features)
├── models/best_model.pkl  (trained pipeline)
└── models/metrics.json    (R², MAE, RMSE per model)

main.py
├── src/preprocessing.py   (load_and_clean_data)
├── src/feature_engineering.py  (create_features)
└── src/train_model.py     (train)

merge_data.py
└── src/preprocessing.py   (load_and_clean_data)
```

---

## Configuration (`src/config.py`)

All shared constants are centralized:

```python
COLUMN_MAP = {
    'cpi': 'INDCPIALLMINMEI',
    'wpi': 'WPIATT01INM661N',
    'interest_rate': 'INTDSRINM193N',
    'usd_inr': 'DEXINUS',
    'brent_crude': 'Average of DCOILBRENTEU',
    'gdp_proxy': 'MKTGDPINA646NWDB'
}

MODEL_PATH = 'models/best_model.pkl'
METRICS_PATH = 'models/metrics.json'
DATA_PATH = 'data/inflation_dataset.csv'
```
