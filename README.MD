# VaticMacro

## Indian Macroeconomic Inflation Prediction and Trend Forecasting

VaticMacro is a machine learning framework for forecasting Indian inflation trends through analysis of domestic macroeconomic indicators, global market dynamics, and policy metrics. It predicts Year-over-Year (YoY) CPI inflation using engineered features from economic time-series data.

The platform provides policy-makers, researchers, and financial analysts with tools to evaluate inflation patterns, assess macroeconomic risks, and make data-driven decisions based on empirical economic projections.

---

## Problem Statement

Inflation prediction is challenging because it depends on interconnected systemic factors:

- Consumer Price Index (CPI) — monthly release frequency
- Wholesale Price Index (WPI) — monthly
- RBI Benchmark Interest Rates — monthly
- GDP Growth Trajectories — annual (daily proxy used)
- Unemployment Rate — annual
- USD/INR Exchange Rate — daily
- Brent Crude Oil Prices — daily

**Key challenge:** These indicators are released at different frequencies (daily, monthly, annual). Naively merging them into daily rows creates 96% redundant data (forward-filled CPI across business days), which destroys model performance. VaticMacro solves this through monthly resampling before training.

---

## Methodology

VaticMacro's pipeline:

1. **Data Aggregation** — Merges 8 economic indicators from FRED/RBI sources into a unified daily dataset (`inflation_dataset.csv`)
2. **Monthly Resampling** — Resamples to monthly frequency, eliminating forward-fill redundancy
3. **Feature Engineering** — Creates percentage-change features (30/90/180-day), lag features, rolling averages, and cross-indicator ratios
4. **Autoregressive Features** — Adds inflation lag-1, lag-3, lag-6, and rolling means (strongest predictors due to inflation persistence)
5. **Feature Selection** — Drops features with |correlation| < 0.1 to target, reducing noise
6. **Model Training** — Trains RidgeCV, RandomForest, and XGBoost with TimeSeriesSplit cross-validation
7. **Model Selection** — Automatically selects best model by mean R² across folds

### Why RidgeCV Wins

With only ~216 monthly observations, tree-based models (RF, XGB) overfit. RidgeCV with autoregressive features achieves **R² = 0.71, MAE = 0.67%** — meaning predictions are typically within 0.67 percentage points of actual inflation.

---

## Model Performance

| Model | Mean R² | Mean MAE | Mean RMSE |
|-------|---------|----------|-----------|
| **RidgeCV** | **0.711** | **0.67%** | **0.85%** |
| XGBoost | 0.074 | 1.28% | 1.58% |
| RandomForest | 0.051 | 1.32% | 1.60% |

Evaluated using 5-fold TimeSeriesSplit cross-validation on monthly data from 2005–2022.

---

## Core Capabilities

### Economic Data Analysis
- Historical inflation trend aggregation
- Multivariate correlation analysis with data-driven key findings
- Time-series visualization of all major economic indicators
- Distribution analysis (histograms) for each indicator

### Machine Learning Forecasting
- YoY inflation prediction using percentage-change features
- Scenario-based prediction (adjust WPI, interest rates, exchange rates, oil, GDP)
- Autoregressive modeling (past inflation predicts future inflation)

**Supported Models:**
- RidgeCV (Regularized Linear Regression with built-in cross-validation)
- Random Forest Regressor
- XGBoost Regressor

### Interactive Dashboard
- Real-time indicator cards (CPI, WPI, Interest Rate, USD/INR, Brent Crude)
- 24-month inflation history chart
- Peak/trough analysis with trend direction
- Model performance comparison with honest R² reporting
- Predicted vs Actual holdout chart

---

## Technical Architecture

### Backend
- Python 3.12+
- Flask 3.0 (Web application)

### Data Processing
- Pandas (data manipulation and resampling)
- NumPy (numerical computation)

### Machine Learning
- Scikit-learn (RidgeCV, RandomForest, preprocessing, cross-validation)
- XGBoost (gradient boosting)

### Frontend
- TailwindCSS (utility-first CSS via CDN)
- Chart.js (interactive charts)
- Google Material Symbols (icons)
- Inter font (typography)

---

## Dataset

The merged dataset (`data/inflation_dataset.csv`) contains 7,023 daily rows across 10 columns:

| Indicator | Column Name | Frequency | Source |
|-----------|-------------|-----------|--------|
| Consumer Price Index | `INDCPIALLMINMEI` | Monthly | FRED |
| Wholesale Price Index | `WPIATT01INM661N` | Monthly | FRED |
| Interest Rate | `INTDSRINM193N` | Monthly | FRED |
| USD/INR Exchange Rate | `DEXINUS` | Daily | FRED |
| Brent Crude Oil | `Average of DCOILBRENTEU` | Daily | FRED |
| India Basket Crude | `datafilenew(india basket crude oil)` | Daily | PPAC |
| GDP Proxy | `MKTGDPINA646NWDB` | Annual | FRED |
| Unemployment Rate | `Unemployment Rate Annually` | Annual | World Bank |
| Stock Market Proxy | `INDLORSGPNOSTSAM` | Daily | FRED |

**Target variable:** YoY CPI inflation (%), computed as `((CPI_t - CPI_{t-12}) / CPI_{t-12}) × 100` on monthly resampled data.

**Training window:** January 2005 – December 2022 (216 monthly observations)

---

## Project Structure

```text
VaticMacro/
│
├── data/
│   ├── [Raw CSV datasets from FRED/RBI/World Bank]
│   └── inflation_dataset.csv       # Merged dataset (7,023 rows × 10 cols)
│
├── models/
│   ├── best_model.pkl              # Trained RidgeCV pipeline + metadata
│   └── metrics.json                # R², MAE, RMSE per model per fold
│
├── src/
│   ├── __init__.py                 # Package marker
│   ├── config.py                   # Column mappings and file paths
│   ├── preprocessing.py            # Data loading, cleaning, date handling
│   ├── feature_engineering.py      # Percentage-change, lag, ratio features
│   └── train_model.py              # Monthly resampling, CV, model selection
│
├── app/
│   └── templates/
│       ├── base.html               # Shared layout (sidebar, head, scripts)
│       ├── dashboard.html          # Inflation overview + indicator cards
│       ├── analysis_page.html      # Correlation matrix + distributions
│       ├── models_page.html        # R² comparison + feature importances
│       ├── predict_page.html       # Scenario builder form
│       ├── forecast_page.html      # 12-month inflation bar chart
│       ├── home.html               # Landing page
│       └── about.html              # Project information
│
├── docs/
│   ├── GETTING_STARTED.md          # Installation and setup guide
│   ├── ARCHITECTURE.md             # System design and data flow
│   └── DATA_DICTIONARY.md          # Column reference and feature docs
│
├── notebooks/
│   └── EDA.ipynb                   # Exploratory data analysis
│
├── app.py                          # Flask application (7 routes)
├── main.py                         # Model training entry point
├── merge_data.py                   # Merges raw CSVs into inflation_dataset.csv
├── run.py                          # Flask launcher with banner
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (optional — pre-trained model included)
```bash
python main.py
```

### 3. Start the dashboard
```bash
python run.py
```
Then open **http://localhost:5000** in your browser.

---

## Application Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/dashboard` | Current inflation rate, indicator cards, 24-month chart |
| Analysis | `/analysis` | Correlation heatmap, time-series plots, distributions |
| Models | `/models` | R² comparison table, feature importances, predicted vs actual |
| Predict | `/predict` | Scenario builder — adjust indicators and get inflation prediction |
| Forecast | `/forecast` | 12-month historical inflation bar chart with indicators |
| About | `/about` | Project description and methodology |

---

## Key Design Decisions

1. **Monthly resampling before training** — CPI is released monthly, but exchange rates are daily. Training on daily data creates 96% redundant rows where CPI doesn't change, destroying cross-validation integrity.

2. **Autoregressive features** — Inflation has autocorrelation > 0.99 at lag-1. Including `inflation_lag1`, `lag3`, `lag6` provides the strongest predictive signal.

3. **RidgeCV over XGBoost** — With only 216 monthly training observations, regularized linear models generalize better than tree ensembles which overfit.

4. **Percentage-change features** — Raw indicator values (e.g., GDP in trillions) have different scales across decades. Percentage changes are scale-independent and generalize across time periods.

5. **Honest metrics display** — The dashboard shows real R² values (including negative ones if they occur) rather than hiding poor performance.

---

## Performance Evaluation

Model performance is validated using:

- **R² Score** (Coefficient of Determination) — proportion of variance explained
- **MAE** (Mean Absolute Error) — average prediction error in percentage points
- **RMSE** (Root Mean Squared Error) — penalizes large prediction errors

All metrics computed via 5-fold TimeSeriesSplit to prevent data leakage.

---

## Future Development

- **Live data integration** via FRED/RBI API pipelines for automatic dataset updates
- **Seasonal features** (month-of-year dummies) for capturing India's seasonal inflation patterns
- **Additional indicators** (M3 money supply, food price index, trade balance)
- **Region-specific inflation** for state-level granularity

---

## Research Contributors

- Dhanush Pillay & Shubhangini Dixit

---

## License

This repository is published for academic, research, and non-commercial educational purposes.