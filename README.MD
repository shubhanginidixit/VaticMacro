# VaticMacro

## Indian Macroeconomic Inflation Prediction and Trend Forecasting

VaticMacro is an advanced machine learning framework explicitly architected for the Indian economy. It is designed to forecast inflation trends through the rigorous analysis of domestic macroeconomic indicators, global market dynamics, and regional policy metrics. The system employs complex data analytics, statistical modeling, and predictive forecasting techniques to yield precise inflation estimates and actionable economic insights tailored to the Indian subcontinent.

This platform empowers domestic policy-makers, researchers, institutional investors, and financial analysts to comprehensively evaluate regional inflation paradigms, assess macroeconomic risks within emerging markets, and execute data-driven strategic decisions based on empirical economic projections.

---

# Problem and Solution

## Problem Statement

Inflation serves as a critical barometer of macroeconomic health, exerting profound effects on purchasing power, capital allocation, municipal policy formulation, and corporate financial strategy.

The accurate prediction of intra- and inter-period inflation is a multifaceted challenge, largely due to its dependency on a vast array of interconnected systemic factors, which include:

- Consumer Price Index (CPI)
- Wholesale Price Index (WPI)
- Benchmark Interest Rates
- Gross Domestic Product (GDP) Growth Trajectories
- Unemployment Metrics
- Foreign Exchange Rate Volatility
- Commodity and Energy Valuations
- Aggregate Money Supply

Conventional forecasting paradigms often fail to discern non-linear or emergent patterns within voluminous economic datasets and frequently lack the agility required to model dynamic macroeconomic shifts.

As global economies grow increasingly interconnected, the necessity for robust, data-driven analytical solutions capable of parsing multivariate economic indicators to synthesize dependable inflation forecasts has become paramount.

---

## The VaticMacro Methodology

VaticMacro centralizes the complex process of economic trend analysis and inflation projection.

The framework systematically:

- Aggregates and normalizes historical macroeconomic data.
- Quantifies correlations and causal relationships among primary economic indicators.
- Implements advanced machine learning topologies to forecast inflation velocity.
- Renders multi-dimensional economic patterns through comprehensive visual dashboards.
- Validates predictive efficacy utilizing industry-standard statistical evaluation metrics.
- Synthesizes complex outputs into actionable strategic intelligence.

By translating dense economic datasets into coherent predictive models, VaticMacro provides stakeholders with the requisite intelligence to anticipate future macroeconomic conditions and formulate rigorous strategic responses.

---

# Core Capabilities

## Economic Data Analysis

- Historical inflation trend aggregation
- Economic indicator benchmarking and comparison
- Multivariate correlation analysis
- Rigorous data preprocessing and normalization
- Advanced feature engineering and selection

---

## Machine Learning Forecasting

- Inflation trajectory prediction utilizing historical datasets
- Predictive modeling via ensemble and linear methodologies
- Time-series statistical forecasting
- Macroeconomic trend extrapolation
- Forward-looking inflation estimation

Supported Methodologies:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- ARIMA (AutoRegressive Integrated Moving Average)

---

## Data Visualization

- Programmatic rendering of inflation trend trajectories
- Forecast divergence and comparison graphing
- Empirical correlation heatmaps
- Macroscopic economic indicator dashboards
- Diagnostic visual insight generation

---

## Performance Evaluation

Algorithmic performance is rigorously validated utilizing:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- Coefficient of Determination (R² Score)

This methodology ensures maximum fidelity and statistical reliability in all generated forecasts.

---

# Technical Architecture

## Backend and Development Environment

- Python 3.10+
- Jupyter Notebook Framework

## Data Processing Libraries

- Pandas
- NumPy

## Data Visualization Modules

- Matplotlib
- Seaborn

## Machine Learning and Forecasting Implementations

- Scikit-learn
- XGBoost
- Statsmodels
- ARIMA

## Analytical Methodologies

- Advanced Statistical Analysis
- Quantitative Feature Engineering
- Iterative Time-Series Analysis

---

# Dataset Characteristics

The programmatic architecture currently ingests a core set of primary macroeconomic indicators to establish the baseline predictive matrix:

### Primary Core Indicators
- Indian Consumer Price Index (CPI) [Monthly]
- Indian Wholesale Price Index (WPI) [Monthly]
- Indian Gross Domestic Product (GDP) Trajectories [Annual / Proxies]
- Reserve Bank of India (RBI) Benchmark Interest Rates [Monthly]
- Foreign Exchange Variations (USD/INR) [Daily]
- Global Energy Benchmarks (Brent Crude Oil) [Daily]

### Derived Target Variable
- **Inflation Rate:** Dynamically calculated via Year-over-Year (YoY) percentage change derivations against the baseline Indian CPI index data.

### Phase II Extensibility (Planned Expansion)
Future iterations of the model will ingest extended external datasets localized for robust anomaly detection within the Indian economic framework:
- Aggregate National Unemployment Rates
- Indian Basket Crude Oil Valuations
- Domestic Monetary Supply Aggregates (M1/M3)

These metrics function as foundational input features for the systemic predictive modeling and temporal forecasting pipeline.

---

# Project Workflow

## 1. Data Aggregation
Systematic collection of historical macroeconomic and inflation-related datasets from authoritative public and institutional financial sources.

## 2. Data Preprocessing Pipeline
- **Time-Series Frequency Alignment:** Algorithmic downsampling of Daily data (Exchange Rates) and interpolation of Annual data (GDP) to synthesize a unified Monthly frequency matrix.
- Mitigation and imputation of null values.
- Comprehensive data sanitization.
- Algorithmic outlier detection and handling.
- Statistical data normalization.
- Predictive feature selection strategies.

## 3. Exploratory Data Analysis (EDA)

- Empirical trend identification and smoothing
- Statistical distribution variance analysis
- Bivariate and multivariate correlation mapping
- Discovery of latent macroeconomic patterns

## 4. Feature Engineering

- Synthesis of derived analytical variables
- Optimization of critical economic indicators
- Algorithmic dataset dimensionality reduction

## 5. Model Architecture and Training

Training of diverse forecasting algorithms, including:

- Linear Regression
- Random Forest Ensembles
- XGBoost Architectures
- ARIMA Time-Series Models

## 6. Model Evaluation and Validation

Rigorous evaluation of predictive fidelity utilizing:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- Coefficient of Determination (R² Score)

## 7. Predictive Generation

Generate future inflation forecasts and economic trend predictions.

---

# Project Structure

```text
VaticMacro/
│
├── data/
│   ├── [Raw CSV datasets...]
│   └── inflation_dataset.csv (Merged output)
│
├── models/
│   ├── best_model.pkl (Dynamically saved)
│   └── metrics.json
│
├── src/
│   ├── preprocessing.py (Cleans & standardizes data formats)
│   ├── feature_engineering.py (Generates lags & rolling averages)
│   └── train_model.py (Trains multiple models and selects best)
│
├── app/
│   └── templates/
│       ├── dashboard.html
│       ├── analysis.html
│       ├── models.html
│       ├── predict.html
│       └── [Forecast/About pages...]
│
├── .venv/ (Local Python Environment)
├── requirements.txt
├── merge_data.py (Master script to build dataset)
├── app.py (Upcoming Flask Web Application)
├── main.py (Background training trigger)
└── README.md
```

---

# Expected Operational Outcomes

The execution of the framework yields:

- Quantifiable future inflation projections
- Formulated economic trend predictions
- Empirical indicator importance metrics
- Robust forecast accuracy analytics
- Synthesized, graphical economic intelligence

---

# Applied Use Cases

VaticMacro is architected to facilitate:

### Government and Policy Planning
Supports macroeconomic stabilization strategies and empirical policy formulation.

### Financial Institutions
Assists banking and regulatory organizations in extensive systemic risk assessment and operational forecasting.

### Investment Analysis
Offers institutional investors quantifiable data regarding future macroeconomic climates.

### Strategic Corporate Planning
Supports rigorous budgeting, pricing models, and structural strategic allocations.

### Academic Research
Serves as an empirical foundation for advanced economic and temporal forecasting research initiatives.

---

# Future Development Horizons

- **Continuous Integration:** Integration of live, real-time macroeconomic datasets via API pipelines.
- **Hybrid Architectures:** Implementation of dual-model ensembles, matching statistical baseline algorithms like ARIMA with advanced non-linear deep learning (e.g., ARIMA-LSTM).
- **Time-Series Transformers:** Application of state-of-the-art Temporal Fusion Transformers (TFT) equipped with attention mechanics for superior multi-horizon extrapolation.
- **Alternative Behavioral Data:** Integration of Natural Language Processing (NLP) models to ingest and execute sentiment analysis on Central Bank minutes and macroeconomic news indicators.
- **Interactive Dashboards:** Implementation of dynamic, interactive forecasting suites.
- **Granularity:** Region-specific and localized inflation metrics.
- **Scale:** Expansion into cross-border, multi-national macroeconomic modeling.

---

# Research Contributors

- Dhanush Pillay & Shubhangini Dixit

---

# Licensing

This repository and framework are published explicitly for academic, research, and non-commercial educational purposes.

---

## Running locally (Windows)

If you see "No module named 'flask'", use the included startup scripts to create a virtual environment and install dependencies.

- PowerShell (recommended):

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

- CMD:

```cmd
start.bat
```

Or install dependencies manually then run the app:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Static-only fallback (no Python required): open `app/templates` in a simple file server:

```powershell
cd app/templates
python -m http.server 8000
# then open http://127.0.0.1:8000/dashboard.html
```
