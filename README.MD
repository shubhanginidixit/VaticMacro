# 📈 VaticMacro

## Inflation Prediction & Economic Trend Forecasting

VaticMacro is a machine learning-based economic forecasting platform designed to predict inflation trends by analyzing historical economic indicators and market conditions. The system leverages data analytics, statistical modeling, and predictive forecasting techniques to provide accurate inflation estimates and actionable economic insights.

The platform helps governments, businesses, researchers, investors, and financial analysts understand inflation patterns, evaluate economic risks, and make informed decisions based on future economic projections.

---

# Problem & Solution

## Problem Statement

Inflation is one of the most significant indicators of economic health, directly impacting purchasing power, investment returns, business operations, government policies, and financial planning.

Predicting inflation accurately is a complex challenge because inflation is influenced by numerous interconnected factors, including:

- Consumer Price Index (CPI)
- Wholesale Price Index (WPI)
- Interest Rates
- GDP Growth
- Unemployment Rates
- Exchange Rates
- Oil and Commodity Prices
- Money Supply

Traditional forecasting methods often struggle to identify hidden patterns within large volumes of economic data and may fail to adapt to rapidly changing economic conditions.

As economies become increasingly dynamic and interconnected, there is a growing need for a data-driven forecasting solution capable of analyzing multiple economic indicators simultaneously and generating reliable inflation predictions.

---

## The VaticMacro Solution

VaticMacro provides a centralized platform for economic trend analysis and inflation forecasting.

The system:

- Collects and processes historical economic data
- Identifies relationships between key economic indicators
- Applies machine learning algorithms to forecast inflation rates
- Visualizes economic trends through charts and dashboards
- Evaluates forecasting performance using industry-standard metrics
- Generates actionable insights for decision-makers

By transforming complex economic datasets into meaningful forecasts, VaticMacro enables users to better understand future economic conditions and make informed strategic decisions.

---

# Key Capabilities

## 📊 Economic Data Analysis

- Historical inflation trend analysis
- Economic indicator comparison
- Correlation analysis
- Data preprocessing and cleaning
- Feature engineering

---

## 🤖 Machine Learning Forecasting

- Inflation prediction using historical economic data
- Predictive modeling using multiple algorithms
- Time-series forecasting
- Economic trend forecasting
- Future inflation estimation

Supported Models:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- ARIMA Time Series Forecasting

---

## 📈 Data Visualization

- Inflation trend charts
- Forecast comparison graphs
- Correlation heatmaps
- Economic indicator dashboards
- Interactive visual insights

---

## 📋 Performance Evaluation

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- R² Score

This ensures accurate and reliable forecasting performance.

---

# Technology Stack

## Backend & Development

- Python 3.10+
- Jupyter Notebook

## Data Processing

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn

## Machine Learning & Forecasting

- Scikit-learn
- XGBoost
- Statsmodels
- ARIMA

## Data Analysis

- Statistical Analysis
- Feature Engineering
- Time-Series Analysis

---

# Dataset Features

The dataset contains historical economic indicators including:

- Consumer Price Index (CPI)
- Wholesale Price Index (WPI)
- GDP Growth Rate
- Interest Rates
- Unemployment Rate
- Exchange Rates
- Crude Oil Prices
- Money Supply
- Inflation Rate Records

These indicators serve as input features for predictive modeling and forecasting.

---

# Project Workflow

## 1. Data Collection

Collect historical economic and inflation-related data from reliable public and financial sources.

## 2. Data Preprocessing

- Missing value handling
- Data cleaning
- Outlier detection
- Data normalization
- Feature selection

## 3. Exploratory Data Analysis (EDA)

- Trend identification
- Distribution analysis
- Correlation analysis
- Economic pattern discovery

## 4. Feature Engineering

- Creation of derived variables
- Selection of relevant economic indicators
- Dataset optimization

## 5. Model Development

Train multiple forecasting models:

- Linear Regression
- Random Forest
- XGBoost
- ARIMA

## 6. Model Evaluation

Evaluate forecasting accuracy using:

- MAE
- RMSE
- MSE
- R² Score

## 7. Inflation Forecast Generation

Generate future inflation forecasts and economic trend predictions.

---

# Project Structure

```text
VaticMacro/
│
├── data/
│   ├── inflation_dataset.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   ├── Forecasting.ipynb
│
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│
├── visualizations/
│   ├── inflation_trends.png
│   ├── correlation_heatmap.png
│   └── forecast_results.png
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── main.py
```

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/VaticMacro.git

cd VaticMacro
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

```bash
python main.py
```

---

# Expected Outcomes

The system provides:

- Future inflation forecasts
- Economic trend predictions
- Indicator importance analysis
- Forecast accuracy reports
- Visualized economic insights

---

# Applications

VaticMacro can be used for:

### Government & Policy Planning
Support inflation control measures and economic policy decisions.

### Financial Institutions
Assist banks and financial organizations in risk assessment and forecasting.

### Investment Analysis
Help investors understand future economic conditions.

### Business Planning
Support budgeting, pricing, and strategic planning decisions.

### Academic Research
Provide a framework for economic and forecasting research projects.

---

# Future Enhancements

- Real-time economic data integration
- Interactive forecasting dashboard
- Regional inflation prediction
- Advanced Deep Learning models (LSTM, GRU)
- Automated report generation
- Multi-country inflation forecasting

---

# Contributors

- Dhanush Pillay & Shubhangini Dixit

---

# License

This project is developed for educational, research, and learning purposes.








