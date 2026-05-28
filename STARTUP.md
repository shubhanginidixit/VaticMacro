# VaticMacro - Institutional Research Dashboard

## Quick Start (One Link!)

### Option 1: Simple Launcher ⭐ (Recommended)
```bash
python run.py
```
Then open your browser to: **http://localhost:5000**

### Option 2: Flask App
```bash
python app.py
```
Then open your browser to: **http://localhost:5000**

---

## Single Entry Point
**http://localhost:5000** - Homepage with navigation to all sections

## All Features
- **Dashboard** - Current inflation rates and key indicators
- **Analysis** - Correlation matrix and statistical patterns  
- **Models** - ML model performance comparison
- **Predict** - Scenario builder for inflation predictions
- **Forecast** - 12-month inflation trends with visualizations

---

## What You Need
- Python 3.12+
- Flask 3.1.3
- pandas 3.0.3
- scikit-learn, XGBoost (for predictions)

## Data Source
All charts and predictions use real data from: `data/inflation_dataset.csv`

---

## Project Structure
```
VaticMacro/
├── run.py                 # ⭐ Easy launcher
├── app.py                 # Flask application
├── data/                  # CSV datasets
├── models/                # ML models
├── src/                   # Python modules
└── app/templates/         # Material Design 3 UI
```
