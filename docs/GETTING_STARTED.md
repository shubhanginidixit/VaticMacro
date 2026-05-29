# Getting Started

## Prerequisites
- Python 3.12+
- pip (Python package manager)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/DhanushPillay/VaticMacro.git
cd VaticMacro
```

### 2. Create virtual environment (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the dashboard
```bash
python run.py
```
Then open **http://localhost:5000** in your browser.

---

## Retraining the Model

A pre-trained model (`models/best_model.pkl`) is included. To retrain with updated data:

```bash
python main.py
```

This runs the full pipeline:
1. Loads and cleans `data/inflation_dataset.csv`
2. Engineers percentage-change and ratio features
3. Resamples to monthly frequency
4. Trains RidgeCV, RandomForest, and XGBoost
5. Saves the best model to `models/best_model.pkl`
6. Saves cross-validation metrics to `models/metrics.json`

---

## Updating the Dataset

To add new economic data:

1. Place updated raw CSV files in `data/`
2. Run the merge script:
   ```bash
   python merge_data.py
   ```
3. Retrain the model:
   ```bash
   python main.py
   ```
4. Restart the Flask app

---

## Application Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `http://localhost:5000/` | Landing page |
| Dashboard | `http://localhost:5000/dashboard` | Inflation overview + indicator cards |
| Analysis | `http://localhost:5000/analysis` | Correlation matrix + distributions |
| Models | `http://localhost:5000/models` | R² comparison + feature importances |
| Predict | `http://localhost:5000/predict` | Scenario builder for what-if analysis |
| Forecast | `http://localhost:5000/forecast` | 12-month historical inflation chart |
| About | `http://localhost:5000/about` | Project information |

---

## Troubleshooting

### Model file not found
```
Warning: Could not load model at models/best_model.pkl
```
**Fix:** Run `python main.py` to train and save the model.

### Import errors
```
ModuleNotFoundError: No module named 'src'
```
**Fix:** Make sure you're running from the project root directory (`VaticMacro/`).

### scikit-learn version warning
```
InconsistentVersionWarning: Trying to unpickle estimator...
```
**Fix:** Run `python main.py` to retrain with your current scikit-learn version.
