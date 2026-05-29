# Dataset Documentation

## Merged Dataset

**File:** `data/inflation_dataset.csv`
**Rows:** 7,023 (daily frequency, 2000-01-01 to 2026-05-15)
**Columns:** 10

### Column Reference

| # | Column Name | Human Name | Frequency | Source | Unit |
|---|-------------|-----------|-----------|--------|------|
| 1 | `Date` | Date | Daily | — | YYYY-MM-DD |
| 2 | `INDCPIALLMINMEI` | Consumer Price Index | Monthly | FRED | Index |
| 3 | `WPIATT01INM661N` | Wholesale Price Index | Monthly | FRED | Index |
| 4 | `INTDSRINM193N` | Interest Rate | Monthly | FRED | % |
| 5 | `DEXINUS` | USD/INR Exchange Rate | Daily | FRED | INR per USD |
| 6 | `Average of DCOILBRENTEU` | Brent Crude Oil | Daily | FRED | USD/barrel |
| 7 | `datafilenew(india basket crude oil)` | India Basket Crude | Daily | PPAC | USD/barrel |
| 8 | `MKTGDPINA646NWDB` | GDP (nominal) | Annual | FRED | USD |
| 9 | `Unemployment Rate Annually` | Unemployment Rate | Annual | World Bank | % |
| 10 | `INDLORSGPNOSTSAM` | Stock Market Proxy | Daily | FRED | Index |

### Important Notes

1. **Forward-fill behavior:** Monthly and annual indicators are forward-filled across daily rows. This means ~96% of CPI values are copies, not new observations. The training pipeline resamples to monthly to handle this.

2. **CPI data ends early:** The last unique CPI value is from March 2025. All rows after that carry the same CPI (157.55) via forward-fill. The dashboard detects this and uses the last real CPI observation.

3. **Target variable:** YoY inflation is not a column in the dataset — it is computed at training time as:
   ```
   inflation_yoy = ((CPI_t - CPI_{t-12}) / CPI_{t-12}) × 100
   ```

---

## Raw Source Files

Located in `data/`, these are the original CSVs before merging:

| File | Indicator | Format |
|------|-----------|--------|
| `INDCPIALLMINMEI (Consumer Price Index).csv` | CPI | observation_date + value |
| `WPIATT01INM661N (Wholesale Prices Industry Aggregates).csv` | WPI | observation_date + value |
| `INTDSRINM193N (Interest Rate).csv` | Interest Rate | observation_date + value |
| `DEXINUS (USDINR).csv` | USD/INR | observation_date + value |
| `DCOILBRENTEU(crude oil).csv` | Brent Crude | observation_date + value |
| `datafilenew(india basket crude oil).csv` | India Basket Crude | Year/Month pivot |
| `MKTGDPINA646NWDB (GDP Annual).csv` | GDP | observation_date + value |
| `Unemployment Rate Annually.csv` | Unemployment | Years as columns |
| `INDLORSGPNOSTSAM (GDP Proxy).csv` | Stock Market | observation_date + value |

### Merging Process

Run `python merge_data.py` to regenerate `inflation_dataset.csv` from raw files. The merge script:
1. Loads each CSV using `src/preprocessing.py` (handles various formats)
2. Aligns all indicators to the same daily date index
3. Forward-fills and backward-fills missing values
4. Saves the merged result

---

## Engineered Features

The feature engineering pipeline (`src/feature_engineering.py`) creates 44 features from the raw columns:

### Feature Types

| Type | Count | Example | Purpose |
|------|-------|---------|---------|
| Percentage change (30d) | 6 | `DEXINUS_pct_change_30` | Short-term momentum |
| Percentage change (90d) | 6 | `WPIATT01INM661N_pct_change_90` | Medium-term momentum |
| Percentage change (180d) | 6 | `INTDSRINM193N_pct_change_180` | Long-term momentum |
| Lag features (30d) | 6 | `DEXINUS_lag_pct_30` | Delayed effects |
| Lag features (90d) | 6 | `DEXINUS_lag_pct_90` | Delayed effects |
| Rolling average (30d) | 6 | `DEXINUS_rolling_pct_avg_30` | Smoothed momentum |
| Rolling average (90d) | 6 | `DEXINUS_rolling_pct_avg_90` | Smoothed momentum |
| Ratio features | 2 | `WPI_to_CPI_ratio` | Cross-indicator relationships |

### Autoregressive Features (added during training)

| Feature | Description |
|---------|-------------|
| `inflation_lag1` | YoY inflation from 1 month ago |
| `inflation_lag3` | YoY inflation from 3 months ago |
| `inflation_lag6` | YoY inflation from 6 months ago |
| `inflation_rolling_mean_3` | 3-month rolling average of inflation |
| `inflation_rolling_mean_6` | 6-month rolling average of inflation |

These are the strongest predictors (inflation autocorrelation > 0.99 at lag-1).

---

## Model Artifacts

### `models/best_model.pkl`

A joblib-serialized dictionary containing:

```python
{
    'pipeline': Pipeline([...]),      # RobustScaler + RidgeCV
    'feature_columns': [...],         # 24 feature names used
    'best_model_choice': 'RidgeCV',   # Model name
    'model_name': 'RidgeCV'           # Display name
}
```

### `models/metrics.json`

Cross-validation results for all three models:

```json
{
    "best_model": "RidgeCV",
    "training_window": "2005-01-01 to 2022-12-01",
    "monthly_observations": 216,
    "features_used": 24,
    "ridgecv": {
        "r2": [0.434, 0.648, 0.811, 0.958, 0.704],
        "r2_mean": 0.711,
        "mae": [1.459, 0.671, 0.441, 0.386, 0.382],
        "rmse": [1.967, 0.822, 0.541, 0.470, 0.450]
    }
}
```
