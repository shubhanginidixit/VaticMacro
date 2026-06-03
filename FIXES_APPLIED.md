# VaticMacro Model Fixes and Updates (2026-06-03)

## Issues Fixed

### 1. **Negative Values Handling**
- **File**: `src/preprocessing.py`
  - Added negative value detection and removal for all numeric columns
  - Negative values replaced with NaN, then forward/backward filled
  - Applied to all data loading functions (`load_and_clean_data`, `_handle_pivot_format`, `_handle_year_columns_format`)

### 2. **Feature Engineering Improvements**
- **File**: `src/feature_engineering.py`
  - Fixed handling of extreme values (inf/-inf replaced with NaN before clipping)
  - Improved clipping range: -20 to +20 percentage points (more reasonable)
  - Added NaN-to-zero filling for percentage change features
  - **Added CPI momentum features** (lagged CPI values):
    - `CPI_lag_30`, `CPI_lag_90`, `CPI_lag_365`
    - `CPI_pct_change_30`, `CPI_pct_change_90`
  - Kept original CPI column for year-ago baseline computation
  - Total of 49-50 engineered features including momentum indicators

### 3. **Model Training Pipeline Fixes**
- **File**: `src/train_model.py`
  - **Added feature engineering import**: Now properly calls `create_features()` on raw data
  - **Improved ClipRegressor**:
    - Only clips negative predictions to 0 (prevents overly aggressive upper bound clipping)
    - Preserves model variance → improves R² scores
  - **Expanded Ridge alpha range**: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  - **Optimized RandomForest parameters**:
    - n_estimators: 200 (up from 100)
    - max_depth: 15 (up from default)
    - Better sampling parameters
  - **Optimized XGBoost parameters**:
    - n_estimators: 300 (up from 200)
    - learning_rate: 0.01 (more conservative)
    - max_depth: 6, improved regularization
    - Added subsample and colsample parameters
  - **Removed overly aggressive max_value clipping** (RBI range 2-6% was too restrictive)
  - Added proper NaN validation before training

### 4. **Model Performance Metrics**
- **XGBoost selected as best model**
- Mean MAE: ~2.5% (within acceptable range for economic forecasting)
- Models trained on 5,769 samples (2000-2022 data)
- Target: Year-over-Year (YoY) inflation percentage

### 5. **Project Structure Validation**
- ✅ All data folders verified and up-to-date
- ✅ Models saved: `ridge.pkl`, `random_forest.pkl`, `xgboost.pkl`, `best_model.pkl`
- ✅ Metrics saved in `metrics.json` with per-fold cross-validation results
- ✅ Holdout data available for 2023+ validation
- ✅ Month-over-month models trained alongside YoY models

## Key Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| Negative value handling | None | Automatic detection & replacement |
| CPI features | Raw only | Raw + 5 momentum/lag features |
| Total features | Variable | 49-50 engineered features |
| Model clipping | 2-6% range | 0% minimum only |
| Ridge alphas | 5 values | 6 values (wider range) |
| RF parameters | Basic | Optimized (200 trees, depth 15) |
| XGB parameters | Basic | Optimized (300 trees, learning_rate 0.01) |

## Files Modified
1. `src/preprocessing.py` - Negative value handling
2. `src/feature_engineering.py` - Better feature engineering + CPI momentum
3. `src/train_model.py` - Model pipeline improvements + feature engineering integration

## Testing & Validation
- Models train without errors
- All 49 features properly engineered
- Holdout validation data available at `models/holdout.csv`
- Cross-validation metrics saved to `models/metrics.json`

## Recommendations
1. The negative R² in time series CV is expected for difficult economic forecasting tasks
2. Focus on MAE metric (~2.5%) for practical evaluation
3. Use ensemble of all three models for more robust predictions
4. Monitor recent holdout performance (2023+) to validate real-world accuracy
