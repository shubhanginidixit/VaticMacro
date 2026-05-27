"""
Complete example: Preprocessing -> Merging -> Using inflation_dataset.csv
This script demonstrates the entire workflow for ML model preparation
"""

import sys
sys.path.insert(0, 'src')

from preprocessing import load_and_clean_data
import pandas as pd
import os

print("\n" + "=" * 100)
print("COMPLETE WORKFLOW EXAMPLE: PREPROCESSING -> MERGING -> ML DATASET")
print("=" * 100)

# ============================================================================
# STEP 1: PREPROCESSING - Load individual CSV files
# ============================================================================
print("\n" + "=" * 100)
print("STEP 1: PREPROCESSING - Load and Clean Individual CSV Files")
print("=" * 100)

# Example 1: Load CPI (Consumer Price Index)
print("\n[Example 1] Loading Consumer Price Index (CPI)")
print("-" * 100)
cpi_df = load_and_clean_data("data/INDCPIALLMINMEI (Consumer Price Index).csv")
print(f"Original shape after preprocessing: {cpi_df.shape}")
print(f"Date range: {cpi_df['observation_date'].min()} to {cpi_df['observation_date'].max()}")
print(f"Missing values: {cpi_df.isnull().sum().sum()}")
print("\nFirst 5 rows of CPI:")
print(cpi_df.head().to_string())

# Example 2: Load Interest Rate
print("\n[Example 2] Loading Interest Rate")
print("-" * 100)
interest_df = load_and_clean_data("data/INTDSRINM193N (Interest Rate).csv")
print(f"Original shape after preprocessing: {interest_df.shape}")
print(f"Date range: {interest_df['observation_date'].min()} to {interest_df['observation_date'].max()}")
print(f"Missing values: {interest_df.isnull().sum().sum()}")
print("\nFirst 5 rows of Interest Rate:")
print(interest_df.head().to_string())

# Example 3: Load USD/INR Exchange Rate
print("\n[Example 3] Loading USD/INR Exchange Rate")
print("-" * 100)
exchange_df = load_and_clean_data("data/DEXINUS (USDINR).csv")
print(f"Original shape after preprocessing: {exchange_df.shape}")
print(f"Date range: {exchange_df['observation_date'].min()} to {exchange_df['observation_date'].max()}")
print(f"Missing values: {exchange_df.isnull().sum().sum()}")
print("\nFirst 5 rows of Exchange Rate:")
print(exchange_df.head().to_string())

# ============================================================================
# STEP 2: MERGING - Combine all datasets
# ============================================================================
print("\n\n" + "=" * 100)
print("STEP 2: MERGING - Combining All Datasets into One")
print("=" * 100)

print("\nBefore Merge:")
print(f"  - CPI: {len(cpi_df)} rows")
print(f"  - Interest Rate: {len(interest_df)} rows")
print(f"  - Exchange Rate: {len(exchange_df)} rows")

# Merge CPI + Interest Rate
merged = pd.merge(cpi_df, interest_df, left_on='observation_date', right_on='observation_date', how='outer')
print(f"\nAfter merging CPI + Interest Rate: {len(merged)} rows")

# Merge with Exchange Rate
merged = pd.merge(merged, exchange_df, left_on='observation_date', right_on='observation_date', how='outer')
print(f"After merging with Exchange Rate: {len(merged)} rows")

# Forward fill to handle missing values from different frequencies
merged = merged.sort_values('observation_date')
merged = merged.ffill().bfill()
print(f"After forward-fill (no missing values): {len(merged)} rows, Missing: {merged.isnull().sum().sum()}")

print("\nSample of merged data:")
print(merged[['observation_date', 'INDCPIALLMINMEI', 'INTDSRINM193N', 'DEXINUS']].head(10).to_string())

# ============================================================================
# STEP 3: USING INFLATION_DATASET - Load the pre-merged file
# ============================================================================
print("\n\n" + "=" * 100)
print("STEP 3: USING INFLATION_DATASET - Load Pre-Merged Complete Dataset")
print("=" * 100)

# Load the final merged dataset
inflation_dataset = pd.read_csv("data/inflation_dataset.csv")

print(f"\nInflation Dataset Shape: {inflation_dataset.shape}")
print(f"Rows: {len(inflation_dataset)} | Columns: {len(inflation_dataset.columns)}")
print(f"Date Range: {inflation_dataset['Date'].min()} to {inflation_dataset['Date'].max()}")
print(f"Missing Values: {inflation_dataset.isnull().sum().sum()}")

print("\nColumn Names (Features for ML):")
for i, col in enumerate(inflation_dataset.columns, 1):
    print(f"  {i}. {col}")

# ============================================================================
# STEP 4: ANALYSIS - Show statistics and insights
# ============================================================================
print("\n\n" + "=" * 100)
print("STEP 4: ANALYSIS - Data Summary Statistics")
print("=" * 100)

print("\nDescriptive Statistics:")
print(inflation_dataset.describe().to_string())

# Show specific date ranges
print("\n\nData from January 2000:")
jan_2000 = inflation_dataset[inflation_dataset['Date'].str.startswith('2000-01')].head(5)
print(jan_2000.to_string())

print("\n\nData from December 2022 (Most Recent):")
dec_2022 = inflation_dataset[inflation_dataset['Date'].str.startswith('2022-12')].tail(5)
print(dec_2022.to_string())

# ============================================================================
# STEP 5: READY FOR ML - Show how to use for training
# ============================================================================
print("\n\n" + "=" * 100)
print("STEP 5: READY FOR ML MODEL TRAINING")
print("=" * 100)

print("\nHow to use this dataset for ML:")
print("""
# Load the dataset
df = pd.read_csv('data/inflation_dataset.csv')

# Define features (X) and target (y)
X = df[['DEXINUS', 'Average of DCOILBRENTEU', 'INTDSRINM193N', 
         'INDLORSGPNOSTSAM', 'MKTGDPINA646NWDB']]  # Predictors
y = df['INDCPIALLMINMEI']  # Target (CPI to forecast)

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")
""")

print("\n" + "=" * 100)
print("WORKFLOW COMPLETE! Data is ready for ML model training.")
print("=" * 100)
