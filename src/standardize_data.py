"""
Standardize all CSV files to a consistent format:
- observation_date column (datetime format)
- Single value column with filename-based naming
- Sorted by date, no missing values
"""

import pandas as pd
import os
from preprocessing import load_and_clean_data

def standardize_all_data(start_date='2000-01-01', end_date='2022-12-31'):
    """Load, clean, and standardize all CSV files in the data directory."""
    
    data_dir = r'c:\users\ad\Desktop\VaticMacro\data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    print("=" * 80)
    print(f"DATA STANDARDIZATION PROCESS ({start_date} to {end_date})")
    print("=" * 80)
    
    standardized_data = {}
    failed_files = {}
    
    for file in sorted(csv_files):
        filepath = os.path.join(data_dir, file)
        print(f"\nProcessing: {file}")
        
        try:
            df = load_and_clean_data(filepath, start_date, end_date)
            print(f"  Status: SUCCESS")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Date range: {df['observation_date'].min()} to {df['observation_date'].max()}")
            standardized_data[file] = df
        except Exception as e:
            print(f"  Status: FAILED")
            print(f"  Error: {e}")
            failed_files[file] = str(e)
    
    # Summary
    print("\n" + "=" * 80)
    print("STANDARDIZATION SUMMARY")
    print("=" * 80)
    print(f"Successfully processed: {len(standardized_data)}/{len(csv_files)} files")
    if failed_files:
        print(f"Failed files: {len(failed_files)}")
        for file, error in failed_files.items():
            print(f"  - {file}: {error}")
    
    print("\nStandardized datasets:")
    for file, df in standardized_data.items():
        print(f"  - {file}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("=" * 80)
    
    return standardized_data, failed_files


if __name__ == "__main__":
    standardized_data, failed_files = standardize_all_data()
    
    # Sample output
    if standardized_data:
        first_file = list(standardized_data.keys())[0]
        print(f"\nSample data from {first_file}:")
        print(standardized_data[first_file].head())
