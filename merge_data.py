import os
import pandas as pd
from src.preprocessing import load_and_clean_data

DATA_FOLDER = "data"

# Columns with stale/synthetic data that provide no predictive signal
EXCLUDED_FILES = [
    "datafilenew(india basket crude oil).csv",   # flat at 64.73 since 2021
    "INDLORSGPNOSTSAM (GDP Proxy).csv",          # flat at 101.52 since 2021
    "Unemployment Rate Annually.csv",             # annual data repeated monthly
    "MKTGDPINA646NWDB (GDP Annual).csv",         # annual GDP repeated monthly
]

def merge_all():
    """Merge all CSV files into a single dataset with aligned dates."""
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv") and f != "inflation_dataset.csv" and f not in EXCLUDED_FILES]

    merged_df = None

    for file in files:
        path = os.path.join(DATA_FOLDER, file)
        print(f"Processing {file}...")

        df = load_and_clean_data(path)

        # Rename date column
        df = df.rename(columns={"observation_date": "Date"})

        # Keep Date and all value columns
        value_cols = [col for col in df.columns if col != "Date"]
        df = df[["Date"] + value_cols]

        if merged_df is None:
            merged_df = df
        else:
            # Use outer join to preserve all dates, then forward fill missing values
            merged_df = pd.merge(
                merged_df,
                df,
                on="Date",
                how="outer",
                suffixes=('', f'_dup_{file}')
            )

    # Drop any duplicate columns created by overlapping merges
    dup_cols = [c for c in merged_df.columns if '_dup_' in str(c)]
    if dup_cols:
        print(f"Warning: Dropping {len(dup_cols)} duplicate columns: {dup_cols}")
        merged_df = merged_df.drop(columns=dup_cols)

    # Sort by date and forward fill missing values
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    merged_df = merged_df.ffill()  # Forward fill
    merged_df = merged_df.bfill()  # Backward fill for remaining NaNs
    
    # Resample to monthly frequency (Month End) to reduce noise
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df = merged_df.set_index('Date').resample('ME').last().reset_index()
    # Forward fill one last time in case some months had entirely missing data prior to resampling
    merged_df = merged_df.ffill()
    
    # Save to CSV
    merged_df.to_csv("data/inflation_dataset.csv", index=False)

    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Date range: {merged_df['Date'].min().strftime('%Y-%m-%d')} to {merged_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Missing values: {merged_df.isnull().sum().sum()}")
    
    print("\n" + "=" * 80)
    print("DATA FRESHNESS REPORT")
    print("=" * 80)
    
    value_cols = [c for c in merged_df.columns if c != 'Date']
    for col in value_cols:
        if col in merged_df.columns:
            # Find the last index where the value changed
            changes = merged_df[col].diff() != 0
            if changes.any():
                last_change_idx = changes[changes].index.max()
                last_change_date = merged_df.loc[last_change_idx, 'Date'].strftime('%Y-%m-%d')
                flat_months = len(merged_df) - 1 - last_change_idx
                
                status = "OK"
                if flat_months > 3:
                    status = f"WARNING: Flatlined for {flat_months} months!"
                    
                print(f"{col[:35]:<35} | Last real data: {last_change_date} | {status}")
            else:
                print(f"{col[:35]:<35} | ALL CONSTANT")
    print("\nDataset saved to: data/inflation_dataset.csv")
    print("=" * 80)
    print("\nReady for ML model training!")

if __name__ == "__main__":
    merge_all()
