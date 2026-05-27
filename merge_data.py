import os
import pandas as pd
from src.preprocessing import load_and_clean_data

DATA_FOLDER = "data"

def merge_all():
    """Merge all CSV files into a single dataset with aligned dates."""
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

    merged_df = None

    for file in files:
        path = os.path.join(DATA_FOLDER, file)
        print(f"Processing {file}...")

        df = load_and_clean_data(path)

        # Rename date column
        df = df.rename(columns={"observation_date": "Date"})

        # Keep only Date + value column
        value_col = [col for col in df.columns if col != "Date"][0]
        df = df[["Date", value_col]]

        if merged_df is None:
            merged_df = df
        else:
            # Use outer join to preserve all dates, then forward fill missing values
            merged_df = pd.merge(
                merged_df,
                df,
                on="Date",
                how="outer"
            )

    # Sort by date and forward fill missing values
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    merged_df = merged_df.ffill()  # Forward fill
    merged_df = merged_df.bfill()  # Backward fill for remaining NaNs
    
    # Save to CSV
    merged_df.to_csv("data/inflation_dataset.csv", index=False)

    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    print(f"Missing values: {merged_df.isnull().sum().sum()}")
    print(f"\nDataset saved to: data/inflation_dataset.csv")
    print("=" * 80)
    print("\nReady for ML model training!")

if __name__ == "__main__":
    merge_all()
