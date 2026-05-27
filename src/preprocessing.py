import pandas as pd
import os

def load_and_clean_data(path, start_date='2000-01-01', end_date='2022-12-31'):
    """
    Load and clean data from CSV files with various formats.
    Handles multiple column naming conventions and date formats.
    Filters data to specified date range.
    """
    df = pd.read_csv(path)
    
    # Get the filename for context
    filename = os.path.basename(path)
    
    # Detect and standardize date column
    date_col = None
    if 'observation_date' in df.columns:
        date_col = 'observation_date'
    elif 'Row Labels' in df.columns:
        date_col = 'Row Labels'
        # Special handling: remove Grand Total row if present
        df = df[df['Row Labels'] != 'Grand Total'].copy()
    elif 'Year/Month' in df.columns:
        # India basket file - needs special handling
        return _handle_pivot_format(df, filename, start_date, end_date)
    elif any(col.isdigit() for col in df.columns):
        # Unemployment Rate Annually format - years as columns
        return _handle_year_columns_format(df, filename, start_date, end_date)
    else:
        # Try to find date-like column
        date_candidates = [col for col in df.columns if 'date' in col.lower() or 'label' in col.lower()]
        if date_candidates:
            date_col = date_candidates[0]
        else:
            raise ValueError(f"No date column found in {filename}. Columns: {list(df.columns)}")
    
    # Rename date column to standard name if needed
    if date_col and date_col != 'observation_date':
        df = df.rename(columns={date_col: 'observation_date'})
    
    # Convert Date to datetime
    df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
    df = df.dropna(subset=['observation_date'])
    
    # Filter to date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df['observation_date'] >= start) & (df['observation_date'] <= end)]
    
    # Sort by date
    df = df.sort_values('observation_date')
    
    # Handle missing values - forward fill then backward fill
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    return df


def _handle_pivot_format(df, filename, start_date='2000-01-01', end_date='2022-12-31'):
    """
    Handle pivot-formatted data (like India basket crude oil).
    Converts from wide format to long format.
    """
    # Melt the dataframe
    melted = df.melt(id_vars=['Year/Month'], var_name='Year', value_name='value')
    melted['observation_date'] = pd.to_datetime(melted['Year/Month'] + ' ' + melted['Year'], errors='coerce')
    
    # Drop rows with invalid dates
    melted = melted.dropna(subset=['observation_date'])
    melted = melted[['observation_date', 'value']].copy()
    melted = melted.rename(columns={'value': filename.replace('.csv', '')})
    
    # Filter to date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    melted = melted[(melted['observation_date'] >= start) & (melted['observation_date'] <= end)]
    
    # Sort by date
    melted = melted.sort_values('observation_date')
    melted = melted.ffill().bfill()
    
    return melted


def _handle_year_columns_format(df, filename, start_date='2000-01-01', end_date='2022-12-31'):
    """
    Handle data with years as columns (like Unemployment Rate Annually).
    Converts from wide format to long format.
    """
    # Get only numeric columns (years)
    year_cols = [col for col in df.columns if col.isdigit()]
    
    # Find the country/indicator identifier column
    id_col = None
    for col in df.columns:
        if not col.isdigit() and df[col].notna().any():
            id_col = col
            break
    
    if id_col is None:
        id_col = df.columns[0]
    
    # Melt the dataframe
    melted = df.melt(id_vars=[id_col], value_vars=year_cols, var_name='Year', value_name='value')
    
    # Create observation_date
    melted['observation_date'] = pd.to_datetime(melted['Year'] + '-01-01')
    
    # Keep only observation_date and value columns
    melted = melted[['observation_date', 'value']].copy()
    melted = melted.rename(columns={'value': filename.replace('.csv', '')})
    
    # Drop NaN values
    melted = melted.dropna(subset=['observation_date', melted.columns[1]])
    
    # Filter to date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    melted = melted[(melted['observation_date'] >= start) & (melted['observation_date'] <= end)]
    
    # Sort by date
    melted = melted.sort_values('observation_date')
    
    return melted
