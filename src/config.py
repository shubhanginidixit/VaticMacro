"""
Centralized configuration for VaticMacro.
All shared constants (column mappings, file paths) live here
so they are defined once and imported everywhere.
"""

# Maps human-readable indicator names to their CSV column names
COLUMN_MAP = {
    'cpi': 'INDCPIALLMINMEI',
    'wpi': 'WPIATT01INM661N',
    'interest_rate': 'INTDSRINM193N',
    'usd_inr': 'DEXINUS',
    'brent_crude': 'Average of DCOILBRENTEU',
    'gdp_proxy': 'MKTGDPINA646NWDB'
}

# File paths (relative to project root)
MODEL_PATH = 'models/best_model.pkl'
METRICS_PATH = 'models/metrics.json'
DATA_PATH = 'data/inflation_dataset.csv'
