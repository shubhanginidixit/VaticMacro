from src.train_model import train
import pandas as pd

print('Loading data (no-prompt runner)...')
df = pd.read_csv('data/inflation_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
if 'INDCPIALLMINMEI' in df.columns and 'CPI' not in df.columns:
    df = df.rename(columns={'INDCPIALLMINMEI': 'CPI'})
    print('Renamed INDCPIALLMINMEI -> CPI')
print('Starting training...')
res = train(df)
print('Training finished, result:', res)
