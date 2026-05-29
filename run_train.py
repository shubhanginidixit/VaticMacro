from src.train_model import train
import pandas as pd

print('Loading data...')
df = pd.read_csv('data/inflation_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
print('Parsed Date column as datetime')
# Ensure column names expected by trainer
if 'INDCPIALLMINMEI' in df.columns and 'CPI' not in df.columns:
	df = df.rename(columns={'INDCPIALLMINMEI': 'CPI'})
	print('Renamed INDCPIALLMINMEI -> CPI')
print('Starting training...')
train(df)
print('Training complete.')
