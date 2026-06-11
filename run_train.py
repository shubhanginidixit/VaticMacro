from src.train_model import train
from src.feature_engineering import create_features
import pandas as pd

print('Loading data...')
df = pd.read_csv('data/inflation_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
print('Parsed Date column as datetime')

print('Engineering features...')
df = create_features(df)

print('Starting training...')
train(df)
print('Training complete.')
