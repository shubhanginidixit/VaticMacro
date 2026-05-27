from src.preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.train_model import train

# Load merged dataset
df = load_and_clean_data("data/inflation_dataset.csv")

# Feature engineering
df = create_features(df)

print("Data ready:")
print(df.head())

# Train model
model = train(df)

print("Model training completed")
