import pandas as pd

df = pd.read_csv('data/inflation_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

print('Last 30 rows of CPI data:')
print(df[['Date', 'INDCPIALLMINMEI']].tail(30).to_string())

print('\n\nHistorical inflation from 2023 onwards:')
for year in [2023, 2024, 2025, 2026]:
    year_data = df[df['Date'].dt.year == year]
    if len(year_data) > 0:
        min_cpi = year_data['INDCPIALLMINMEI'].min()
        max_cpi = year_data['INDCPIALLMINMEI'].max()
        print(f"{year}: Min={min_cpi:.4f}, Max={max_cpi:.4f}, Change={(max_cpi-min_cpi)/min_cpi*100:.2f}%")
