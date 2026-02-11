import pandas as pd
from config import MIN_DOLLAR_VOLUME_ZSCORE, LIMIT_UP_THRESHOLD, LIMIT_DOWN_THRESHOLD

df = pd.read_parquet('data/feat_z.parquet')
print(f'Total rows: {len(df)}')

# Get one day's data
sample_date = df['date'].iloc[len(df)//2]
day_df = df[df['date'] == sample_date].copy()
print(f'Sample date: {sample_date}, stocks: {len(day_df)}')

# Apply filters
orig = len(day_df)

# Filter limit-up using binary column
if 'at_limit_up' in day_df.columns:
    day_df = day_df[day_df['at_limit_up'] == 0]
    print(f'After limit-up filter: {len(day_df)} (filtered {orig - len(day_df)})')

# Filter limit-down
temp = len(day_df)
day_df = day_df[day_df['at_limit_down'] == 0]
print(f'After limit-down filter: {len(day_df)} (filtered {temp - len(day_df)})')

# Filter liquidity (z-scored)
temp = len(day_df)
day_df = day_df[day_df['dollar_volume_5'] >= MIN_DOLLAR_VOLUME_ZSCORE]
print(f'After liquidity filter: {len(day_df)} (filtered {temp - len(day_df)})')

print(f'Final: {len(day_df)}/{orig} = {100*len(day_df)/orig:.1f}% remaining')
