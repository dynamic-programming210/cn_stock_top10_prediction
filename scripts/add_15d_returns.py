"""
Add 15-day forward returns to the features file
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BARS_FILE, FEATURES_Z_FILE

print('Loading bars data...')
bars = pd.read_parquet(BARS_FILE)
bars = bars.sort_values(['symbol', 'date'])

# Compute 15-day forward returns
print('Computing 15-day forward returns...')
bars['fwd_ret_15'] = bars.groupby('symbol')['close'].pct_change(15).shift(-15)

# Check results
print(f'Bars with valid fwd_ret_15: {bars["fwd_ret_15"].notna().sum()} / {len(bars)}')
print(f'Mean fwd_ret_15: {bars["fwd_ret_15"].mean():.4f}')
print(f'Std fwd_ret_15: {bars["fwd_ret_15"].std():.4f}')

# Load features and merge
print()
print('Loading features...')
feat = pd.read_parquet(FEATURES_Z_FILE)
print(f'Features shape before: {feat.shape}')

# Remove existing fwd_ret_15 if present (avoid duplicates)
if 'fwd_ret_15' in feat.columns:
    feat = feat.drop(columns=['fwd_ret_15'])

# Merge 15-day forward returns
feat = feat.merge(
    bars[['symbol', 'date', 'fwd_ret_15']],
    on=['symbol', 'date'],
    how='left'
)
print(f'Features shape after merge: {feat.shape}')
print(f'Valid fwd_ret_15 in features: {feat["fwd_ret_15"].notna().sum()}')

# Save updated features
print()
print('Saving updated features...')
feat.to_parquet(FEATURES_Z_FILE, index=False)
print('Done! Features now include fwd_ret_15')
