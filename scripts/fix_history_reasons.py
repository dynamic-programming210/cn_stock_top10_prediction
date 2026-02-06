#!/usr/bin/env python3
"""Fix missing reasons in history file"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from models.selection_reasons import add_selection_reasons
from config import TOP10_HISTORY_FILE

print('Loading history...')
history = pd.read_parquet(TOP10_HISTORY_FILE)
print(f'Total entries: {len(history)}')

# Count entries without reasons
missing = history['reason_cn'].isna().sum()
print(f'Entries without reasons: {missing}')

# Add reasons for entries that don't have them
if missing > 0:
    print('Adding reasons to entries without them...')
    # Process entries without reasons
    mask = history['reason_cn'].isna()
    entries_without_reasons = history[mask].copy()
    entries_without_reasons = add_selection_reasons(entries_without_reasons, include_news=True)
    
    # Update the history
    history.loc[mask, 'reason_cn'] = entries_without_reasons['reason_cn'].values
    
    # Save
    history.to_parquet(TOP10_HISTORY_FILE, index=False)
    print('Saved updated history')

# Verify
print()
print('After fix:')
for date in sorted(history['date'].unique()):
    sample = history[history['date'] == date].iloc[0]
    reason = sample.get('reason_cn', 'NOT FOUND')
    reason_preview = reason[:50] if reason else "None"
    print(f'{date.date()}: {reason_preview}...')

print('\nDone!')
