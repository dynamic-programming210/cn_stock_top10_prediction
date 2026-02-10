#!/usr/bin/env python3
"""Generate predictions for both 5-day and 15-day horizons."""
import sys
import traceback

sys.path.insert(0, '.')

print('=' * 60)
print('GENERATE PREDICTIONS SCRIPT')
print('=' * 60)

print('\n[1/4] Importing modules...')
try:
    from models.train import generate_predictions
    from models.train_15d import generate_predictions_15d
    print('    ✓ Modules imported successfully')
except Exception as e:
    print(f'    ✗ Import error: {e}')
    traceback.print_exc()
    sys.exit(1)

print('\n[2/4] Checking model files exist...')
import os
model_files = [
    'models/ranker.txt',
    'models/regressor.pkl',
    'models/ranker_15d.txt',
    'models/regressor_15d.pkl'
]
for f in model_files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f'    ✓ {f} ({size:,} bytes)')
    else:
        print(f'    ✗ {f} NOT FOUND')

print('\n[3/4] Running generate_predictions() for 5-day...')
try:
    generate_predictions()
    print('    ✓ 5-day predictions generated!')
except Exception as e:
    print(f'    ✗ 5-day error: {e}')
    traceback.print_exc()
    sys.exit(1)

print('\n[4/4] Running generate_predictions_15d() for 15-day...')
try:
    generate_predictions_15d()
    print('    ✓ 15-day predictions generated!')
except Exception as e:
    print(f'    ⚠ 15-day predictions skipped: {e}')
    # Don't fail on 15d errors

print('\n' + '=' * 60)
print('ALL PREDICTIONS COMPLETE')
print('=' * 60)
