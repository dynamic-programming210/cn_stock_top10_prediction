#!/usr/bin/env python3
"""Estimate backtest time"""
import time
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/yadu/Documents/doc/cn_stock')
from models.train import TwoStageModel

df = pd.read_parquet('/Users/yadu/Documents/doc/cn_stock/data/feat_z.parquet')
print('Data loaded')
print('Total rows:', len(df))
print('Total symbols:', df['symbol'].nunique())

# Sample 800 symbols, take 252 days of data
symbols = df['symbol'].unique()[:800]
dates = sorted(df['date'].unique())[:252]
train_df = df[(df['symbol'].isin(symbols)) & (df['date'].isin(dates))]
print('Training samples:', len(train_df))

# Time fast mode
model = TwoStageModel()
t0 = time.time()
model.train(train_df, validation_split=0.15, fast_mode=True)
fast_time = time.time() - t0
print('Fast mode:', round(fast_time, 1), 's')

# Time normal mode  
model2 = TwoStageModel()
t0 = time.time()
model2.train(train_df, validation_split=0.15, fast_mode=False)
normal_time = time.time() - t0
print('Normal mode:', round(normal_time, 1), 's')

# ~23 folds for 480 test days
n_folds = 23
print()
print('='*50)
print('ESTIMATED TOTAL BACKTEST TIME (800 symbols):')
print('  Fast mode:  ', round(n_folds * fast_time / 60, 1), 'minutes')
print('  Normal mode:', round(n_folds * normal_time / 60, 1), 'minutes')
print('='*50)
