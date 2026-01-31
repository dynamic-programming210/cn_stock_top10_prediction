#!/usr/bin/env python
"""Quick script to analyze backtest results"""
import pandas as pd
import numpy as np

df = pd.read_parquet('outputs/backtest_results.parquet')
df = df.dropna(subset=['avg_return'])

print("="*60)
print("       BACKTEST SUMMARY")
print("="*60)
print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Trading Days: {len(df)}")
print("-"*60)

total_ret = (1 + df['avg_return']).prod() - 1
ann_vol = df['avg_return'].std() * np.sqrt(252)
sharpe = (df['avg_return'].mean() * 252) / ann_vol if ann_vol > 0 else 0

print(f"Total Return:        {total_ret:.2%}")
print(f"Annualized Vol:      {ann_vol:.2%}")
print(f"Sharpe Ratio:        {sharpe:.2f}")
print("-"*60)
print(f"Win Rate:            {(df['avg_return'] > 0).mean():.2%}")
print(f"Avg Return/Day:      {df['avg_return'].mean()*100:.3f}%")
print(f"Avg Hit Rate:        {df['hit_rate'].mean():.2%}")
print(f"Direction Accuracy:  {df['direction_accuracy'].mean():.2%}")
print("="*60)
