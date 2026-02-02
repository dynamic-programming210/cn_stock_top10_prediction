#!/usr/bin/env python3
"""Display top 10 stock predictions with names and chart links"""
import pandas as pd
from pathlib import Path

# Load predictions
df = pd.read_parquet('outputs/top10_latest.parquet')

# Load Chinese names from CSV (preferred)
cn_names_dict = {}
csv_path = Path('cn_stocks_shg_she_code_name.csv')
if csv_path.exists():
    cn_names = pd.read_csv(csv_path, dtype={'code': str})
    cn_names['code'] = cn_names['code'].str.zfill(6)
    cn_names_dict = dict(zip(cn_names['code'], cn_names['name']))

# Load English names from universe metadata (fallback)
meta = pd.read_parquet('data/universe_meta.parquet')
en_names_dict = dict(zip(meta['symbol'], meta['name']))

def get_name(symbol):
    """Get stock name - prefer Chinese, fallback to English"""
    code = symbol.split('.')[0]
    if code in cn_names_dict:
        return cn_names_dict[code]
    if symbol in en_names_dict:
        return en_names_dict[symbol]
    return 'N/A'

print('='*80)
print('         TOP 10 CHINESE STOCK PREDICTIONS - 5-Day Forward Returns')
print('='*80)
print()

# Get prediction date
pred_date = df['date'].iloc[0] if 'date' in df.columns else 'Latest'
print(f"Prediction Date: {pred_date}")
print()

print(f"{'Rank':<6}{'Symbol':<12}{'Name':<22}{'Pred Ret':<12}{'Score':<10}")
print('-'*80)

for i, row in df.iterrows():
    symbol = row['symbol']
    code = symbol.split('.')[0]
    exchange = row.get('exchange', 'SHE' if code.startswith('0') or code.startswith('3') else 'SHG')
    
    # Get name (Chinese preferred)
    name = get_name(symbol)
    if len(name) > 20:
        name = name[:18] + '..'
    
    pred_ret = row.get('pred_ret_5', row.get('pred_ret', 0))
    rank_score = row.get('rank_score', 0)
    
    print(f"{i+1:<6}{symbol:<12}{name:<22}{pred_ret:>8.2%}    {rank_score:>6.3f}")

print('-'*80)
print()
print("Chart Links (East Money 东方财富):")
print()

for i, row in df.iterrows():
    symbol = row['symbol']
    code = symbol.split('.')[0]
    exchange = row.get('exchange', 'SHE' if code.startswith('0') or code.startswith('3') else 'SHG')
    name = get_name(symbol)
    if len(name) > 12:
        name = name[:10] + '..'
    
    # East Money chart link
    mkt = 1 if exchange == 'SHG' else 0
    chart_url = f'https://quote.eastmoney.com/{mkt}.{code}.html'
    
    print(f"  {i+1}. {symbol} ({name}): {chart_url}")

print()
print('='*80)
