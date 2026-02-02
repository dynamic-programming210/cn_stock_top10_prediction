#!/usr/bin/env python3
"""Display top 10 stock predictions with names, sectors, and chart links"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

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

# Import sector functions
try:
    from data.sectors import get_stock_sector, get_sector_name
    HAS_SECTORS = True
except ImportError:
    HAS_SECTORS = False

def get_name(symbol):
    """Get stock name - prefer Chinese, fallback to English"""
    code = symbol.split('.')[0]
    if code in cn_names_dict:
        return cn_names_dict[code]
    if symbol in en_names_dict:
        return en_names_dict[symbol]
    return 'N/A'

def get_sector_info(symbol, name=''):
    """Get sector info for a stock"""
    if not HAS_SECTORS:
        return ('其他', 'Other')
    code = symbol.split('.')[0]
    sector = get_stock_sector(code, name)
    return (get_sector_name(sector, chinese=True), get_sector_name(sector, chinese=False))

print('='*90)
print('           TOP 10 CHINESE STOCK PREDICTIONS - 5-Day Forward Returns')
print('='*90)
print()

# Get prediction date
pred_date = df['date'].iloc[0] if 'date' in df.columns else 'Latest'
print(f"Prediction Date: {pred_date}")
print()

# Header with sector column
print(f"{'Rank':<5}{'Symbol':<12}{'Name':<18}{'Sector':<10}{'Pred Ret':<10}{'Score':<8}")
print('-'*90)

sector_counts = {}

for i, row in df.iterrows():
    symbol = row['symbol']
    code = symbol.split('.')[0]
    exchange = row.get('exchange', 'SHE' if code.startswith('0') or code.startswith('3') else 'SHG')
    
    # Get name (Chinese preferred)
    name = get_name(symbol)
    name_display = name[:16] + '..' if len(name) > 16 else name
    
    # Get sector
    if 'sector_cn' in row:
        sector_cn = row['sector_cn']
    else:
        sector_cn, _ = get_sector_info(symbol, name)
    
    # Track sector counts
    sector_counts[sector_cn] = sector_counts.get(sector_cn, 0) + 1
    
    sector_display = sector_cn[:8] if len(sector_cn) > 8 else sector_cn
    
    pred_ret = row.get('pred_ret_5', row.get('pred_ret', 0))
    rank_score = row.get('rank_score', 0)
    
    print(f"{i+1:<5}{symbol:<12}{name_display:<18}{sector_display:<10}{pred_ret:>7.2%}   {rank_score:>6.3f}")

print('-'*90)

# Show sector distribution
print()
print("Sector Distribution (行业分布):")
for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
    print(f"  • {sector}: {count} stocks")

print()
print("Chart Links (东方财富):")
print()

for i, row in df.iterrows():
    symbol = row['symbol']
    code = symbol.split('.')[0]
    exchange = row.get('exchange', 'SHE' if code.startswith('0') or code.startswith('3') else 'SHG')
    name = get_name(symbol)
    name_short = name[:10] + '..' if len(name) > 10 else name
    
    # East Money chart link
    mkt = 1 if exchange == 'SHG' else 0
    chart_url = f'https://quote.eastmoney.com/{mkt}.{code}.html'
    
    print(f"  {i+1}. {symbol} ({name_short}): {chart_url}")

print()
print('='*90)
