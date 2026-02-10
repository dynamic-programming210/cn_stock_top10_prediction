#!/usr/bin/env python3
"""Check how many Chinese stocks are missing from universe"""
import os
import requests
import pandas as pd

api_key = os.environ.get('EODHD_API_KEY', '697c670dd1a9d9.07390782')

# Fetch all symbols from both exchanges
all_stocks = []
for exchange in ['SHG', 'SHE']:
    url = f'https://eodhd.com/api/exchange-symbol-list/{exchange}?api_token={api_key}&fmt=json'
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        df = pd.DataFrame(data)
        df['exchange'] = exchange
        all_stocks.append(df)
        print(f'{exchange}: {len(df)} total symbols')

all_df = pd.concat(all_stocks, ignore_index=True)

# Filter to common stocks only
common_stocks = all_df[all_df['Type'] == 'Common Stock'].copy()
print(f'Total Common Stocks: {len(common_stocks)}')

# Breakdown by board type
def get_board(code, exchange):
    code = str(code)
    if exchange == 'SHG':
        if code.startswith(('600', '601', '603', '605')):
            return 'Main Board (SHG)'
        elif code.startswith('688'):
            return 'STAR Market (科创板)'
        else:
            return 'Other SHG'
    elif exchange == 'SHE':
        if code.startswith(('000', '001', '002')):
            return 'Main Board (SHE)'
        elif code.startswith(('300', '301')):
            return 'ChiNext (创业板)'
        else:
            return 'Other SHE'
    return 'Unknown'

common_stocks['board'] = common_stocks.apply(lambda r: get_board(r['Code'], r['exchange']), axis=1)

print()
print('=== Breakdown by Board ===')
for board, count in common_stocks['board'].value_counts().items():
    print(f'  {board}: {count}')

# Check current universe
universe = pd.read_parquet('data/universe.parquet')
print()
print(f'=== Current Universe: {len(universe)} stocks ===')

# Calculate missing
chinext = common_stocks[common_stocks['board'] == 'ChiNext (创业板)']
star = common_stocks[common_stocks['board'] == 'STAR Market (科创板)']

print()
print('=== MISSING STOCKS ===')
print(f'ChiNext (创业板) 300xxx/301xxx: {len(chinext)} stocks - ALL MISSING!')
print(f'STAR Market (科创板) 688xxx: {len(star)} stocks - ALL MISSING!')
print(f'Total missed: {len(chinext) + len(star)} stocks')

# Check if 300136 is in the list
has_300136 = '300136' in chinext['Code'].values
print()
print(f'300136 (信维通信) in ChiNext list: {has_300136}')
