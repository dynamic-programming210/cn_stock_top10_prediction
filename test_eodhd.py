#!/usr/bin/env python3
"""Test EODHD API for stock 300136"""
import os
import requests

# Get API key
api_key = os.environ.get('EODHD_API_KEY')
if not api_key:
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('EODHD_API_KEY'):
                    api_key = line.split('=')[1].strip().strip('"').strip("'")
                    break
    except:
        pass

if not api_key:
    print('No API key found')
    exit(1)

print(f'API key found: {api_key[:8]}...')

# Test 300136 on Shenzhen exchange (SHE)
symbol = '300136.SHE'
url = f'https://eodhd.com/api/eod/{symbol}?api_token={api_key}&fmt=json&period=d&from=2026-01-01'

print(f'Fetching: {symbol}')
resp = requests.get(url)
print(f'Status: {resp.status_code}')

if resp.status_code == 200:
    data = resp.json()
    if data:
        print(f'Records: {len(data)}')
        print('Latest data:')
        for row in data[-5:]:
            print(f"  {row['date']}: open={row['open']}, high={row['high']}, low={row['low']}, close={row['close']}, volume={row['volume']}")
    else:
        print('No data returned (empty list)')
else:
    print(f'Error: {resp.text[:500]}')
