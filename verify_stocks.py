#!/usr/bin/env python3
"""Verify stocks can be fetched from EODHD and are in universe"""
import requests
import pandas as pd

api_key = '697c670dd1a9d9.07390782'
stocks = ['002156', '300017', '300136', '300308', '300502', '603629', '688012', '688691', '688981']

def get_exchange(code):
    """Determine exchange based on stock code"""
    if code.startswith(('600', '601', '603', '605', '688', '689')):
        return 'SHG'
    elif code.startswith(('000', '001', '002', '003', '300', '301')):
        return 'SHE'
    return 'Unknown'

def is_in_new_universe(code, exchange):
    """Check if stock would be in the updated universe"""
    if exchange == 'SHG':
        return code.startswith(('600', '601', '603', '605', '688', '689'))
    elif exchange == 'SHE':
        return code.startswith(('000', '001', '002', '003', '300', '301'))
    return False

print("=" * 80)
print("Verifying stocks from EODHD API")
print("=" * 80)

results = []
for code in stocks:
    exchange = get_exchange(code)
    symbol = f"{code}.{exchange}"
    
    # Check if in new universe definition
    in_universe = is_in_new_universe(code, exchange)
    
    # Fetch from EODHD
    url = f'https://eodhd.com/api/eod/{symbol}?api_token={api_key}&fmt=json&period=d&from=2026-01-01'
    resp = requests.get(url)
    
    if resp.status_code == 200:
        data = resp.json()
        if data:
            latest = data[-1]
            results.append({
                'code': code,
                'exchange': exchange,
                'in_universe': '✅ Yes' if in_universe else '❌ No',
                'api_status': '✅ OK',
                'records': len(data),
                'latest_date': latest['date'],
                'latest_close': latest['close']
            })
            print(f"✅ {code}.{exchange}: {len(data)} records, latest={latest['date']}, close={latest['close']}, in_universe={in_universe}")
        else:
            results.append({
                'code': code,
                'exchange': exchange,
                'in_universe': '✅ Yes' if in_universe else '❌ No',
                'api_status': '⚠️ Empty',
                'records': 0,
                'latest_date': 'N/A',
                'latest_close': 'N/A'
            })
            print(f"⚠️ {code}.{exchange}: No data returned, in_universe={in_universe}")
    else:
        results.append({
            'code': code,
            'exchange': exchange,
            'in_universe': '✅ Yes' if in_universe else '❌ No',
            'api_status': f'❌ Error {resp.status_code}',
            'records': 0,
            'latest_date': 'N/A',
            'latest_close': 'N/A'
        })
        print(f"❌ {code}.{exchange}: Error {resp.status_code}, in_universe={in_universe}")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Check current local universe
print()
print("=" * 80)
print("Checking current local universe.parquet")
print("=" * 80)
try:
    universe = pd.read_parquet('data/universe.parquet')
    for code in stocks:
        exists = code in universe['symbol'].values
        print(f"  {code}: {'✅ In local universe' if exists else '❌ NOT in local universe (needs refresh)'}")
except Exception as e:
    print(f"  Could not read universe.parquet: {e}")
