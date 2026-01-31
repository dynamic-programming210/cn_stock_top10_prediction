"""
EODHD API Client for Chinese Stocks
Fetches data from Shanghai (SHG) and Shenzhen (SHE) Stock Exchanges
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EODHD_API_TOKEN, EODHD_BASE_URL, EXCHANGE_CODES,
    MIN_PRICE, MIN_AVG_VOLUME, TARGET_UNIVERSE_SIZE, LOOKBACK_DAYS,
    BARS_FILE, UNIVERSE_FILE, UNIVERSE_META_FILE, FUNDAMENTALS_FILE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EODHDClient:
    """Client for EODHD API"""
    
    def __init__(self, api_token: str = None):
        self.api_token = api_token or EODHD_API_TOKEN
        self.base_url = EODHD_BASE_URL
        self.session = requests.Session()
        
    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with error handling"""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params['api_token'] = self.api_token
        params['fmt'] = 'json'
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_exchange_symbols(self, exchange: str, stock_type: str = 'common_stock') -> pd.DataFrame:
        """Get list of symbols for an exchange"""
        logger.info(f"Fetching symbols for {exchange}...")
        
        data = self._request(f"exchange-symbol-list/{exchange}", {'type': stock_type})
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        # Use our exchange code (SHG/SHE) - drop the API's Exchange column if it exists
        if 'Exchange' in df.columns:
            df = df.drop(columns=['Exchange'])
        df['exchange'] = exchange
        
        logger.info(f"Found {len(df)} symbols for {exchange}")
        return df
    
    def get_eod_data(self, symbol: str, exchange: str, 
                     from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """Get end-of-day historical data for a symbol"""
        ticker = f"{symbol}.{exchange}"
        
        params = {'period': 'd'}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        try:
            data = self._request(f"eod/{ticker}", params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['symbol'] = symbol
            df['exchange'] = exchange
            df['date'] = pd.to_datetime(df['date'])
            
            # Rename columns to standard format
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'adjusted_close': 'adj_close',
                'volume': 'volume'
            })
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_fundamentals(self, symbol: str, exchange: str) -> dict:
        """Get fundamental data for a symbol"""
        ticker = f"{symbol}.{exchange}"
        
        try:
            data = self._request(f"fundamentals/{ticker}")
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
            return {}
    
    def get_bulk_eod(self, exchange: str, date: str = None) -> pd.DataFrame:
        """Get bulk EOD data for entire exchange (single day)"""
        params = {}
        if date:
            params['date'] = date
        
        try:
            data = self._request(f"eod-bulk-last-day/{exchange}", params)
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['exchange'] = exchange
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to fetch bulk EOD for {exchange}: {e}")
            return pd.DataFrame()


def fetch_universe(client: EODHDClient = None) -> pd.DataFrame:
    """Fetch and filter stock universe from Chinese exchanges"""
    if client is None:
        client = EODHDClient()
    
    all_symbols = []
    
    for exchange in EXCHANGE_CODES:
        df = client.get_exchange_symbols(exchange)
        if not df.empty:
            all_symbols.append(df)
    
    if not all_symbols:
        raise ValueError("No symbols fetched from any exchange")
    
    universe = pd.concat(all_symbols, ignore_index=True)
    
    # Clean up the data
    universe = universe.rename(columns={
        'Code': 'symbol',
        'Name': 'name',
        'Exchange': 'exchange',
        'Currency': 'currency',
        'Type': 'type',
        'Country': 'country',
        'Isin': 'isin'
    })
    
    # Filter to common stocks only
    universe = universe[universe['type'] == 'Common Stock'].copy()
    
    # Filter out special boards (科创板 688xxx, 创业板 300xxx, 北交所 etc.)
    # Keep main board stocks for now: 
    # Shanghai: 600xxx, 601xxx, 603xxx, 605xxx
    # Shenzhen: 000xxx, 001xxx, 002xxx
    def is_main_board(symbol, exchange):
        code = str(symbol)
        if exchange == 'SHG':
            return code.startswith(('600', '601', '603', '605'))
        elif exchange == 'SHE':
            return code.startswith(('000', '001', '002'))
        return False
    
    mask = universe.apply(lambda row: is_main_board(row['symbol'], row['exchange']), axis=1)
    universe = universe[mask].copy()
    
    logger.info(f"Universe after filtering: {len(universe)} stocks")
    
    return universe


def fetch_historical_bars(universe: pd.DataFrame, 
                          client: EODHDClient = None,
                          lookback_days: int = LOOKBACK_DAYS,
                          batch_size: int = 50,
                          sleep_time: float = 0.2) -> pd.DataFrame:
    """Fetch historical OHLCV data for all symbols in universe"""
    if client is None:
        client = EODHDClient()
    
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    
    all_bars = []
    total = len(universe)
    
    logger.info(f"Fetching historical data for {total} symbols from {from_date} to {to_date}")
    
    for i, (_, row) in enumerate(universe.iterrows()):
        symbol = row['symbol']
        exchange = row['exchange']
        
        df = client.get_eod_data(symbol, exchange, from_date, to_date)
        
        if not df.empty:
            all_bars.append(df)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{total} symbols fetched")
        
        # Rate limiting
        if (i + 1) % batch_size == 0:
            time.sleep(sleep_time)
    
    if not all_bars:
        raise ValueError("No historical data fetched")
    
    bars = pd.concat(all_bars, ignore_index=True)
    bars = bars.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    logger.info(f"Fetched {len(bars)} total bar records for {bars['symbol'].nunique()} symbols")
    
    return bars


def filter_liquid_stocks(bars: pd.DataFrame, 
                         min_price: float = MIN_PRICE,
                         min_avg_volume: float = MIN_AVG_VOLUME,
                         min_days: int = 60) -> pd.DataFrame:
    """Filter stocks based on liquidity criteria"""
    
    # Calculate recent statistics (last 60 days)
    recent = bars.groupby('symbol').tail(min_days)
    
    stats = recent.groupby('symbol').agg({
        'close': 'mean',
        'volume': 'mean',
        'date': 'count'
    }).rename(columns={'close': 'avg_price', 'volume': 'avg_volume', 'date': 'trading_days'})
    
    # Apply filters
    liquid = stats[
        (stats['avg_price'] >= min_price) &
        (stats['avg_volume'] >= min_avg_volume) &
        (stats['trading_days'] >= min_days * 0.8)  # At least 80% trading days
    ]
    
    logger.info(f"Liquid stocks after filtering: {len(liquid)}")
    
    return bars[bars['symbol'].isin(liquid.index)].copy()


def update_data(full_refresh: bool = False, 
                max_symbols: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main function to update all data"""
    client = EODHDClient()
    
    # Step 1: Fetch universe
    logger.info("Step 1: Fetching universe...")
    universe = fetch_universe(client)
    
    if max_symbols:
        universe = universe.head(max_symbols)
        logger.info(f"Limited to {max_symbols} symbols for testing")
    
    # Save universe metadata
    universe.to_parquet(UNIVERSE_META_FILE, index=False)
    logger.info(f"Saved universe metadata to {UNIVERSE_META_FILE}")
    
    # Step 2: Fetch historical bars
    logger.info("Step 2: Fetching historical bars...")
    
    if not full_refresh and BARS_FILE.exists():
        # Incremental update
        existing = pd.read_parquet(BARS_FILE)
        last_date = existing['date'].max()
        
        from_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        if from_date < to_date:
            logger.info(f"Incremental update from {from_date} to {to_date}")
            # Fetch only new data
            new_bars = []
            for _, row in universe.iterrows():
                df = client.get_eod_data(row['symbol'], row['exchange'], from_date, to_date)
                if not df.empty:
                    new_bars.append(df)
                time.sleep(0.1)
            
            if new_bars:
                new_df = pd.concat(new_bars, ignore_index=True)
                bars = pd.concat([existing, new_df], ignore_index=True)
                bars = bars.drop_duplicates(subset=['symbol', 'date'], keep='last')
                bars = bars.sort_values(['symbol', 'date']).reset_index(drop=True)
            else:
                bars = existing
        else:
            logger.info("No new data to fetch")
            bars = existing
    else:
        # Full refresh
        bars = fetch_historical_bars(universe, client)
    
    # Step 3: Filter liquid stocks
    logger.info("Step 3: Filtering liquid stocks...")
    bars = filter_liquid_stocks(bars)
    
    # Step 4: Save data
    bars.to_parquet(BARS_FILE, index=False)
    logger.info(f"Saved {len(bars)} bars to {BARS_FILE}")
    
    # Update universe to only include liquid stocks
    liquid_symbols = bars['symbol'].unique()
    universe = universe[universe['symbol'].isin(liquid_symbols)].copy()
    universe.to_parquet(UNIVERSE_FILE, index=False)
    logger.info(f"Saved {len(universe)} symbols to {UNIVERSE_FILE}")
    
    return bars, universe


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Chinese stock data from EODHD")
    parser.add_argument("--full-refresh", action="store_true", help="Full data refresh")
    parser.add_argument("--max-symbols", type=int, help="Limit symbols for testing")
    parser.add_argument("--test", action="store_true", help="Test mode with 10 symbols")
    
    args = parser.parse_args()
    
    if args.test:
        args.max_symbols = 10
        args.full_refresh = True
    
    bars, universe = update_data(
        full_refresh=args.full_refresh,
        max_symbols=args.max_symbols
    )
    
    print(f"\n✅ Data update complete!")
    print(f"   - Symbols: {len(universe)}")
    print(f"   - Total bars: {len(bars)}")
    print(f"   - Date range: {bars['date'].min()} to {bars['date'].max()}")
