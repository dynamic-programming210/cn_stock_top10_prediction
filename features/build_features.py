"""
Feature Engineering for Chinese Stock Top-10 Predictor
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BARS_FILE, FEATURES_FILE, FEATURES_Z_FILE,
    FEATURE_COLS, TARGET_COL, RETURN_HORIZONS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_returns(df: pd.DataFrame, horizons: List[int] = RETURN_HORIZONS) -> pd.DataFrame:
    """Compute returns over multiple horizons"""
    df = df.copy()
    
    for h in horizons:
        df[f'ret_{h}'] = df.groupby('symbol')['close'].pct_change(h)
    
    return df


def compute_forward_returns(df: pd.DataFrame, horizons: List[int] = [5]) -> pd.DataFrame:
    """Compute forward returns (targets)"""
    df = df.copy()
    
    for h in horizons:
        df[f'fwd_ret_{h}'] = df.groupby('symbol')['close'].pct_change(h).shift(-h)
    
    return df


def compute_volatility(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Compute rolling volatility"""
    df = df.copy()
    
    # First compute daily returns if not present
    if 'ret_1' not in df.columns:
        df['ret_1'] = df.groupby('symbol')['close'].pct_change(1)
    
    for w in windows:
        df[f'vol_{w}'] = df.groupby('symbol')['ret_1'].transform(
            lambda x: x.rolling(w, min_periods=w//2).std() * np.sqrt(252)
        )
    
    return df


def compute_volume_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Compute volume-based features"""
    df = df.copy()
    
    for w in windows:
        # Volume ratio vs moving average
        ma = df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(w, min_periods=w//2).mean()
        )
        df[f'volume_ratio_{w}'] = df['volume'] / (ma + 1)
        
        # Dollar volume
        if w <= 10:
            df[f'dollar_volume_{w}'] = df.groupby('symbol').apply(
                lambda x: (x['close'] * x['volume']).rolling(w, min_periods=w//2).mean()
            ).reset_index(level=0, drop=True)
    
    return df


def compute_price_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price momentum features"""
    df = df.copy()
    
    # Price vs moving averages
    for w in [5, 10, 20, 50]:
        ma = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(w, min_periods=w//2).mean()
        )
        df[f'price_vs_ma{w}'] = df['close'] / (ma + 0.01) - 1
    
    # MA crossovers
    ma5 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    ma10 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(10).mean())
    ma20 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
    ma50 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(50).mean())
    
    df['ma5_vs_ma20'] = ma5 / (ma20 + 0.01) - 1
    df['ma10_vs_ma50'] = ma10 / (ma50 + 0.01) - 1
    
    return df


def compute_high_low_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute high/low range features"""
    df = df.copy()
    
    # Distance from 20-day high/low
    high_20 = df.groupby('symbol')['high'].transform(lambda x: x.rolling(20).max())
    low_20 = df.groupby('symbol')['low'].transform(lambda x: x.rolling(20).min())
    
    df['high_20d_dist'] = (df['close'] - high_20) / (high_20 + 0.01)
    df['low_20d_dist'] = (df['close'] - low_20) / (low_20 + 0.01)
    
    # High-low range
    for w in [5, 10]:
        high_w = df.groupby('symbol')['high'].transform(lambda x: x.rolling(w).max())
        low_w = df.groupby('symbol')['low'].transform(lambda x: x.rolling(w).min())
        df[f'hl_range_{w}'] = (high_w - low_w) / (low_w + 0.01)
    
    return df


def compute_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candlestick pattern features"""
    df = df.copy()
    
    # Body ratio (close-open relative to high-low)
    hl_range = df['high'] - df['low']
    body = df['close'] - df['open']
    
    df['body_ratio'] = body / (hl_range + 0.01)
    
    # Upper shadow
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (hl_range + 0.01)
    
    # Lower shadow
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (hl_range + 0.01)
    
    # Gap open (vs previous close)
    prev_close = df.groupby('symbol')['close'].shift(1)
    df['gap_open'] = (df['open'] - prev_close) / (prev_close + 0.01)
    
    # Intraday return
    df['intraday_ret'] = (df['close'] - df['open']) / (df['open'] + 0.01)
    
    return df


def compute_china_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute China A-share specific features
    
    Key characteristics of China A-shares:
    - 10% daily price limit (涨跌停)
    - T+1 settlement (cannot sell same day)
    - Retail-dominated market with momentum effects
    - Limit-up/down stocks often continue next day
    """
    df = df.copy()
    
    # Daily return for limit detection
    df['daily_ret'] = df.groupby('symbol')['close'].pct_change(1)
    prev_close = df.groupby('symbol')['close'].shift(1)
    
    # Price limit detection (10% limit, with some tolerance for rounding)
    df['near_limit_up'] = (df['daily_ret'] >= 0.095).astype(int)  # Near 涨停
    df['near_limit_down'] = (df['daily_ret'] <= -0.095).astype(int)  # Near 跌停
    df['at_limit_up'] = (df['daily_ret'] >= 0.099).astype(int)  # At 涨停 
    df['at_limit_down'] = (df['daily_ret'] <= -0.099).astype(int)  # At 跌停
    
    # Count of limit-up/down days in past N days (momentum/reversal signal)
    for w in [5, 10, 20]:
        df[f'limit_up_count_{w}'] = df.groupby('symbol')['at_limit_up'].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        )
        df[f'limit_down_count_{w}'] = df.groupby('symbol')['at_limit_down'].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        )
    
    # Consecutive limit-up/down (strong momentum)
    df['consec_limit_up'] = df.groupby('symbol')['at_limit_up'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    df['consec_limit_down'] = df.groupby('symbol')['at_limit_down'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    
    # Days since last limit-up/down (mean reversion signal)
    df['days_since_limit_up'] = df.groupby('symbol')['at_limit_up'].transform(
        lambda x: (~x.astype(bool)).cumsum() - (~x.astype(bool)).cumsum().where(x.astype(bool)).ffill().fillna(0)
    ).clip(upper=60)  # Cap at 60 days
    
    df['days_since_limit_down'] = df.groupby('symbol')['at_limit_down'].transform(
        lambda x: (~x.astype(bool)).cumsum() - (~x.astype(bool)).cumsum().where(x.astype(bool)).ffill().fillna(0)
    ).clip(upper=60)
    
    # Amplitude (振幅) - intraday high-low range as % of prev close
    df['amplitude'] = (df['high'] - df['low']) / (prev_close + 0.01)
    df['amplitude_5d_avg'] = df.groupby('symbol')['amplitude'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # Turnover rate proxy (volume relative to recent average - retail activity)
    df['turnover_surge'] = df['volume'] / (df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    ) + 1)
    df['turnover_surge'] = df['turnover_surge'].clip(upper=10)  # Cap outliers
    
    # Price position in daily range (close relative to high-low)
    hl_range = df['high'] - df['low']
    df['close_position'] = (df['close'] - df['low']) / (hl_range + 0.01)
    
    # Opening gap pattern (retail often chases gaps)
    df['gap_up'] = ((df['open'] - prev_close) / (prev_close + 0.01) > 0.02).astype(int)
    df['gap_down'] = ((df['open'] - prev_close) / (prev_close + 0.01) < -0.02).astype(int)
    
    # Momentum reversal signals
    df['strong_up_day'] = (df['daily_ret'] > 0.05).astype(int)
    df['strong_down_day'] = (df['daily_ret'] < -0.05).astype(int)
    
    # 5-day pattern: count of up days
    df['up_days_5'] = df.groupby('symbol')['daily_ret'].transform(
        lambda x: (x > 0).rolling(5, min_periods=1).sum()
    )
    
    # Clean up intermediate column
    df = df.drop(columns=['daily_ret'], errors='ignore')
    
    return df


def compute_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional rank features"""
    df = df.copy()
    
    rank_cols = ['ret_5', 'vol_5', 'volume_ratio_5', 'amplitude', 'turnover_surge']
    for col in rank_cols:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)
    
    return df


def zscore_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Z-score features cross-sectionally by date.
    
    Note: Binary columns (0/1 indicators) should be excluded from z-scoring.
    """
    df = df.copy()
    
    # Columns to exclude from z-scoring (binary flags)
    binary_cols = {
        'at_limit_up', 'at_limit_down', 
        'near_limit_up', 'near_limit_down',
        # Any other binary columns can be added here
    }
    
    for col in feature_cols:
        if col in df.columns and col not in binary_cols:
            df[col] = df.groupby('date')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    
    return df


def build_features(bars: pd.DataFrame = None, 
                   save: bool = True,
                   zscore: bool = True) -> pd.DataFrame:
    """Main function to build all features"""
    
    if bars is None:
        logger.info(f"Loading bars from {BARS_FILE}")
        bars = pd.read_parquet(BARS_FILE)
    
    logger.info(f"Building features for {bars['symbol'].nunique()} symbols")
    
    # Sort by symbol and date
    bars = bars.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Compute all features
    logger.info("Computing returns...")
    df = compute_returns(bars)
    
    logger.info("Computing forward returns (targets)...")
    df = compute_forward_returns(df)
    
    logger.info("Computing volatility...")
    df = compute_volatility(df)
    
    logger.info("Computing volume features...")
    df = compute_volume_features(df)
    
    logger.info("Computing price momentum...")
    df = compute_price_momentum(df)
    
    logger.info("Computing high/low features...")
    df = compute_high_low_features(df)
    
    logger.info("Computing candlestick features...")
    df = compute_candlestick_features(df)
    
    logger.info("Computing China-specific features...")
    df = compute_china_specific_features(df)
    
    logger.info("Computing rank features...")
    df = compute_rank_features(df)
    
    # Save raw features
    if save:
        df.to_parquet(FEATURES_FILE, index=False)
        logger.info(f"Saved raw features to {FEATURES_FILE}")
    
    # Z-score features
    if zscore:
        logger.info("Z-scoring features...")
        available_cols = [c for c in FEATURE_COLS if c in df.columns]
        df_z = zscore_features(df, available_cols)
        
        if save:
            df_z.to_parquet(FEATURES_Z_FILE, index=False)
            logger.info(f"Saved z-scored features to {FEATURES_Z_FILE}")
        
        return df_z
    
    return df


def get_training_data(df: pd.DataFrame = None,
                      min_date: str = None,
                      max_date: str = None) -> pd.DataFrame:
    """Get clean training data with no NaNs in features"""
    
    if df is None:
        df = pd.read_parquet(FEATURES_Z_FILE)
    
    # Filter by date if specified
    if min_date:
        df = df[df['date'] >= min_date]
    if max_date:
        df = df[df['date'] <= max_date]
    
    # Get available feature columns
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    required_cols = ['symbol', 'date', TARGET_COL] + available_cols
    
    # Drop rows with NaN in features or target
    clean = df[required_cols].dropna()
    
    logger.info(f"Training data: {len(clean)} rows, {clean['symbol'].nunique()} symbols")
    
    return clean


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build features for CN stocks")
    parser.add_argument("--no-zscore", action="store_true", help="Skip z-scoring")
    
    args = parser.parse_args()
    
    df = build_features(zscore=not args.no_zscore)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   - Rows: {len(df)}")
    print(f"   - Symbols: {df['symbol'].nunique()}")
    print(f"   - Features: {len([c for c in FEATURE_COLS if c in df.columns])}")
