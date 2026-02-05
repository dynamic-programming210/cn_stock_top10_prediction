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


def compute_strong_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute "Strong Stock" (强势股) features based on institutional trading patterns.
    
    These features capture:
    A. Capital Flow & Sentiment (institutional intent)
    B. Technical Pattern Confirmation  
    C. Trend Quality & Momentum
    D. Volume-Price Health
    E. Market Behavior & Relative Strength
    """
    df = df.copy()
    
    # ===== A. CAPITAL FLOW & SENTIMENT =====
    
    # A1. Volume expansion (倍量): 2x+ recent average
    vol_20_avg = df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    df['volume_expansion'] = df['volume'] / (vol_20_avg + 1)
    df['is_volume_2x'] = (df['volume_expansion'] >= 2.0).astype(int)
    df['is_volume_3x'] = (df['volume_expansion'] >= 3.0).astype(int)
    
    # Sustained volume expansion: 2x+ for multiple days
    df['volume_2x_count_5'] = df.groupby('symbol')['is_volume_2x'].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    
    # A2. Limit-up with volume (涨停放量): strong signal
    df['limit_up_with_volume'] = (
        (df['at_limit_up'] == 1) & (df['volume_expansion'] >= 1.5)
    ).astype(int)
    
    # A3. Upward gap (向上跳空缺口): gap that persists
    prev_high = df.groupby('symbol')['high'].shift(1)
    df['gap_up_size'] = (df['low'] - prev_high) / (prev_high + 0.01)
    df['strong_gap_up'] = (df['gap_up_size'] > 0.02).astype(int)  # >2% gap
    
    # Gap-up count in recent days
    df['gap_up_count_10'] = df.groupby('symbol')['strong_gap_up'].transform(
        lambda x: x.rolling(10, min_periods=1).sum()
    )
    
    # ===== B. TECHNICAL PATTERN CONFIRMATION =====
    
    # B1. Consecutive bullish candles (连阳)
    df['is_bullish_day'] = (df['close'] > df['open']).astype(int)
    
    # Count consecutive bullish days
    df['consec_bullish'] = df.groupby('symbol')['is_bullish_day'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    
    # Bullish days in last 5
    df['bullish_days_5'] = df.groupby('symbol')['is_bullish_day'].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    
    # B2. Higher highs pattern (创新高)
    high_5 = df.groupby('symbol')['high'].transform(lambda x: x.rolling(5).max())
    high_10 = df.groupby('symbol')['high'].transform(lambda x: x.rolling(10).max())
    high_20 = df.groupby('symbol')['high'].transform(lambda x: x.rolling(20).max())
    
    df['is_new_high_5'] = (df['high'] >= high_5.shift(1)).astype(int)
    df['is_new_high_10'] = (df['high'] >= high_10.shift(1)).astype(int)
    df['is_new_high_20'] = (df['high'] >= high_20.shift(1)).astype(int)
    
    # Count of new highs in last 10 days
    df['new_high_count_10'] = df.groupby('symbol')['is_new_high_5'].transform(
        lambda x: x.rolling(10, min_periods=1).sum()
    )
    
    # ===== C. TREND QUALITY & MA ALIGNMENT (均线多头排列) =====
    
    # Moving averages
    ma5 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    ma10 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(10).mean())
    ma20 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
    ma60 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(60, min_periods=30).mean())
    
    # MA alignment score: 5 > 10 > 20 > 60 (bullish alignment)
    df['ma_bullish_align'] = (
        (ma5 > ma10).astype(int) + 
        (ma10 > ma20).astype(int) + 
        (ma20 > ma60).astype(int)
    )  # Score 0-3
    
    # Price above all short-term MAs
    df['above_all_ma'] = (
        (df['close'] > ma5) & 
        (df['close'] > ma10) & 
        (df['close'] > ma20)
    ).astype(int)
    
    # Price holding key MA (not breaking 10MA on pullback)
    df['holding_ma10'] = (df['low'] > ma10 * 0.98).astype(int)  # Within 2%
    
    # MA slope (trending up)
    ma5_slope = ma5 - ma5.shift(3)
    ma10_slope = ma10 - ma10.shift(5)
    df['ma5_rising'] = (ma5_slope > 0).astype(int)
    df['ma10_rising'] = (ma10_slope > 0).astype(int)
    
    # Combined trend score
    df['trend_score'] = (
        df['ma_bullish_align'] + 
        df['above_all_ma'] * 2 + 
        df['ma5_rising'] + 
        df['ma10_rising']
    )  # Score 0-7
    
    # ===== D. VOLUME-PRICE HEALTH =====
    
    # D1. Volume expands on up days, contracts on down days (涨放量、跌缩量)
    df['daily_ret_temp'] = df.groupby('symbol')['close'].pct_change(1)
    df['is_up_day'] = (df['daily_ret_temp'] > 0).astype(int)
    df['is_down_day'] = (df['daily_ret_temp'] < 0).astype(int)
    
    # Volume on up vs down days (rolling 10-day)
    df['vol_on_up'] = df.groupby('symbol').apply(
        lambda x: (x['volume'] * x['is_up_day']).rolling(10, min_periods=3).sum() / 
                  (x['is_up_day'].rolling(10, min_periods=3).sum() + 0.01)
    ).reset_index(level=0, drop=True)
    
    df['vol_on_down'] = df.groupby('symbol').apply(
        lambda x: (x['volume'] * x['is_down_day']).rolling(10, min_periods=3).sum() / 
                  (x['is_down_day'].rolling(10, min_periods=3).sum() + 0.01)
    ).reset_index(level=0, drop=True)
    
    # Volume-price health: up-volume > down-volume is healthy
    df['vol_price_health'] = df['vol_on_up'] / (df['vol_on_down'] + 1)
    df['vol_price_health'] = df['vol_price_health'].clip(0.1, 10)  # Bound outliers
    
    # D2. Price action quality: close near high of day (buying pressure)
    df['close_near_high'] = (
        (df['close'] - df['low']) / (df['high'] - df['low'] + 0.01)
    )  # 1 = closed at high, 0 = closed at low
    
    # Average close position over 5 days
    df['close_position_5d'] = df.groupby('symbol')['close_near_high'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # ===== E. MARKET BEHAVIOR & RELATIVE STRENGTH =====
    
    # E1. Market-relative performance (强于大盘)
    # Compute market average return each day
    df['market_ret_1'] = df.groupby('date')['daily_ret_temp'].transform('mean')
    df['market_ret_5'] = df.groupby('date')['ret_5'].transform('mean')
    
    # Stock vs market
    df['vs_market_1d'] = df['daily_ret_temp'] - df['market_ret_1']
    df['vs_market_5d'] = df['ret_5'] - df['market_ret_5']
    
    # Outperformance count: days beating market in last 10
    df['beat_market'] = (df['vs_market_1d'] > 0).astype(int)
    df['beat_market_count_10'] = df.groupby('symbol')['beat_market'].transform(
        lambda x: x.rolling(10, min_periods=1).sum()
    )
    
    # E2. Resilience: smaller drawdown during market pullback (抗跌)
    # Rolling max price and drawdown
    rolling_max = df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(20, min_periods=5).max()
    )
    df['drawdown_20d'] = (df['close'] - rolling_max) / (rolling_max + 0.01)
    
    # Market average drawdown
    df['market_drawdown'] = df.groupby('date')['drawdown_20d'].transform('mean')
    
    # Resilience score: less drawdown than market = more resilient
    df['resilience'] = df['drawdown_20d'] - df['market_drawdown']  # Higher = more resilient
    
    # ===== F. COMPOSITE STRONG STOCK SCORE =====
    
    # Combine key signals into a single score
    df['strong_stock_score'] = (
        # Capital flow signals (weight: 3)
        df['is_volume_2x'] * 1.0 +
        df['limit_up_count_10'].clip(0, 3) * 0.5 +
        df['strong_gap_up'] * 1.5 +
        
        # Technical patterns (weight: 3)
        df['consec_bullish'].clip(0, 5) * 0.5 +
        df['is_new_high_20'] * 1.5 +
        
        # Trend quality (weight: 4)
        df['trend_score'] * 0.5 +
        
        # Volume-price health (weight: 2)
        (df['vol_price_health'] > 1.2).astype(int) * 1.0 +
        df['close_position_5d'] * 1.0 +
        
        # Market outperformance (weight: 3)
        df['beat_market_count_10'] * 0.2 +
        (df['resilience'] > 0).astype(int) * 1.5
    )
    
    # Clean up temp columns
    df = df.drop(columns=['daily_ret_temp', 'is_up_day', 'is_down_day', 
                          'market_ret_1', 'market_ret_5', 'beat_market',
                          'is_bullish_day', 'close_near_high'], errors='ignore')
    
    return df


def compute_sector_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sector-relative features to compare stocks within their sector.
    This prevents sectors like banking from dominating due to their unique characteristics.
    """
    df = df.copy()
    
    # Import sector classification
    try:
        from data.sectors import get_stock_sector, add_sector_to_dataframe
        
        # Add sector column if not present
        if 'sector' not in df.columns:
            logger.info("Adding sector classification...")
            # Get unique symbols and their sectors
            unique_symbols = df[['symbol']].drop_duplicates()
            unique_symbols['sector'] = unique_symbols['symbol'].apply(
                lambda x: get_stock_sector(str(x))
            )
            df = df.merge(unique_symbols[['symbol', 'sector']], on='symbol', how='left')
        
        # Features to compute sector-relative versions for
        sector_relative_features = [
            'ret_5', 'ret_10', 'ret_20',
            'vol_5', 'vol_10', 
            'volume_ratio_5', 'volume_ratio_10',
            'dollar_volume_5', 'dollar_volume_10',
            'amplitude', 'turnover_surge',
            'price_vs_ma5', 'price_vs_ma20',
        ]
        
        logger.info(f"Computing sector-relative features for {df['sector'].nunique()} sectors...")
        
        for col in sector_relative_features:
            if col in df.columns:
                # Sector-relative z-score: compare within sector on same day
                df[f'{col}_sector_rel'] = df.groupby(['date', 'sector'])[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
        
        # Sector momentum: how is this sector performing vs market?
        df['sector_ret_5'] = df.groupby(['date', 'sector'])['ret_5'].transform('mean')
        df['vs_sector_ret_5'] = df['ret_5'] - df['sector_ret_5']
        
        # Sector volume: is this sector getting more attention?
        df['sector_volume_ratio'] = df.groupby(['date', 'sector'])['volume_ratio_5'].transform('mean')
        df['vs_sector_volume'] = df['volume_ratio_5'] - df['sector_volume_ratio']
        
        # Rank within sector
        df['sector_ret_rank'] = df.groupby(['date', 'sector'])['ret_5'].rank(pct=True)
        df['sector_vol_rank'] = df.groupby(['date', 'sector'])['volume_ratio_5'].rank(pct=True)
        
        logger.info(f"Added {len(sector_relative_features)} sector-relative features")
        
    except ImportError as e:
        logger.warning(f"Could not import sector module: {e}. Skipping sector features.")
    except Exception as e:
        logger.warning(f"Error computing sector features: {e}. Skipping.")
    
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
    
    logger.info("Computing strong stock features (强势股)...")
    df = compute_strong_stock_features(df)
    
    logger.info("Computing rank features...")
    df = compute_rank_features(df)
    
    # NEW: Compute sector-relative features
    logger.info("Computing sector-relative features...")
    df = compute_sector_relative_features(df)
    
    # NEW: Add news sentiment features (v2.2)
    try:
        logger.info("Adding news sentiment features...")
        from data.fetch_news import get_news_features
        df = get_news_features(df)
        news_cols = [c for c in df.columns if c.startswith('news_')]
        logger.info(f"Added {len(news_cols)} news sentiment features")
    except Exception as e:
        logger.warning(f"Could not add news features: {e}")
        logger.warning("Continuing without news features...")
    
    # Save raw features
    if save:
        df.to_parquet(FEATURES_FILE, index=False)
        logger.info(f"Saved raw features to {FEATURES_FILE}")
    
    # Z-score features
    if zscore:
        logger.info("Z-scoring features...")
        available_cols = [c for c in FEATURE_COLS if c in df.columns]
        # Also include sector-relative features in z-scoring
        sector_rel_cols = [c for c in df.columns if '_sector_rel' in c or c.startswith('vs_sector') or c.startswith('sector_')]
        all_zscore_cols = list(set(available_cols + sector_rel_cols))
        df_z = zscore_features(df, all_zscore_cols)
        
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
