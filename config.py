"""
Configuration for Chinese Stock Top-10 Predictor
Supports Shanghai (SHG) and Shenzhen (SHE) Stock Exchanges
"""
from pathlib import Path
import os

# ============ Paths ============
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Data files
BARS_FILE = DATA_DIR / "bars.parquet"
UNIVERSE_FILE = DATA_DIR / "universe.parquet"
UNIVERSE_META_FILE = DATA_DIR / "universe_meta.parquet"
FEATURES_FILE = DATA_DIR / "features.parquet"
FEATURES_Z_FILE = DATA_DIR / "feat_z.parquet"
NEWS_SENTIMENT_FILE = DATA_DIR / "news_sentiment.parquet"
FUNDAMENTALS_FILE = DATA_DIR / "fundamentals.parquet"

# Output files
TOP10_LATEST_FILE = OUTPUTS_DIR / "top10_latest.parquet"
TOP10_HISTORY_FILE = OUTPUTS_DIR / "top10_history.parquet"
QUALITY_REPORT_FILE = OUTPUTS_DIR / "quality_report.json"

# Model files
RANKER_MODEL_FILE = MODELS_DIR / "ranker.txt"
REGRESSOR_MODEL_FILE = MODELS_DIR / "regressor.pkl"

# ============ API Configuration ============
EODHD_API_TOKEN = os.environ.get("EODHD_API_TOKEN", "697c670dd1a9d9.07390782")
EODHD_BASE_URL = "https://eodhd.com/api"

# ============ Chinese Exchanges ============
EXCHANGES = {
    "SHG": "Shanghai Stock Exchange",
    "SHE": "Shenzhen Stock Exchange"
}

# Exchange codes for API calls
EXCHANGE_CODES = ["SHG", "SHE"]

# ============ Universe Settings ============
# Filter settings for stock selection
MIN_PRICE = 2.0  # Minimum stock price (CNY)
MIN_AVG_VOLUME = 1_000_000  # Minimum average daily volume
MIN_MARKET_CAP = 5_000_000_000  # Minimum market cap (5B CNY) - approx large/mid cap

# Universe size target (top stocks by market cap from each exchange)
TARGET_UNIVERSE_SIZE = 500  # Top 500 stocks combined

# ============ Data Settings ============
LOOKBACK_DAYS = 365 * 3  # 3 years of historical data
MIN_HISTORY_DAYS = 252  # Minimum 1 year of trading history required

# ============ Feature Engineering ============
# Return horizons (days)
RETURN_HORIZONS = [1, 3, 5, 10, 20]

# Feature columns used in model
FEATURE_COLS = [
    # Returns
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
    # Volatility
    'vol_5', 'vol_10', 'vol_20',
    # Volume signals
    'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
    'dollar_volume_5', 'dollar_volume_10',
    # Price momentum
    'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20', 'price_vs_ma50',
    'ma5_vs_ma20', 'ma10_vs_ma50',
    # High/Low signals
    'high_20d_dist', 'low_20d_dist',
    'hl_range_5', 'hl_range_10',
    # Candlestick patterns
    'body_ratio', 'upper_shadow', 'lower_shadow',
    'gap_open', 'intraday_ret',
    # China-specific features (涨跌停, T+1 effects)
    'near_limit_up', 'near_limit_down',
    'at_limit_up', 'at_limit_down',
    'limit_up_count_5', 'limit_up_count_10',
    'limit_down_count_5', 'limit_down_count_10',
    'consec_limit_up', 'consec_limit_down',
    'days_since_limit_up', 'days_since_limit_down',
    'amplitude', 'amplitude_5d_avg',
    'turnover_surge', 'close_position',
    'gap_up', 'gap_down',
    'strong_up_day', 'strong_down_day',
    'up_days_5',
    # Cross-sectional rank features
    'ret_5_rank', 'vol_5_rank', 'volume_ratio_5_rank',
    'amplitude_rank', 'turnover_surge_rank',
    # Sector-relative features (compare within sector, not vs all stocks)
    'ret_5_sector_rel', 'ret_10_sector_rel', 
    'vol_5_sector_rel', 'volume_ratio_5_sector_rel',
    'amplitude_sector_rel', 'turnover_surge_sector_rel',
    'vs_sector_ret_5', 'vs_sector_volume',
    'sector_ret_rank', 'sector_vol_rank',
    
    # ===== STRONG STOCK (强势股) FEATURES =====
    # Capital flow & sentiment signals
    'volume_expansion', 'is_volume_2x', 'is_volume_3x',
    'volume_2x_count_5', 'limit_up_with_volume',
    'gap_up_size', 'strong_gap_up', 'gap_up_count_10',
    
    # Technical pattern confirmation
    'consec_bullish', 'bullish_days_5',
    'is_new_high_5', 'is_new_high_10', 'is_new_high_20',
    'new_high_count_10',
    
    # Trend quality & MA alignment (均线多头排列)
    'ma_bullish_align', 'above_all_ma', 'holding_ma10',
    'ma5_rising', 'ma10_rising', 'trend_score',
    
    # Volume-price health (涨放量、跌缩量)
    'vol_on_up', 'vol_on_down', 'vol_price_health',
    'close_position_5d',
    
    # Market behavior & relative strength (强于大盘)
    'vs_market_1d', 'vs_market_5d',
    'beat_market_count_10',
    'drawdown_20d', 'resilience',
    
    # Composite score
    'strong_stock_score',
    
    # ===== TREND INITIATION (趋势起始) FEATURES =====
    # Volume expansion after consolidation
    'vol_healthy_expansion', 'vol_breakout_signal', 'vol_trending_up',
    
    # Golden cross signals (bullish MA crossover)
    'golden_cross_5_20', 'golden_cross_10_60', 'golden_cross_5_10',
    'golden_cross_any', 'golden_cross_5d',
    'crossed_above_ma20', 'crossed_above_ma60',
    
    # Breakout signals
    'was_in_consolidation', 'breakout_20d', 'breakout_with_volume',
    'consolidation_breakout', 'near_52w_high', 'breakout_52w',
    'was_lower_highs', 'resistance_breakout',
    
    # Gap-up signals
    'has_gap_up', 'gap_up_pct', 'gap_held_5d', 'significant_gap_up',
    
    # Higher lows pattern
    'is_swing_low', 'higher_low', 'higher_low_count_20',
    'consecutive_higher_lows', 'recent_vs_older_low',
    
    # Candlestick reversal patterns
    'morning_star', 'bullish_engulfing', 'hammer', 'piercing_pattern',
    'reversal_pattern', 'reversal_count_10',
    
    # Composite trend initiation score
    'trend_initiation_score',
    
    # ===== NEWS SENTIMENT (新闻舆情) FEATURES =====
    # News activity
    'news_news_count',
    # Sentiment scores
    'news_sentiment_mean', 'news_sentiment_std',
    'news_sentiment_positive_ratio', 'news_sentiment_trend',
]

# Target column
TARGET_COL = 'fwd_ret_5'  # 5-day forward return

# ============ Sector Settings ============
# Maximum stocks per sector in top-10 (for diversification)
MAX_STOCKS_PER_SECTOR = 2

# ============ Model Settings ============
CURRENT_MODEL_VERSION = "cn-v2.2.0"  # Updated for news sentiment features

# High-weight features (trend initiation signals get boosted)
HIGH_WEIGHT_FEATURES = [
    # Trend initiation signals (highest priority)
    'trend_initiation_score', 'golden_cross_10_60', 'golden_cross_5_20',
    'breakout_with_volume', 'consolidation_breakout', 'vol_breakout_signal',
    'morning_star', 'bullish_engulfing', 'higher_low_count_20',
    'gap_held_5d', 'crossed_above_ma60',
    # News sentiment (can be early signals)
    'news_sentiment_mean', 'news_sentiment_trend',
    # Strong stock signals
    'strong_stock_score', 'trend_score', 'ma_bullish_align',
    'vol_price_health', 'resilience',
]

# LightGBM Ranker params
RANKER_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 200,
    'early_stopping_rounds': 20,
}

# Regressor params
REGRESSOR_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
}

# ============ Top-10 Selection ============
TOP_K = 10
MIN_RANK_SCORE = 0.3  # Minimum ranking score threshold

# ============ Transaction Costs (China A-shares) ============
# Stamp tax: 0.1% on sell only (set by government)
# Commission: ~0.025% each way (negotiable, using conservative estimate)
# Slippage: estimated market impact
STAMP_TAX = 0.001  # 0.1% on sell
COMMISSION_RATE = 0.00025  # 0.025% each way
SLIPPAGE = 0.001  # 0.1% estimated slippage
# Total round-trip cost: buy commission + sell commission + stamp tax + slippage
ROUND_TRIP_COST = COMMISSION_RATE * 2 + STAMP_TAX + SLIPPAGE * 2  # ~0.35%

# ============ Liquidity & Trading Filters ============
# Minimum daily dollar volume to ensure tradability
# For raw values: 50M CNY
MIN_DOLLAR_VOLUME = 50_000_000  # 50M CNY daily volume (for raw features)
# For z-scored features: use z-score threshold (roughly -0.35 corresponds to 50M CNY)
MIN_DOLLAR_VOLUME_ZSCORE = -0.35  # Use when working with z-scored features
# Limit-up threshold (10% for main board, 20% for STAR/ChiNext - using conservative 9.8%)
LIMIT_UP_THRESHOLD = 0.098  # Stocks at 9.8%+ gain can't be bought
LIMIT_DOWN_THRESHOLD = -0.098  # Stocks at 9.8%+ loss may be hard to sell

# ============ Trading Calendar ============
# Chinese market trading hours (for reference)
MARKET_OPEN = "09:30"
MARKET_CLOSE = "15:00"
LUNCH_BREAK_START = "11:30"
LUNCH_BREAK_END = "13:00"

# ============ Create directories ============
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
