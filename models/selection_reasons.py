"""
Generate Chinese reasons for stock selection
Based on technical indicators, recent performance, and news sentiment
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import NEWS_SENTIMENT_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_reason_cn(row: pd.Series) -> str:
    """
    Generate a Chinese reason for why this stock was selected
    
    Args:
        row: A row from the predictions DataFrame with all features
        
    Returns:
        Chinese string explaining the selection reason
    """
    reasons = []
    
    # 1. Strong momentum
    ret_5 = row.get('ret_5', 0)
    ret_10 = row.get('ret_10', 0)
    if ret_5 > 0.05:
        reasons.append(f"近5日涨幅{ret_5*100:.1f}%")
    elif ret_5 > 0.02:
        reasons.append(f"近5日上涨{ret_5*100:.1f}%")
    
    # 2. Volume surge
    volume_ratio = row.get('volume_ratio_5', 1)
    if volume_ratio > 2:
        reasons.append(f"成交量放大{volume_ratio:.1f}倍")
    elif volume_ratio > 1.5:
        reasons.append("放量上涨")
    
    # 3. Limit-up related
    limit_up_count = row.get('limit_up_count_5', 0)
    if limit_up_count >= 2:
        reasons.append(f"近5日{int(limit_up_count)}次涨停")
    elif limit_up_count == 1:
        reasons.append("近期涨停板")
    
    # 4. Breaking new high
    is_new_high_20 = row.get('is_new_high_20', 0)
    is_new_high_10 = row.get('is_new_high_10', 0)
    if is_new_high_20:
        reasons.append("创20日新高")
    elif is_new_high_10:
        reasons.append("创10日新高")
    
    # 5. Strong stock score
    strong_score = row.get('strong_stock_score', 0)
    if strong_score >= 0.7:
        reasons.append("强势股特征明显")
    elif strong_score >= 0.5:
        reasons.append("走势强于大盘")
    
    # 6. MA alignment
    ma_bullish = row.get('ma_bullish_align', 0)
    above_all_ma = row.get('above_all_ma', 0)
    if ma_bullish and above_all_ma:
        reasons.append("均线多头排列")
    elif above_all_ma:
        reasons.append("站上所有均线")
    
    # 7. Sector relative strength
    vs_sector = row.get('vs_sector_ret_5', 0)
    sector_cn = row.get('sector_cn', '')
    if vs_sector > 0.03 and sector_cn:
        reasons.append(f"跑赢{sector_cn}板块")
    
    # 8. Trend initiation features
    gap_up = row.get('strong_gap_up', 0)
    if gap_up:
        reasons.append("跳空高开突破")
    
    bullish_days = row.get('bullish_days_5', 0)
    if bullish_days >= 4:
        reasons.append("连续阳线")
    
    # 9. Resilience
    resilience = row.get('resilience', 0)
    if resilience > 0.7:
        reasons.append("抗跌性强")
    
    # 10. Volume-price health
    vol_price_health = row.get('vol_price_health', 0)
    if vol_price_health > 0.7:
        reasons.append("量价配合良好")
    
    # If no specific reasons, use generic
    if not reasons:
        pred_ret = row.get('pred_ret_5', 0)
        if pred_ret > 0:
            reasons.append("综合技术指标看好")
        else:
            reasons.append("模型预测入选")
    
    # Combine reasons (max 3)
    return "，".join(reasons[:3])


def generate_reason_with_news(row: pd.Series, news_df: pd.DataFrame = None) -> str:
    """
    Generate reason including news sentiment if available
    
    Args:
        row: A row from predictions DataFrame
        news_df: News sentiment DataFrame
        
    Returns:
        Chinese reason string with news info if available
    """
    # Get base reason from technical indicators
    base_reason = generate_reason_cn(row)
    
    # Try to add news info
    if news_df is not None and not news_df.empty:
        symbol = row.get('symbol', '')
        # Extract just the code part
        code = str(symbol).split('.')[0]
        
        # Look for news for this symbol
        news_row = news_df[news_df['symbol'].astype(str).str.contains(code)]
        
        if not news_row.empty:
            news_row = news_row.iloc[0]
            sentiment = news_row.get('sentiment_mean', 0.5)
            news_count = news_row.get('news_count', 0)
            positive_ratio = news_row.get('sentiment_positive_ratio', 0.5)
            
            if news_count > 0:
                if sentiment > 0.6 and positive_ratio > 0.6:
                    base_reason += "；近期利好消息多"
                elif sentiment > 0.5 and positive_ratio > 0.5:
                    base_reason += "；舆论偏正面"
                elif sentiment < 0.4:
                    base_reason += "；关注近期消息面"
    
    return base_reason


def add_selection_reasons(df: pd.DataFrame, include_news: bool = True) -> pd.DataFrame:
    """
    Add Chinese selection reasons to predictions DataFrame
    
    Args:
        df: Predictions DataFrame with features
        include_news: Whether to include news sentiment info
        
    Returns:
        DataFrame with 'reason_cn' column added
    """
    df = df.copy()
    
    # Load news data if requested
    news_df = None
    if include_news:
        try:
            news_df = pd.read_parquet(NEWS_SENTIMENT_FILE)
            logger.info(f"Loaded news sentiment for {len(news_df)} symbols")
        except Exception as e:
            logger.warning(f"Could not load news data: {e}")
    
    # Generate reasons
    if news_df is not None:
        df['reason_cn'] = df.apply(lambda row: generate_reason_with_news(row, news_df), axis=1)
    else:
        df['reason_cn'] = df.apply(generate_reason_cn, axis=1)
    
    logger.info(f"Generated Chinese reasons for {len(df)} stocks")
    
    return df


def get_latest_news_headlines(symbol: str, max_headlines: int = 3) -> List[str]:
    """
    Get latest news headlines for a stock
    
    Args:
        symbol: Stock symbol
        max_headlines: Maximum number of headlines to return
        
    Returns:
        List of headline strings
    """
    try:
        from data.fetch_news import fetch_eastmoney_news
        
        articles = fetch_eastmoney_news(symbol, max_articles=max_headlines)
        headlines = [a.get('title', '') for a in articles if a.get('title')]
        return headlines[:max_headlines]
    except Exception as e:
        logger.debug(f"Could not fetch headlines for {symbol}: {e}")
        return []


if __name__ == "__main__":
    # Test the reason generation
    import pandas as pd
    from config import TOP10_LATEST_FILE
    
    print("Loading latest predictions...")
    df = pd.read_parquet(TOP10_LATEST_FILE)
    
    print("\nGenerating reasons...")
    df = add_selection_reasons(df, include_news=True)
    
    print("\nTop-10 with reasons:")
    for _, row in df.iterrows():
        print(f"{row['symbol']}: {row['reason_cn']}")
