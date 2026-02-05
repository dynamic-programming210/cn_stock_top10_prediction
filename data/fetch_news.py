"""
News Sentiment Analysis for Chinese A-Shares
Fetches news from Chinese financial sources and computes sentiment features

Sources:
- East Money (东方财富): https://stock.eastmoney.com/
- Sina Finance (新浪财经): https://cj.sina.cn/
- THS (同花顺): https://www.10jqka.com.cn/

Uses SnowNLP for Chinese sentiment analysis (simple, no GPU needed)
Alternative: Use transformers with chinese-roberta-wwm-ext for better accuracy
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import requests
import json
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, NEWS_SENTIMENT_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output files
NEWS_CACHE_FILE = DATA_DIR / "news_cache.parquet"

# Settings
MAX_NEWS_AGE_DAYS = 7  # Only consider news from last N days
MAX_NEWS_PER_STOCK = 20  # Maximum news articles to fetch per stock
REQUEST_DELAY = 0.5  # Delay between requests to avoid rate limiting

# Try to import Chinese NLP
try:
    from snownlp import SnowNLP
    SNOWNLP_AVAILABLE = True
except ImportError:
    SNOWNLP_AVAILABLE = False
    logger.warning("SnowNLP not installed. Run: pip install snownlp")

# Try to import transformers for better sentiment (optional)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ChineseSentimentAnalyzer:
    """Chinese text sentiment analyzer using SnowNLP or transformers"""
    
    def __init__(self, use_transformers: bool = False):
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.model = None
        
        if self.use_transformers:
            try:
                # Use a Chinese sentiment model
                model_name = "uer/roberta-base-finetuned-jd-binary-chinese"
                self.model = pipeline("sentiment-analysis", model=model_name)
                logger.info(f"Loaded transformer model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}")
                self.use_transformers = False
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of Chinese text
        
        Returns:
            dict with 'positive', 'negative', 'compound' scores (0-1 scale)
        """
        if not text or len(text.strip()) < 5:
            return {'positive': 0.5, 'negative': 0.5, 'compound': 0.0}
        
        try:
            if self.use_transformers and self.model:
                result = self.model(text[:512])[0]  # Truncate to 512 tokens
                score = result['score']
                if result['label'] == 'positive':
                    return {'positive': score, 'negative': 1-score, 'compound': score - 0.5}
                else:
                    return {'positive': 1-score, 'negative': score, 'compound': 0.5 - score}
            elif SNOWNLP_AVAILABLE:
                s = SnowNLP(text)
                sentiment = s.sentiments  # 0-1, higher = more positive
                return {
                    'positive': sentiment,
                    'negative': 1 - sentiment,
                    'compound': sentiment - 0.5  # -0.5 to 0.5 scale
                }
            else:
                # Fallback: simple keyword-based
                return self._keyword_sentiment(text)
        except Exception as e:
            logger.debug(f"Sentiment analysis error: {e}")
            return {'positive': 0.5, 'negative': 0.5, 'compound': 0.0}
    
    def _keyword_sentiment(self, text: str) -> Dict[str, float]:
        """Simple keyword-based sentiment as fallback"""
        positive_words = ['涨', '上涨', '利好', '增长', '突破', '买入', '看好', '牛市', 
                         '反弹', '上升', '盈利', '增持', '推荐', '强势', '创新高']
        negative_words = ['跌', '下跌', '利空', '下降', '跌破', '卖出', '看空', '熊市',
                         '回调', '下滑', '亏损', '减持', '风险', '弱势', '创新低']
        
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        total = pos_count + neg_count + 1
        
        positive = pos_count / total
        negative = neg_count / total
        compound = (pos_count - neg_count) / total
        
        return {'positive': positive, 'negative': negative, 'compound': compound}


def convert_stock_code(symbol: str) -> Tuple[str, str]:
    """
    Convert symbol to market code format
    
    Returns: (code, market) where market is 'sh' or 'sz'
    """
    code = str(symbol).split('.')[0].zfill(6)
    
    # Shanghai: 600xxx, 601xxx, 603xxx, 605xxx, 688xxx (STAR)
    # Shenzhen: 000xxx, 001xxx, 002xxx, 003xxx, 300xxx (ChiNext)
    if code.startswith(('600', '601', '603', '605', '688')):
        return code, 'sh'
    else:
        return code, 'sz'


def fetch_eastmoney_news(symbol: str, max_articles: int = MAX_NEWS_PER_STOCK) -> List[Dict]:
    """
    Fetch news from East Money (东方财富) API
    
    East Money has a public API that returns stock-specific news
    """
    code, market = convert_stock_code(symbol)
    full_code = f"{market}{code}"
    
    # East Money news API endpoint
    url = f"https://search-api-web.eastmoney.com/search/jsonp"
    
    params = {
        'cb': 'jQuery_callback',
        'param': json.dumps({
            'uid': '',
            'keyword': full_code,
            'type': ['cmsArticleWebOld'],  # News articles
            'client': 'web',
            'clientType': 'web',
            'clientVersion': 'curr',
            'pageIndex': 1,
            'pageSize': max_articles,
        }),
        '_': int(time.time() * 1000)
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://so.eastmoney.com/'
    }
    
    articles = []
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        # Parse JSONP response
        text = response.text
        json_str = re.search(r'jQuery_callback\((.*)\)', text)
        if not json_str:
            return articles
            
        data = json.loads(json_str.group(1))
        
        news_list = data.get('result', {}).get('cmsArticleWebOld', [])
        
        for item in news_list:
            pub_time = None
            date_str = item.get('date', '')
            if date_str:
                try:
                    pub_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                except:
                    pass
            
            article = {
                'symbol': symbol,
                'title': item.get('title', ''),
                'content': item.get('content', ''),  # Brief content/summary
                'source': item.get('mediaName', 'eastmoney'),
                'url': item.get('url', ''),
                'publish_time': pub_time,
            }
            articles.append(article)
            
    except Exception as e:
        logger.debug(f"East Money fetch error for {symbol}: {e}")
    
    return articles


def fetch_sina_news(symbol: str, max_articles: int = MAX_NEWS_PER_STOCK) -> List[Dict]:
    """
    Fetch news from Sina Finance (新浪财经) API
    """
    code, market = convert_stock_code(symbol)
    
    # Sina news API
    url = f"https://feed.mix.sina.com.cn/api/roll/get"
    
    params = {
        'pageid': '153',
        'lid': '2516',  # Stock news category
        'k': code,
        'num': max_articles,
        'page': 1,
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://finance.sina.com.cn/'
    }
    
    articles = []
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        news_list = data.get('result', {}).get('data', [])
        
        for item in news_list:
            pub_time = None
            ctime = item.get('ctime')
            if ctime:
                try:
                    pub_time = datetime.fromtimestamp(int(ctime))
                except:
                    pass
            
            article = {
                'symbol': symbol,
                'title': item.get('title', ''),
                'content': item.get('intro', ''),
                'source': item.get('media_name', 'sina'),
                'url': item.get('url', ''),
                'publish_time': pub_time,
            }
            articles.append(article)
            
    except Exception as e:
        logger.debug(f"Sina fetch error for {symbol}: {e}")
    
    return articles


def fetch_ths_news(symbol: str, max_articles: int = MAX_NEWS_PER_STOCK) -> List[Dict]:
    """
    Fetch news from THS/同花顺 (10jqka)
    
    Note: THS has stricter anti-scraping, this may need adjustment
    """
    code, market = convert_stock_code(symbol)
    
    # THS individual stock news page
    url = f"http://basic.10jqka.com.cn/{code}/news.html"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': f'http://stockpage.10jqka.com.cn/{code}/'
    }
    
    articles = []
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'gbk'  # THS uses GBK encoding
        
        # Simple regex extraction (basic parsing)
        # Full implementation would use BeautifulSoup
        title_pattern = r'<span class="title"[^>]*>([^<]+)</span>'
        titles = re.findall(title_pattern, response.text)
        
        for title in titles[:max_articles]:
            article = {
                'symbol': symbol,
                'title': title.strip(),
                'content': '',
                'source': '10jqka',
                'url': url,
                'publish_time': datetime.now(),  # THS doesn't always show timestamps
            }
            articles.append(article)
            
    except Exception as e:
        logger.debug(f"THS fetch error for {symbol}: {e}")
    
    return articles


def fetch_all_news_for_symbol(symbol: str, sources: List[str] = None) -> List[Dict]:
    """
    Fetch news from multiple sources for a single symbol
    
    Args:
        symbol: Stock symbol (e.g., '600519' or '600519.SHG')
        sources: List of sources to use. Default: ['eastmoney', 'sina']
    """
    if sources is None:
        sources = ['eastmoney', 'sina']  # THS is harder to scrape
    
    all_articles = []
    
    if 'eastmoney' in sources:
        articles = fetch_eastmoney_news(symbol)
        all_articles.extend(articles)
        time.sleep(REQUEST_DELAY)
    
    if 'sina' in sources:
        articles = fetch_sina_news(symbol)
        all_articles.extend(articles)
        time.sleep(REQUEST_DELAY)
    
    if 'ths' in sources:
        articles = fetch_ths_news(symbol)
        all_articles.extend(articles)
        time.sleep(REQUEST_DELAY)
    
    # Filter by date
    cutoff = datetime.now() - timedelta(days=MAX_NEWS_AGE_DAYS)
    filtered = [
        a for a in all_articles 
        if a.get('publish_time') is None or a['publish_time'] >= cutoff
    ]
    
    return filtered


def compute_news_sentiment(articles: List[Dict], analyzer: ChineseSentimentAnalyzer = None) -> Dict:
    """
    Compute aggregated sentiment features from news articles
    
    Returns dict with:
        - news_count: Number of articles
        - sentiment_mean: Average sentiment (-0.5 to 0.5)
        - sentiment_std: Sentiment variance (disagreement)
        - sentiment_positive_ratio: % of positive articles
        - sentiment_trend: Recent vs older sentiment
    """
    if not articles:
        return {
            'news_count': 0,
            'sentiment_mean': 0.0,
            'sentiment_std': 0.0,
            'sentiment_positive_ratio': 0.5,
            'sentiment_trend': 0.0,
            'sentiment_max': 0.0,
            'sentiment_min': 0.0,
        }
    
    if analyzer is None:
        analyzer = ChineseSentimentAnalyzer()
    
    # Analyze each article
    sentiments = []
    for article in articles:
        # Combine title and content for analysis
        text = f"{article.get('title', '')} {article.get('content', '')}"
        result = analyzer.analyze(text)
        sentiments.append({
            **result,
            'publish_time': article.get('publish_time')
        })
    
    # Compute aggregated features
    compounds = [s['compound'] for s in sentiments]
    
    features = {
        'news_count': len(articles),
        'sentiment_mean': np.mean(compounds),
        'sentiment_std': np.std(compounds) if len(compounds) > 1 else 0.0,
        'sentiment_positive_ratio': np.mean([1 if s['compound'] > 0 else 0 for s in sentiments]),
        'sentiment_max': np.max(compounds),
        'sentiment_min': np.min(compounds),
    }
    
    # Compute trend (recent vs older articles)
    if len(sentiments) >= 4:
        mid = len(sentiments) // 2
        recent = np.mean(compounds[:mid])
        older = np.mean(compounds[mid:])
        features['sentiment_trend'] = recent - older
    else:
        features['sentiment_trend'] = 0.0
    
    return features


def fetch_news_sentiment_batch(symbols: List[str], 
                               max_workers: int = 5,
                               sources: List[str] = None) -> pd.DataFrame:
    """
    Fetch news and compute sentiment for multiple symbols in parallel
    
    Args:
        symbols: List of stock symbols
        max_workers: Number of parallel workers
        sources: News sources to use
    
    Returns:
        DataFrame with sentiment features per symbol
    """
    analyzer = ChineseSentimentAnalyzer()
    results = []
    
    logger.info(f"Fetching news for {len(symbols)} symbols...")
    
    def process_symbol(symbol):
        try:
            articles = fetch_all_news_for_symbol(symbol, sources)
            sentiment = compute_news_sentiment(articles, analyzer)
            sentiment['symbol'] = symbol
            sentiment['fetch_time'] = datetime.now()
            return sentiment
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")
            return {'symbol': symbol, 'news_count': 0, 'sentiment_mean': 0.0}
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_symbol, s): s for s in symbols}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/{len(symbols)} symbols")
    
    df = pd.DataFrame(results)
    return df


def update_news_sentiment(symbols: List[str] = None, save: bool = True) -> pd.DataFrame:
    """
    Update news sentiment data for all symbols or specified list
    
    Args:
        symbols: List of symbols to update. If None, uses universe.
        save: Whether to save to parquet file
    
    Returns:
        DataFrame with news sentiment features
    """
    from config import UNIVERSE_FILE
    
    if symbols is None:
        # Load universe
        universe = pd.read_parquet(UNIVERSE_FILE)
        symbols = universe['symbol'].unique().tolist()
    
    # Fetch and compute sentiment
    df = fetch_news_sentiment_batch(symbols)
    
    # Add date column (for merging with price data)
    df['date'] = pd.Timestamp.now().normalize()
    
    if save:
        # Append to existing file or create new
        if NEWS_SENTIMENT_FILE.exists():
            existing = pd.read_parquet(NEWS_SENTIMENT_FILE)
            # Remove old data for today (in case of re-run)
            today = pd.Timestamp.now().normalize()
            existing = existing[existing['date'] != today]
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_parquet(NEWS_SENTIMENT_FILE, index=False)
        logger.info(f"Saved news sentiment to {NEWS_SENTIMENT_FILE}")
    
    return df


def get_news_features(df: pd.DataFrame, lookback_days: int = 7) -> pd.DataFrame:
    """
    Add news sentiment features to price/feature DataFrame
    
    Args:
        df: DataFrame with 'symbol' and 'date' columns
        lookback_days: Number of days to aggregate news
    
    Returns:
        DataFrame with added news features
    """
    if not NEWS_SENTIMENT_FILE.exists():
        logger.warning("No news sentiment data found. Run update_news_sentiment() first.")
        return df
    
    news_df = pd.read_parquet(NEWS_SENTIMENT_FILE)
    
    # Aggregate news features over lookback period
    feature_cols = ['news_count', 'sentiment_mean', 'sentiment_std', 
                    'sentiment_positive_ratio', 'sentiment_trend']
    
    # For each date in df, look back N days of news
    # This is a simplified version - production would use rolling windows
    latest_news = news_df.groupby('symbol')[feature_cols].mean().reset_index()
    latest_news.columns = ['symbol'] + [f'news_{c}' for c in feature_cols]
    
    df = df.merge(latest_news, on='symbol', how='left')
    
    # Fill NaN with neutral values
    df['news_news_count'] = df.get('news_news_count', 0).fillna(0)
    df['news_sentiment_mean'] = df.get('news_sentiment_mean', 0).fillna(0)
    df['news_sentiment_std'] = df.get('news_sentiment_std', 0).fillna(0)
    df['news_sentiment_positive_ratio'] = df.get('news_sentiment_positive_ratio', 0.5).fillna(0.5)
    df['news_sentiment_trend'] = df.get('news_sentiment_trend', 0).fillna(0)
    
    return df


# ============ Market-wide sentiment (大盘情绪) ============

def fetch_market_sentiment() -> Dict:
    """
    Fetch overall market sentiment indicators
    
    Sources: 
    - Index news (上证指数, 沪深300, 创业板)
    - Hot stock rankings (热门股)
    - Overall market headlines
    """
    market_codes = ['000001', '399001', '399006']  # 上证, 深证, 创业板
    
    all_articles = []
    for code in market_codes:
        articles = fetch_eastmoney_news(code)
        all_articles.extend(articles)
        time.sleep(REQUEST_DELAY)
    
    analyzer = ChineseSentimentAnalyzer()
    return compute_news_sentiment(all_articles, analyzer)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Chinese stock news sentiment")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")
    parser.add_argument("--test", action="store_true", help="Test with a few symbols")
    parser.add_argument("--market", action="store_true", help="Fetch market-wide sentiment")
    
    args = parser.parse_args()
    
    if args.test:
        # Test with famous stocks
        test_symbols = ['600519', '000858', '601318', '600036', '000001']
        print(f"\n🧪 Testing with {len(test_symbols)} symbols...")
        
        analyzer = ChineseSentimentAnalyzer()
        
        for symbol in test_symbols:
            print(f"\n📰 {symbol}:")
            articles = fetch_all_news_for_symbol(symbol)
            print(f"   Found {len(articles)} articles")
            
            if articles:
                sentiment = compute_news_sentiment(articles, analyzer)
                print(f"   Sentiment: {sentiment['sentiment_mean']:.3f}")
                print(f"   News count: {sentiment['news_count']}")
                
                # Show first article
                if articles[0].get('title'):
                    print(f"   Latest: {articles[0]['title'][:50]}...")
    
    elif args.market:
        print("\n📊 Fetching market-wide sentiment...")
        sentiment = fetch_market_sentiment()
        print(f"Market sentiment: {sentiment['sentiment_mean']:.3f}")
        print(f"News count: {sentiment['news_count']}")
    
    elif args.symbols:
        df = fetch_news_sentiment_batch(args.symbols)
        print(df.to_string())
    
    else:
        print("Updating news sentiment for all symbols...")
        df = update_news_sentiment()
        print(f"\n✅ Updated sentiment for {len(df)} symbols")
        print(df.head(10).to_string())
