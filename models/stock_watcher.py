"""
Stock Watcher Module - Analyzes stocks from input file
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
from typing import List, Dict, Optional

from config import (
    WATCHER_INPUT_FILE, WATCHER_OUTPUT_FILE, WATCHER_HISTORY_FILE,
    BARS_FILE, FEATURES_Z_FILE,
    RANKER_MODEL_FILE, REGRESSOR_MODEL_FILE,
    RANKER_15D_MODEL_FILE, REGRESSOR_15D_MODEL_FILE,
    FEATURE_COLS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_watch_list():
    """Read stock codes from input file"""
    if not WATCHER_INPUT_FILE.exists():
        return []
    with open(WATCHER_INPUT_FILE, "r") as f:
        text = f.read().strip()
    codes = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for code in line.split(","):
            code = code.strip()
            if code and code.isdigit():
                codes.append(code)
    return list(set(codes))


def get_stock_exchange(symbol):
    """Get exchange based on stock code prefix"""
    if symbol.startswith(("600", "601", "603", "605", "688", "689")):
        return "SHG"
    elif symbol.startswith(("000", "001", "002", "003", "300", "301")):
        return "SHE"
    return "SHG"


def calc_support_resistance(df, symbol, lookback=20):
    """Calculate support and resistance levels"""
    stock_df = df[df["symbol"] == symbol].copy()
    if stock_df.empty or len(stock_df) < lookback:
        return {"support_1": None, "resistance_1": None}
    stock_df = stock_df.sort_values("date").tail(lookback)
    highs = stock_df["high"].values
    lows = stock_df["low"].values
    closes = stock_df["close"].values
    current_price = closes[-1]
    recent_low, recent_high = lows.min(), highs.max()
    pivot = (recent_high + recent_low + current_price) / 3
    return {
        "support_1": round(2 * pivot - recent_high, 2),
        "support_2": round(recent_low, 2),
        "resistance_1": round(2 * pivot - recent_low, 2),
        "resistance_2": round(recent_high, 2),
    }


def calc_buy_sell_prices(price, pred5, pred15, vol, sr):
    """Calculate recommended buy and sell prices"""
    pred_price_5d = price * (1 + pred5)
    pred_price_15d = price * (1 + pred15)
    vol_factor = max(vol, 0.02)
    support = sr.get("support_1") or (price * 0.97)
    buy_price = max(min(support, price * (1 - vol_factor * 0.5)), price * 0.93)
    resistance = sr.get("resistance_1") or (price * 1.05)
    sell_5d = pred_price_5d * 0.95 if pred5 > 0.02 else min(resistance, price * (1 + vol_factor))
    sell_15d = pred_price_15d * 0.95 if pred15 > 0.03 else min(resistance * 1.02, price * (1 + vol_factor * 1.5))
    stop = max(sr.get("support_2") or (price * 0.92), price * 0.90)
    return {
        "buy_price": round(buy_price, 2),
        "sell_price_5d": round(sell_5d, 2),
        "sell_price_15d": round(sell_15d, 2),
        "stop_loss": round(stop, 2),
        "pred_price_5d": round(pred_price_5d, 2),
        "pred_price_15d": round(pred_price_15d, 2)
    }


def gen_recommendation(p5, p15, r5, r15, vol, trend=0, news=0.5):
    """Generate buy/sell recommendation with reason and confidence"""
    ret_score = p5 * 0.4 + p15 * 0.6
    conf = 0.5
    if p5 > 0.03 and p15 > 0.05:
        conf += 0.2
    elif p5 < -0.02 and p15 < -0.03:
        conf += 0.15
    if r5 * 0.4 + r15 * 0.6 > 0.7:
        conf += 0.1
    if vol < 0.03:
        conf += 0.1
    elif vol > 0.06:
        conf -= 0.1
    conf = min(max(conf, 0.1), 0.95)
    
    reasons = []
    if ret_score > 0.03:
        action = "买入"
        if p5 > 0.03:
            reasons.append(f"5日预测涨幅{p5*100:.1f}%")
        if p15 > 0.05:
            reasons.append(f"15日预测涨幅{p15*100:.1f}%")
    elif ret_score < -0.02:
        action = "卖出"
        if p5 < -0.02:
            reasons.append(f"5日预测跌幅{abs(p5)*100:.1f}%")
    elif ret_score > 0.01:
        action = "持有"
        reasons.append("预期小幅上涨")
    else:
        action = "观望"
        reasons.append("预期平稳")
    
    return action, "，".join(reasons) if reasons else "综合分析", round(conf, 2)


class StockWatcher:
    """Stock Watcher for analyzing watched stocks"""
    
    def __init__(self):
        self.model_5d = None
        self.model_15d = None
        self.ranker_5d = None
        self.ranker_15d = None
        self.features_df = None
        self.bars_df = None
        
    def load_models(self):
        """Load prediction models - models are stored as dicts with 'model' key"""
        
        if RANKER_MODEL_FILE.exists():
            with open(RANKER_MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
                self.ranker_5d = data['model'] if isinstance(data, dict) and 'model' in data else data
            logger.info("Loaded 5-day ranker")
            
        if REGRESSOR_MODEL_FILE.exists():
            with open(REGRESSOR_MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
                self.model_5d = data['model'] if isinstance(data, dict) and 'model' in data else data
            logger.info("Loaded 5-day regressor")
            
        if RANKER_15D_MODEL_FILE.exists():
            with open(RANKER_15D_MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
                self.ranker_15d = data['model'] if isinstance(data, dict) and 'model' in data else data
            logger.info("Loaded 15-day ranker")
            
        if REGRESSOR_15D_MODEL_FILE.exists():
            with open(REGRESSOR_15D_MODEL_FILE, 'rb') as f:
                data = pickle.load(f)
                self.model_15d = data['model'] if isinstance(data, dict) and 'model' in data else data
            logger.info("Loaded 15-day regressor")
    
    def load_data(self):
        """Load features and bars data"""
        if FEATURES_Z_FILE.exists():
            self.features_df = pd.read_parquet(FEATURES_Z_FILE)
            logger.info(f"Loaded features: {len(self.features_df)} rows")
        if BARS_FILE.exists():
            self.bars_df = pd.read_parquet(BARS_FILE)
            logger.info(f"Loaded bars: {len(self.bars_df)} rows")
    
    def analyze_stock(self, symbol, latest_date):
        """Analyze a single stock"""
        if self.features_df is None or self.bars_df is None:
            return None
        
        # Check what symbol format is used in features
        sample_sym = self.features_df["symbol"].iloc[0] if len(self.features_df) > 0 else ""
        has_exchange_suffix = "." in str(sample_sym)
        
        if has_exchange_suffix:
            # Features use format like "300136.SHE"
            exchange = get_stock_exchange(symbol)
            full_symbol = f"{symbol}.{exchange}"
        else:
            # Features use plain format like "300136"
            full_symbol = symbol
            exchange = get_stock_exchange(symbol)
        
        stock_features = self.features_df[
            (self.features_df["symbol"] == full_symbol) & 
            (self.features_df["date"] == latest_date)
        ]
        
        if stock_features.empty:
            logger.warning(f"No features for {full_symbol} on {latest_date}")
            return None
        
        stock_row = stock_features.iloc[0]
        available_cols = [c for c in FEATURE_COLS if c in stock_features.columns]
        X = stock_features[available_cols].values
        
        pred_ret_5, pred_ret_15 = 0.0, 0.0
        rank_5, rank_15 = 0.5, 0.5
        
        if self.model_5d:
            pred_ret_5 = float(self.model_5d.predict(X)[0])
        if self.model_15d:
            pred_ret_15 = float(self.model_15d.predict(X)[0])
        if self.ranker_5d:
            rank_5 = float(self.ranker_5d.predict_proba(X)[0][1]) if hasattr(self.ranker_5d, "predict_proba") else 0.5
        if self.ranker_15d:
            rank_15 = float(self.ranker_15d.predict_proba(X)[0][1]) if hasattr(self.ranker_15d, "predict_proba") else 0.5
        
        current_price = float(stock_row.get("close", stock_row.get("adj_close", 0)))
        vol = float(stock_row.get("vol_5", 0.03))
        trend = float(stock_row.get("trend_score", 0.5))
        news = float(stock_row.get("news_sentiment_mean", 0.5))
        
        # Bars also use plain symbol format
        bars_symbol = symbol  # Always use plain symbol for bars lookup
        sr = calc_support_resistance(self.bars_df, bars_symbol)
        prices = calc_buy_sell_prices(current_price, pred_ret_5, pred_ret_15, vol, sr)
        action, reason, conf = gen_recommendation(pred_ret_5, pred_ret_15, rank_5, rank_15, vol, trend, news)
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "date": latest_date,
            "current_price": current_price,
            "pred_ret_5": pred_ret_5,
            "pred_ret_15": pred_ret_15,
            "pred_price_5d": prices["pred_price_5d"],
            "pred_price_15d": prices["pred_price_15d"],
            "rank_score_5": rank_5,
            "rank_score_15": rank_15,
            "buy_price": prices["buy_price"],
            "sell_price_5d": prices["sell_price_5d"],
            "sell_price_15d": prices["sell_price_15d"],
            "stop_loss": prices["stop_loss"],
            "action": action,
            "reason_cn": reason,
            "confidence": conf,
            "volatility": vol,
            "support_1": sr.get("support_1"),
            "resistance_1": sr.get("resistance_1"),
        }
    
    def analyze_watch_list(self):
        """Analyze all stocks in watch list"""
        symbols = read_watch_list()
        if not symbols:
            logger.warning("No symbols in watch list")
            return pd.DataFrame()
        
        logger.info(f"Analyzing {len(symbols)} stocks: {symbols}")
        self.load_models()
        self.load_data()
        
        if self.features_df is None:
            logger.error("No features data")
            return pd.DataFrame()
        
        latest_date = self.features_df["date"].max()
        logger.info(f"Latest date: {latest_date}")
        
        results = []
        for symbol in symbols:
            result = self.analyze_stock(symbol, latest_date)
            if result:
                results.append(result)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return df
    
    def save_predictions(self, df):
        """Save predictions to output files"""
        if df.empty:
            return
        
        df.to_parquet(WATCHER_OUTPUT_FILE, index=False)
        logger.info(f"Saved to {WATCHER_OUTPUT_FILE}")
        
        if WATCHER_HISTORY_FILE.exists():
            history = pd.read_parquet(WATCHER_HISTORY_FILE)
            mask = ~((history["symbol"].isin(df["symbol"])) & 
                     (history["date"] == df["date"].iloc[0]))
            history = pd.concat([history[mask], df], ignore_index=True)
        else:
            history = df
        
        history.to_parquet(WATCHER_HISTORY_FILE, index=False)
        logger.info(f"History updated: {len(history)} records")


def run_stock_watcher():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("Running Stock Watcher")
    logger.info("=" * 50)
    
    watcher = StockWatcher()
    predictions = watcher.analyze_watch_list()
    
    if predictions.empty:
        logger.warning("No predictions generated")
        return None
    
    print("\n📊 Stock Watcher Predictions")
    print("=" * 60)
    for _, row in predictions.iterrows():
        print(f"\n🔷 {row['symbol']} - {row['action']} (置信度: {row['confidence']*100:.0f}%)")
        print(f"   当前价格: ¥{row['current_price']:.2f}")
        print(f"   5日预测: {row['pred_ret_5']*100:+.2f}% → ¥{row['pred_price_5d']:.2f}")
        print(f"   15日预测: {row['pred_ret_15']*100:+.2f}% → ¥{row['pred_price_15d']:.2f}")
        print(f"   建议买入: ¥{row['buy_price']:.2f}")
        print(f"   建议卖出(5日): ¥{row['sell_price_5d']:.2f}")
        print(f"   建议卖出(15日): ¥{row['sell_price_15d']:.2f}")
        print(f"   止损价: ¥{row['stop_loss']:.2f}")
        print(f"   理由: {row['reason_cn']}")
    print("\n" + "=" * 60)
    
    watcher.save_predictions(predictions)
    return predictions


if __name__ == "__main__":
    run_stock_watcher()
