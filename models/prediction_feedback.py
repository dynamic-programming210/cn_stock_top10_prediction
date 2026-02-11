"""
Prediction Feedback Module - Learn from past prediction errors

This module implements a feedback loop where the model learns from the difference
between its predictions and actual outcomes. Key features:

1. Track prediction history with actual outcomes
2. Compute prediction errors (predicted - actual)
3. Generate feedback features for training
4. Create sample weights based on prediction errors
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

from config import (
    TOP10_HISTORY_FILE, TOP10_HISTORY_15D_FILE,
    BARS_FILE, FEATURES_Z_FILE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feedback data files
FEEDBACK_FILE = Path(__file__).parent.parent / "outputs" / "prediction_feedback.parquet"
FEEDBACK_STATS_FILE = Path(__file__).parent.parent / "outputs" / "feedback_stats.parquet"


def load_prediction_history() -> pd.DataFrame:
    """Load all historical predictions (5-day and 15-day)"""
    dfs = []
    
    if TOP10_HISTORY_FILE.exists():
        df_5d = pd.read_parquet(TOP10_HISTORY_FILE)
        df_5d['horizon'] = 5
        df_5d['pred_return'] = df_5d.get('pred_ret_5', df_5d.get('pred_ret', 0))
        dfs.append(df_5d)
        logger.info(f"Loaded 5-day history: {len(df_5d)} records, {df_5d['date'].nunique()} dates")
    
    if TOP10_HISTORY_15D_FILE.exists():
        df_15d = pd.read_parquet(TOP10_HISTORY_15D_FILE)
        df_15d['horizon'] = 15
        df_15d['pred_return'] = df_15d.get('pred_ret_15', df_15d.get('pred_ret', 0))
        dfs.append(df_15d)
        logger.info(f"Loaded 15-day history: {len(df_15d)} records, {df_15d['date'].nunique()} dates")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    
    return combined


def compute_actual_returns(predictions: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute actual returns for past predictions.
    
    For each prediction made on date D for horizon H days:
    - actual_return = (close on D+H) / (close on D) - 1
    """
    if predictions.empty:
        return predictions
    
    predictions = predictions.copy()
    bars = bars.copy()
    bars['date'] = pd.to_datetime(bars['date'])
    
    # Create a lookup for close prices
    price_lookup = bars.set_index(['symbol', 'date'])['close'].to_dict()
    
    actual_returns = []
    
    for _, row in predictions.iterrows():
        symbol = str(row['symbol'])
        pred_date = row['date']
        horizon = row['horizon']
        
        # Find the actual date H trading days later
        symbol_dates = bars[bars['symbol'] == symbol]['date'].sort_values().unique()
        
        try:
            pred_date_idx = np.where(symbol_dates == pred_date)[0]
            if len(pred_date_idx) == 0:
                # Find closest date
                pred_date_idx = np.searchsorted(symbol_dates, pred_date)
            else:
                pred_date_idx = pred_date_idx[0]
            
            future_idx = pred_date_idx + horizon
            
            if future_idx < len(symbol_dates):
                future_date = symbol_dates[future_idx]
                
                price_at_pred = price_lookup.get((symbol, pred_date))
                price_at_future = price_lookup.get((symbol, future_date))
                
                if price_at_pred and price_at_future and price_at_pred > 0:
                    actual_ret = (price_at_future / price_at_pred) - 1
                    actual_returns.append({
                        'actual_return': actual_ret,
                        'actual_date': future_date,
                        'has_actual': True
                    })
                else:
                    actual_returns.append({'actual_return': np.nan, 'actual_date': None, 'has_actual': False})
            else:
                # Future date not yet available
                actual_returns.append({'actual_return': np.nan, 'actual_date': None, 'has_actual': False})
        except Exception as e:
            actual_returns.append({'actual_return': np.nan, 'actual_date': None, 'has_actual': False})
    
    actual_df = pd.DataFrame(actual_returns)
    predictions = pd.concat([predictions.reset_index(drop=True), actual_df], axis=1)
    
    # Compute prediction error
    predictions['prediction_error'] = predictions['pred_return'] - predictions['actual_return']
    predictions['absolute_error'] = predictions['prediction_error'].abs()
    predictions['correct_direction'] = (
        (predictions['pred_return'] > 0) == (predictions['actual_return'] > 0)
    ).astype(int)
    
    return predictions


def build_feedback_dataset() -> pd.DataFrame:
    """
    Build a dataset of predictions with actual outcomes.
    This is the core feedback data for learning from past errors.
    """
    logger.info("Building feedback dataset...")
    
    # Load predictions
    predictions = load_prediction_history()
    if predictions.empty:
        logger.warning("No prediction history found")
        return pd.DataFrame()
    
    # Load bars for actual returns
    if not BARS_FILE.exists():
        logger.warning("No bars data found")
        return predictions
    
    bars = pd.read_parquet(BARS_FILE)
    
    # Compute actual returns
    feedback = compute_actual_returns(predictions, bars)
    
    # Filter to rows with actual outcomes
    feedback_complete = feedback[feedback['has_actual'] == True].copy()
    
    logger.info(f"Feedback dataset: {len(feedback_complete)} predictions with actual outcomes")
    logger.info(f"  Date range: {feedback_complete['date'].min()} to {feedback_complete['date'].max()}")
    
    if len(feedback_complete) > 0:
        mae_5d = feedback_complete[feedback_complete['horizon'] == 5]['absolute_error'].mean()
        mae_15d = feedback_complete[feedback_complete['horizon'] == 15]['absolute_error'].mean()
        dir_acc_5d = feedback_complete[feedback_complete['horizon'] == 5]['correct_direction'].mean()
        dir_acc_15d = feedback_complete[feedback_complete['horizon'] == 15]['correct_direction'].mean()
        
        logger.info(f"  5-day MAE: {mae_5d:.4f}, Direction accuracy: {dir_acc_5d:.2%}")
        logger.info(f"  15-day MAE: {mae_15d:.4f}, Direction accuracy: {dir_acc_15d:.2%}")
    
    # Save feedback dataset
    feedback_complete.to_parquet(FEEDBACK_FILE, index=False)
    logger.info(f"Saved feedback dataset to {FEEDBACK_FILE}")
    
    return feedback_complete


def compute_stock_feedback_stats(feedback: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-stock feedback statistics.
    These can be used as features for future predictions.
    
    Returns a DataFrame with:
    - symbol
    - avg_prediction_error_5d: Average error for 5-day predictions
    - avg_prediction_error_15d: Average error for 15-day predictions
    - prediction_bias_5d: Systematic over/under prediction
    - prediction_bias_15d: Systematic over/under prediction
    - direction_accuracy_5d: How often we get the direction right
    - direction_accuracy_15d: How often we get the direction right
    - prediction_count: Number of times we predicted this stock
    """
    if feedback.empty:
        return pd.DataFrame()
    
    stats_list = []
    
    for symbol in feedback['symbol'].unique():
        stock_fb = feedback[feedback['symbol'] == symbol]
        
        stats = {'symbol': symbol}
        
        for horizon in [5, 15]:
            h_fb = stock_fb[stock_fb['horizon'] == horizon]
            if len(h_fb) > 0:
                stats[f'avg_abs_error_{horizon}d'] = h_fb['absolute_error'].mean()
                stats[f'prediction_bias_{horizon}d'] = h_fb['prediction_error'].mean()  # +ve = over-predict
                stats[f'direction_accuracy_{horizon}d'] = h_fb['correct_direction'].mean()
                stats[f'prediction_count_{horizon}d'] = len(h_fb)
            else:
                stats[f'avg_abs_error_{horizon}d'] = np.nan
                stats[f'prediction_bias_{horizon}d'] = np.nan
                stats[f'direction_accuracy_{horizon}d'] = np.nan
                stats[f'prediction_count_{horizon}d'] = 0
        
        stats['total_predictions'] = len(stock_fb)
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    # Save stats
    stats_df.to_parquet(FEEDBACK_STATS_FILE, index=False)
    logger.info(f"Saved feedback stats for {len(stats_df)} stocks to {FEEDBACK_STATS_FILE}")
    
    return stats_df


def add_feedback_features(df: pd.DataFrame, feedback_stats: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add feedback-based features to the training data.
    
    New features:
    - hist_pred_bias_5d: Historical prediction bias for this stock (5-day)
    - hist_pred_bias_15d: Historical prediction bias for this stock (15-day)
    - hist_direction_acc_5d: Historical direction accuracy (5-day)
    - hist_direction_acc_15d: Historical direction accuracy (15-day)
    """
    if feedback_stats is None:
        if FEEDBACK_STATS_FILE.exists():
            feedback_stats = pd.read_parquet(FEEDBACK_STATS_FILE)
        else:
            logger.warning("No feedback stats available")
            return df
    
    if feedback_stats.empty:
        return df
    
    df = df.copy()
    
    # Merge feedback stats
    merge_cols = ['symbol']
    feedback_cols = [
        'prediction_bias_5d', 'prediction_bias_15d',
        'direction_accuracy_5d', 'direction_accuracy_15d',
        'avg_abs_error_5d', 'avg_abs_error_15d'
    ]
    
    available_cols = [c for c in feedback_cols if c in feedback_stats.columns]
    
    df = df.merge(
        feedback_stats[['symbol'] + available_cols],
        on='symbol',
        how='left'
    )
    
    # Rename to indicate these are historical features
    rename_map = {
        'prediction_bias_5d': 'hist_pred_bias_5d',
        'prediction_bias_15d': 'hist_pred_bias_15d',
        'direction_accuracy_5d': 'hist_dir_acc_5d',
        'direction_accuracy_15d': 'hist_dir_acc_15d',
        'avg_abs_error_5d': 'hist_avg_error_5d',
        'avg_abs_error_15d': 'hist_avg_error_15d'
    }
    
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Fill NaN with neutral values (0 bias, 0.5 direction accuracy)
    fill_values = {
        'hist_pred_bias_5d': 0.0,
        'hist_pred_bias_15d': 0.0,
        'hist_dir_acc_5d': 0.5,
        'hist_dir_acc_15d': 0.5,
        'hist_avg_error_5d': 0.02,  # Default 2% error
        'hist_avg_error_15d': 0.03   # Default 3% error
    }
    
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    
    logger.info(f"Added feedback features. Stocks with feedback: {(df['hist_pred_bias_5d'] != 0).sum()}")
    
    return df


def compute_feedback_sample_weights(
    df: pd.DataFrame,
    feedback: pd.DataFrame = None,
    error_weight_factor: float = 2.0
) -> np.ndarray:
    """
    Compute sample weights based on past prediction errors.
    
    Stocks where we made larger errors get higher weights so the model
    pays more attention to learning their patterns.
    
    Args:
        df: Training data with 'symbol' column
        feedback: Feedback dataset with prediction errors
        error_weight_factor: How much to boost weights for high-error stocks
        
    Returns:
        Array of sample weights
    """
    if feedback is None:
        if FEEDBACK_FILE.exists():
            feedback = pd.read_parquet(FEEDBACK_FILE)
        else:
            return np.ones(len(df))
    
    if feedback.empty:
        return np.ones(len(df))
    
    # Compute average absolute error per stock
    stock_errors = feedback.groupby('symbol')['absolute_error'].mean().to_dict()
    
    # Compute global average error
    global_avg_error = feedback['absolute_error'].mean()
    
    # Create weights: higher error = higher weight
    weights = np.ones(len(df))
    
    for i, symbol in enumerate(df['symbol'].values):
        symbol_str = str(symbol)
        if symbol_str in stock_errors:
            error = stock_errors[symbol_str]
            # Weight = 1 + (error / global_avg - 1) * factor
            # If error is 2x average, weight becomes 1 + (2-1)*factor = 1 + factor
            relative_error = error / global_avg_error if global_avg_error > 0 else 1.0
            weights[i] = 1.0 + (relative_error - 1.0) * error_weight_factor
    
    # Clip weights to reasonable range
    weights = np.clip(weights, 0.5, 5.0)
    
    logger.info(f"Feedback weights - Mean: {weights.mean():.2f}, Min: {weights.min():.2f}, Max: {weights.max():.2f}")
    logger.info(f"  Stocks with boosted weights (>1.1): {(weights > 1.1).sum()} ({100*(weights > 1.1).mean():.1f}%)")
    
    return weights


def compute_calibration_adjustment(feedback: pd.DataFrame) -> Dict[str, float]:
    """
    Compute global calibration adjustments based on systematic biases.
    
    If the model consistently over-predicts by 1%, we can subtract 1% from future predictions.
    
    Returns:
        Dictionary with calibration adjustments for 5d and 15d predictions
    """
    if feedback.empty:
        return {'bias_5d': 0.0, 'bias_15d': 0.0, 'scale_5d': 1.0, 'scale_15d': 1.0}
    
    calibration = {}
    
    for horizon in [5, 15]:
        h_fb = feedback[feedback['horizon'] == horizon]
        if len(h_fb) > 10:  # Need enough data points
            # Compute bias (mean prediction error)
            bias = h_fb['prediction_error'].mean()
            
            # Compute scale factor using linear regression
            # actual = scale * predicted + intercept
            pred = h_fb['pred_return'].values
            actual = h_fb['actual_return'].values
            
            # Simple linear regression
            if np.std(pred) > 0:
                correlation = np.corrcoef(pred, actual)[0, 1]
                scale = np.std(actual) / np.std(pred) if np.std(pred) > 0 else 1.0
                scale = np.clip(scale, 0.5, 2.0)  # Limit adjustment range
            else:
                scale = 1.0
            
            calibration[f'bias_{horizon}d'] = bias
            calibration[f'scale_{horizon}d'] = scale
            
            logger.info(f"Calibration {horizon}d - Bias: {bias:.4f}, Scale: {scale:.2f}")
        else:
            calibration[f'bias_{horizon}d'] = 0.0
            calibration[f'scale_{horizon}d'] = 1.0
    
    return calibration


def apply_calibration(predictions: pd.DataFrame, calibration: Dict[str, float] = None) -> pd.DataFrame:
    """
    Apply calibration adjustments to predictions.
    
    Adjusted prediction = (raw_prediction * scale) - bias
    """
    if calibration is None:
        # Load from feedback
        feedback = pd.read_parquet(FEEDBACK_FILE) if FEEDBACK_FILE.exists() else pd.DataFrame()
        calibration = compute_calibration_adjustment(feedback)
    
    predictions = predictions.copy()
    
    # Apply to 5-day predictions
    if 'pred_ret_5' in predictions.columns:
        scale = calibration.get('scale_5d', 1.0)
        bias = calibration.get('bias_5d', 0.0)
        predictions['pred_ret_5_raw'] = predictions['pred_ret_5']
        predictions['pred_ret_5'] = predictions['pred_ret_5'] * scale - bias
        logger.info(f"Applied 5d calibration: scale={scale:.2f}, bias={bias:.4f}")
    
    # Apply to 15-day predictions
    if 'pred_ret_15' in predictions.columns:
        scale = calibration.get('scale_15d', 1.0)
        bias = calibration.get('bias_15d', 0.0)
        predictions['pred_ret_15_raw'] = predictions['pred_ret_15']
        predictions['pred_ret_15'] = predictions['pred_ret_15'] * scale - bias
        logger.info(f"Applied 15d calibration: scale={scale:.2f}, bias={bias:.4f}")
    
    return predictions


def generate_feedback_report() -> str:
    """Generate a human-readable feedback report"""
    feedback = build_feedback_dataset()
    
    if feedback.empty:
        return "No feedback data available yet. Run predictions for several days to build feedback."
    
    report_lines = [
        "=" * 60,
        "PREDICTION FEEDBACK REPORT",
        "=" * 60,
        "",
        f"Total predictions tracked: {len(feedback)}",
        f"Date range: {feedback['date'].min().date()} to {feedback['date'].max().date()}",
        "",
        "--- 5-Day Predictions ---"
    ]
    
    fb_5d = feedback[feedback['horizon'] == 5]
    if len(fb_5d) > 0:
        report_lines.extend([
            f"  Predictions: {len(fb_5d)}",
            f"  Mean Absolute Error: {fb_5d['absolute_error'].mean():.2%}",
            f"  Direction Accuracy: {fb_5d['correct_direction'].mean():.1%}",
            f"  Avg Predicted Return: {fb_5d['pred_return'].mean():.2%}",
            f"  Avg Actual Return: {fb_5d['actual_return'].mean():.2%}",
            f"  Bias (pred - actual): {fb_5d['prediction_error'].mean():.2%}",
        ])
    
    report_lines.append("")
    report_lines.append("--- 15-Day Predictions ---")
    
    fb_15d = feedback[feedback['horizon'] == 15]
    if len(fb_15d) > 0:
        report_lines.extend([
            f"  Predictions: {len(fb_15d)}",
            f"  Mean Absolute Error: {fb_15d['absolute_error'].mean():.2%}",
            f"  Direction Accuracy: {fb_15d['correct_direction'].mean():.1%}",
            f"  Avg Predicted Return: {fb_15d['pred_return'].mean():.2%}",
            f"  Avg Actual Return: {fb_15d['actual_return'].mean():.2%}",
            f"  Bias (pred - actual): {fb_15d['prediction_error'].mean():.2%}",
        ])
    
    # Top stocks by error
    report_lines.append("")
    report_lines.append("--- Stocks with Highest Errors (need more attention) ---")
    
    stock_errors = feedback.groupby('symbol')['absolute_error'].mean().sort_values(ascending=False).head(10)
    for symbol, error in stock_errors.items():
        report_lines.append(f"  {symbol}: {error:.2%} avg error")
    
    # Best performing predictions
    report_lines.append("")
    report_lines.append("--- Most Accurate Predictions ---")
    
    stock_errors_best = feedback.groupby('symbol')['absolute_error'].mean().sort_values().head(10)
    for symbol, error in stock_errors_best.items():
        report_lines.append(f"  {symbol}: {error:.2%} avg error")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    # Build feedback dataset
    feedback = build_feedback_dataset()
    
    if not feedback.empty:
        # Compute stock-level stats
        stats = compute_stock_feedback_stats(feedback)
        
        # Compute calibration
        calibration = compute_calibration_adjustment(feedback)
        
        # Print report
        print(generate_feedback_report())
    else:
        print("No feedback data available yet.")
        print("Run predictions for several days, then run this script to analyze accuracy.")
