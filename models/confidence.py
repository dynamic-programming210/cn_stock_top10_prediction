"""
Confidence Interval Estimation for Chinese Stock Predictions

Since features are z-normalized, we use a different approach:
- Use rank-based confidence from the model's prediction strength
- Higher predicted return rank = higher confidence
- Combine with model agreement signal
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple


def add_confidence_intervals(df: pd.DataFrame, 
                            confidence_level: float = 0.90) -> pd.DataFrame:
    """
    Add confidence intervals and confidence scores to predictions.
    
    For z-normalized features, we base confidence on:
    1. Rank position in predictions (higher rank = more confident)
    2. Prediction magnitude relative to peers
    3. Model agreement (if both prob_top10 and pred_ret_5 agree)
    """
    df = df.copy()
    
    # 1. Rank-based confidence: top stocks get higher confidence
    n = len(df)
    if n > 0:
        # Sort by predicted return and assign rank-based confidence
        df['_rank'] = df['pred_ret_5'].rank(ascending=False, method='min')
        # Top stock gets 0.9, bottom gets 0.5
        rank_confidence = 0.9 - (df['_rank'] - 1) / max(n - 1, 1) * 0.4
    else:
        rank_confidence = 0.7
    
    # 2. Prediction strength: how much higher than average
    pred_mean = df['pred_ret_5'].mean()
    pred_std_val = df['pred_ret_5'].std()
    if pred_std_val > 0:
        z_pred = (df['pred_ret_5'] - pred_mean) / pred_std_val
        # Convert z-score to confidence boost: z=2 -> +0.1, z=0 -> 0
        strength_boost = (z_pred.clip(-2, 2) / 20)
    else:
        strength_boost = 0
    
    # 3. Model agreement bonus (if prob_top10 exists)
    agreement_boost = 0
    if 'prob_top10' in df.columns:
        # High probability from classifier + high predicted return = bonus
        prob_rank = df['prob_top10'].rank(ascending=False, method='min', pct=True)
        ret_rank = df['pred_ret_5'].rank(ascending=False, method='min', pct=True)
        # If both ranks are in top 50%, give a boost
        agreement_boost = ((prob_rank < 0.5) & (ret_rank < 0.5)).astype(float) * 0.05
    
    # Combine confidence components
    df['confidence_score'] = rank_confidence + strength_boost + agreement_boost
    df['confidence_score'] = df['confidence_score'].clip(0.50, 0.95)
    
    # Clean up temporary column
    if '_rank' in df.columns:
        df = df.drop(columns=['_rank'])
    
    # Calculate confidence intervals based on historical prediction error
    # Use a fixed estimate of prediction std (typical for stock predictions)
    base_std = 0.03  # ~3% typical prediction error for 5-day returns
    
    # Stocks with higher volatility (z-score) have wider intervals
    if 'vol_5' in df.columns:
        # vol_5 is z-normalized: higher z-score = more volatile = wider interval
        vol_factor = 1 + df['vol_5'].clip(-2, 2) * 0.1  # +/-20% adjustment
    else:
        vol_factor = 1.0
    
    df['pred_std'] = base_std * vol_factor
    df['pred_std'] = df['pred_std'].clip(0.02, 0.06)
    
    # Calculate confidence intervals
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    df['pred_lower'] = df['pred_ret_5'] - z_score * df['pred_std']
    df['pred_upper'] = df['pred_ret_5'] + z_score * df['pred_std']
    
    return df
