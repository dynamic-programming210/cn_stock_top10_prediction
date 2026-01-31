"""
Confidence Interval Estimation for Chinese Stock Predictions
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple


def add_confidence_intervals(df: pd.DataFrame, 
                            confidence_level: float = 0.90) -> pd.DataFrame:
    """Add confidence intervals to predictions"""
    df = df.copy()
    
    # Estimate prediction uncertainty based on:
    # 1. Volatility of the stock
    # 2. Magnitude of prediction
    # 3. Model uncertainty (approximated)
    
    # Base uncertainty from volatility
    if 'vol_5' in df.columns:
        vol_component = df['vol_5'].abs() * 0.3
    else:
        vol_component = 0.05
    
    # Prediction magnitude uncertainty (larger predictions = more uncertainty)
    pred_component = df['pred_ret_5'].abs() * 0.5
    
    # Base model uncertainty
    base_uncertainty = 0.02
    
    # Combined standard deviation estimate
    df['pred_std'] = np.sqrt(vol_component**2 + pred_component**2 + base_uncertainty**2)
    df['pred_std'] = df['pred_std'].clip(0.01, 0.15)  # Reasonable bounds
    
    # Calculate confidence intervals
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    df['pred_lower'] = df['pred_ret_5'] - z_score * df['pred_std']
    df['pred_upper'] = df['pred_ret_5'] + z_score * df['pred_std']
    
    # Confidence score (inverse of relative uncertainty)
    df['confidence_score'] = 1 - (df['pred_std'] / (df['pred_ret_5'].abs() + 0.01)).clip(0, 1)
    df['confidence_score'] = df['confidence_score'].clip(0.3, 0.95)
    
    return df
