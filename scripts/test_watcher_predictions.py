#!/usr/bin/env python3
"""
Test script to verify stock watcher predictions are accurate.

This tests:
1. Features are available for watched stocks
2. Predictions are different for different stocks (not all same value)
3. Predictions match what the model would generate
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from config import (
    BARS_FILE, FEATURES_Z_FILE, FEATURE_COLS,
    RANKER_MODEL_FILE, REGRESSOR_MODEL_FILE,
    RANKER_15D_MODEL_FILE, REGRESSOR_15D_MODEL_FILE,
    WATCHER_OUTPUT_FILE, WATCHER_INPUT_FILE
)
from features.build_features import build_features

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


def test_features_available():
    """Test that features are available for watched stocks"""
    print("\n=== Test 1: Features Available for Watched Stocks ===")
    
    symbols = read_watch_list()
    if not symbols:
        print("❌ No watch list found")
        return False
    print(f"Watch list: {symbols}")
    
    # Load bars and build features
    print("Loading bars...")
    bars = pd.read_parquet(BARS_FILE)
    latest_bar_date = bars['date'].max()
    print(f"Bars latest date: {latest_bar_date}")
    
    # Check if symbols are in bars
    missing_in_bars = []
    for sym in symbols:
        if sym not in bars['symbol'].unique():
            missing_in_bars.append(sym)
    
    if missing_in_bars:
        print(f"❌ Symbols not in bars: {missing_in_bars}")
        return False
    print(f"✅ All {len(symbols)} symbols found in bars")
    
    # Build features for watched stocks
    print("Building features for watched stocks...")
    sample_bars = bars[bars['symbol'].isin(symbols)].copy()
    features = build_features(sample_bars, save=False, zscore=True)
    latest_feat_date = features['date'].max()
    print(f"Features latest date: {latest_feat_date}")
    
    # Check if features are available for all symbols on latest date
    missing_features = []
    for sym in symbols:
        matches = features[(features['symbol'] == sym) & (features['date'] == latest_feat_date)]
        if len(matches) == 0:
            missing_features.append(sym)
    
    if missing_features:
        print(f"❌ Symbols missing features on {latest_feat_date}: {missing_features}")
        return False
    print(f"✅ All {len(symbols)} symbols have features on {latest_feat_date}")
    return True


def test_predictions_vary():
    """Test that predictions are different for different stocks"""
    print("\n=== Test 2: Predictions Vary by Stock ===")
    
    if not WATCHER_OUTPUT_FILE.exists():
        print("❌ Watcher output file not found")
        return False
    
    df = pd.read_parquet(WATCHER_OUTPUT_FILE)
    print(f"Watcher predictions: {len(df)} stocks")
    
    # Check for duplicate pred_ret_5 values
    pred_5_counts = df['pred_ret_5'].value_counts()
    print(f"\npred_ret_5 value distribution:")
    for val, count in pred_5_counts.items():
        print(f"  {val:.6f}: {count} stocks")
    
    pred_15_counts = df['pred_ret_15'].value_counts()
    print(f"\npred_ret_15 value distribution:")
    for val, count in pred_15_counts.items():
        print(f"  {val:.6f}: {count} stocks")
    
    # Flag if too many stocks have the same prediction
    max_same_pred = max(pred_5_counts.max(), pred_15_counts.max())
    if max_same_pred > len(df) / 2:
        print(f"\n❌ Too many stocks ({max_same_pred}/{len(df)}) have same prediction - likely a bug!")
        return False
    
    print(f"\n✅ Predictions vary appropriately")
    return True


def test_predictions_match_model():
    """Test that predictions match what the model would generate"""
    print("\n=== Test 3: Predictions Match Model Output ===")
    
    symbols = read_watch_list()
    if not symbols:
        print("❌ No watch list found")
        return False
    
    # Load bars and build features
    bars = pd.read_parquet(BARS_FILE)
    sample_bars = bars[bars['symbol'].isin(symbols)].copy()
    features = build_features(sample_bars, save=False, zscore=True)
    latest_date = features['date'].max()
    
    # Load models
    with open(REGRESSOR_MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
        model_5d = data['model'] if isinstance(data, dict) and 'model' in data else data
    
    with open(REGRESSOR_15D_MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
        model_15d = data['model'] if isinstance(data, dict) and 'model' in data else data
    
    # Load watcher output
    watcher_df = pd.read_parquet(WATCHER_OUTPUT_FILE)
    
    print(f"\nComparing predictions for {len(symbols)} stocks:")
    print(f"{'Symbol':<10} {'Watcher 5d':>12} {'Model 5d':>12} {'Match':>8} {'Watcher 15d':>12} {'Model 15d':>12} {'Match':>8}")
    print("-" * 80)
    
    all_match = True
    for sym in symbols:
        # Get features
        feat_row = features[(features['symbol'] == sym) & (features['date'] == latest_date)]
        if feat_row.empty:
            print(f"{sym:<10} {'NO FEATURES':<50}")
            continue
        
        available_cols = [c for c in FEATURE_COLS if c in features.columns]
        X = feat_row[available_cols].values
        
        # Model predictions
        model_pred_5 = float(model_5d.predict(X)[0])
        model_pred_15 = float(model_15d.predict(X)[0])
        
        # Watcher predictions
        watcher_row = watcher_df[watcher_df['symbol'] == sym]
        if watcher_row.empty:
            print(f"{sym:<10} {'NOT IN WATCHER':<50}")
            continue
        
        watcher_pred_5 = watcher_row['pred_ret_5'].iloc[0]
        watcher_pred_15 = watcher_row['pred_ret_15'].iloc[0]
        
        match_5 = abs(watcher_pred_5 - model_pred_5) < 0.0001
        match_15 = abs(watcher_pred_15 - model_pred_15) < 0.0001
        
        print(f"{sym:<10} {watcher_pred_5:>12.6f} {model_pred_5:>12.6f} {'✅' if match_5 else '❌':>8} "
              f"{watcher_pred_15:>12.6f} {model_pred_15:>12.6f} {'✅' if match_15 else '❌':>8}")
        
        if not match_5 or not match_15:
            all_match = False
    
    if all_match:
        print(f"\n✅ All predictions match model output")
    else:
        print(f"\n❌ Some predictions don't match - investigate stock_watcher.py")
    
    return all_match


def main():
    print("=" * 60)
    print("Stock Watcher Prediction Verification")
    print("=" * 60)
    
    results = []
    
    # Test 1: Features available
    results.append(("Features Available", test_features_available()))
    
    # Test 2: Predictions vary
    results.append(("Predictions Vary", test_predictions_vary()))
    
    # Test 3: Predictions match model
    results.append(("Predictions Match Model", test_predictions_match_model()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
