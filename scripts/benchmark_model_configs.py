#!/usr/bin/env python3
"""
Benchmark different model configurations to find the optimal balance
between training speed and prediction performance.
"""
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, '.')

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from config import FEATURES_Z_FILE, TARGET_COL, FEATURE_COLS

# Configurations to test
CONFIGS = {
    'ultra_fast': {
        'rf_n_estimators': 10,
        'rf_max_depth': 4,
        'gb_n_estimators': 10,
        'gb_max_depth': 3,
        'max_samples': 200000,
    },
    'fast_ci': {  # Current CI config
        'rf_n_estimators': 20,
        'rf_max_depth': 5,
        'gb_n_estimators': 20,
        'gb_max_depth': 3,
        'max_samples': 500000,
    },
    'balanced': {
        'rf_n_estimators': 50,
        'rf_max_depth': 7,
        'gb_n_estimators': 50,
        'gb_max_depth': 4,
        'max_samples': 1000000,
    },
    'full': {  # Production config
        'rf_n_estimators': 100,
        'rf_max_depth': 10,
        'gb_n_estimators': 100,
        'gb_max_depth': 5,
        'max_samples': None,
    },
}


def prepare_data(df: pd.DataFrame, max_samples: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare train/val split"""
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    clean = df[['date', 'symbol'] + feature_cols + [TARGET_COL]].dropna()
    
    if max_samples and len(clean) > max_samples:
        clean = clean.sample(n=max_samples, random_state=42)
    
    # Split by date
    dates = sorted(clean['date'].unique())
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    val_dates = dates[split_idx:]
    
    train_df = clean[clean['date'].isin(train_dates)]
    val_df = clean[clean['date'].isin(val_dates)]
    
    return train_df, val_df, feature_cols


def create_ranking_labels(df: pd.DataFrame) -> np.ndarray:
    """Create quintile labels for ranking"""
    labels = np.zeros(len(df))
    for date in df['date'].unique():
        mask = df['date'] == date
        day_returns = df.loc[mask, TARGET_COL].values
        try:
            quintiles = pd.qcut(day_returns, q=5, labels=False, duplicates='drop')
            labels[mask.values] = quintiles
        except:
            labels[mask.values] = 2  # middle quintile
    return labels


def evaluate_predictions(val_df: pd.DataFrame, predictions: np.ndarray) -> Dict:
    """Evaluate model predictions"""
    val_df = val_df.copy()
    val_df['pred'] = predictions
    
    metrics = {
        'avg_top10_return': [],
        'direction_accuracy': [],
        'top10_vs_bottom10': [],
    }
    
    for date in val_df['date'].unique():
        day_df = val_df[val_df['date'] == date].copy()
        if len(day_df) < 20:
            continue
            
        day_df = day_df.sort_values('pred', ascending=False)
        
        top10 = day_df.head(10)
        bottom10 = day_df.tail(10)
        
        metrics['avg_top10_return'].append(top10[TARGET_COL].mean())
        metrics['direction_accuracy'].append(
            ((day_df['pred'] > 0) == (day_df[TARGET_COL] > 0)).mean()
        )
        metrics['top10_vs_bottom10'].append(
            top10[TARGET_COL].mean() - bottom10[TARGET_COL].mean()
        )
    
    return {
        'avg_top10_return': np.mean(metrics['avg_top10_return']) * 100,  # percentage
        'direction_accuracy': np.mean(metrics['direction_accuracy']) * 100,
        'top10_vs_bottom10': np.mean(metrics['top10_vs_bottom10']) * 100,
        'positive_days_pct': np.mean([r > 0 for r in metrics['avg_top10_return']]) * 100,
    }


def benchmark_config(name: str, config: Dict, train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Benchmark a single configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Config: {config}")
    print(f"{'='*60}")
    
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values
    
    # Create ranking labels
    train_labels = create_ranking_labels(train_df)
    
    # Train RandomForest Ranker
    print("Training RandomForest Ranker...")
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=config['rf_n_estimators'],
        max_depth=config['rf_max_depth'],
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, train_labels)
    rf_time = time.time() - start
    print(f"  RF training time: {rf_time:.1f}s")
    
    # Train Regressor
    print("Training HistGradientBoostingRegressor...")
    start = time.time()
    gb = HistGradientBoostingRegressor(
        max_iter=config['gb_n_estimators'],
        max_depth=config['gb_max_depth'],
        learning_rate=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_time = time.time() - start
    print(f"  GB training time: {gb_time:.1f}s")
    
    # Generate predictions
    rf_proba = rf.predict_proba(X_val)
    rf_score = np.sum(rf_proba * np.arange(5), axis=1)  # weighted score
    gb_pred = gb.predict(X_val)
    
    # Combine predictions (simple average)
    combined_pred = 0.5 * (rf_score / 4) + 0.5 * gb_pred
    
    # Evaluate
    metrics = evaluate_predictions(val_df, combined_pred)
    metrics['train_time'] = rf_time + gb_time
    metrics['train_samples'] = len(train_df)
    
    return metrics


def main():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_Z_FILE)
    print(f"Total samples: {len(df):,}")
    
    results = {}
    
    for name, config in CONFIGS.items():
        # Prepare data with sampling
        train_df, val_df, feature_cols = prepare_data(df, config['max_samples'])
        print(f"\n{name}: Train={len(train_df):,}, Val={len(val_df):,}")
        
        metrics = benchmark_config(name, config, train_df, val_df, feature_cols)
        results[name] = metrics
    
    # Print comparison table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n{'Config':<15} {'Train Time':>12} {'Samples':>12} {'Top10 Ret%':>12} {'Dir Acc%':>10} {'Spread%':>10} {'Pos Days%':>10}")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['train_time']:>10.1f}s {metrics['train_samples']:>12,} "
              f"{metrics['avg_top10_return']:>11.2f}% {metrics['direction_accuracy']:>9.1f}% "
              f"{metrics['top10_vs_bottom10']:>9.2f}% {metrics['positive_days_pct']:>9.1f}%")
    
    print("-"*80)
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find best value config (performance per training second)
    best_value = None
    best_value_score = 0
    for name, metrics in results.items():
        # Score = top10_return / sqrt(train_time) - balance speed and performance
        value_score = metrics['avg_top10_return'] / np.sqrt(metrics['train_time'])
        if value_score > best_value_score:
            best_value_score = value_score
            best_value = name
    
    print(f"\nBest value (performance/speed): {best_value}")
    print(f"Best accuracy: {max(results.items(), key=lambda x: x[1]['avg_top10_return'])[0]}")
    print(f"Fastest: {min(results.items(), key=lambda x: x[1]['train_time'])[0]}")
    
    # Performance degradation analysis
    if 'full' in results and 'fast_ci' in results:
        full = results['full']
        fast = results['fast_ci']
        print(f"\nFast CI vs Full comparison:")
        print(f"  Speed improvement: {full['train_time']/fast['train_time']:.1f}x faster")
        print(f"  Top10 return delta: {fast['avg_top10_return'] - full['avg_top10_return']:+.2f}%")
        print(f"  Direction accuracy delta: {fast['direction_accuracy'] - full['direction_accuracy']:+.1f}%")
    
    return results


if __name__ == "__main__":
    results = main()
