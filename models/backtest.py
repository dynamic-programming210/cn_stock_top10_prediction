"""
Backtesting Module for Chinese Stock Top-10 Predictor
Walk-forward validation with realistic simulation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURES_Z_FILE, FEATURE_COLS, TARGET_COL, TOP_K,
    OUTPUTS_DIR
)
from models.train import TwoStageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """Walk-forward backtesting for stock ranking model"""
    
    def __init__(self, 
                 train_window: int = 252,  # ~1 year training
                 test_window: int = 21,    # ~1 month test  
                 min_train_samples: int = 5000,
                 top_k: int = TOP_K,
                 max_train_symbols: int = None,  # Limit symbols for faster training
                 fast_mode: bool = False):  # Use faster model settings
        self.train_window = train_window
        self.test_window = test_window
        self.min_train_samples = min_train_samples
        self.top_k = top_k
        self.max_train_symbols = max_train_symbols
        self.fast_mode = fast_mode
        self.results = []
        
    def run_backtest(self, 
                     df: pd.DataFrame,
                     start_date: str = None,
                     end_date: str = None,
                     verbose: bool = True) -> pd.DataFrame:
        """
        Run walk-forward backtest
        
        Args:
            df: DataFrame with features and target
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            verbose: Print progress
            
        Returns:
            DataFrame with daily backtest results
        """
        # Prepare data
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        # Get unique dates
        dates = sorted(df['date'].unique())
        
        # Apply date filters
        if start_date:
            start_date = pd.to_datetime(start_date)
            dates = [d for d in dates if d >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            dates = [d for d in dates if d <= end_date]
        
        # Need enough history for training
        if len(dates) < self.train_window + self.test_window:
            raise ValueError(f"Not enough dates for backtest. Have {len(dates)}, need {self.train_window + self.test_window}")
        
        logger.info(f"Running backtest from {dates[self.train_window]} to {dates[-1]}")
        logger.info(f"Train window: {self.train_window} days, Test window: {self.test_window} days")
        
        self.results = []
        
        # Walk-forward loop
        test_start_idx = self.train_window
        n_folds = 0
        
        while test_start_idx < len(dates):
            test_end_idx = min(test_start_idx + self.test_window, len(dates))
            
            # Define train/test date ranges
            train_dates = dates[test_start_idx - self.train_window:test_start_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            
            if len(test_dates) == 0:
                break
                
            # Get train/test data
            train_df = df[df['date'].isin(train_dates)].copy()
            test_df = df[df['date'].isin(test_dates)].copy()
            
            # Sample training symbols if too many (for speed)
            if self.max_train_symbols and train_df['symbol'].nunique() > self.max_train_symbols:
                sample_symbols = train_df['symbol'].unique()
                np.random.seed(n_folds)  # Reproducible
                sample_symbols = np.random.choice(sample_symbols, self.max_train_symbols, replace=False)
                train_df = train_df[train_df['symbol'].isin(sample_symbols)]
            
            # Check minimum samples
            if len(train_df) < self.min_train_samples:
                logger.warning(f"Fold {n_folds}: Only {len(train_df)} train samples, skipping")
                test_start_idx += self.test_window
                continue
            
            if verbose and n_folds % 3 == 0:
                logger.info(f"Fold {n_folds}: Train {train_dates[0].date()} to {train_dates[-1].date()}, "
                           f"Test {test_dates[0].date()} to {test_dates[-1].date()}")
            
            # Train model
            model = TwoStageModel()
            try:
                model.train(train_df, validation_split=0.15, fast_mode=self.fast_mode)
            except Exception as e:
                logger.warning(f"Fold {n_folds}: Training failed: {e}")
                test_start_idx += self.test_window
                continue
            
            # Generate predictions for each test date
            for test_date in test_dates:
                day_df = test_df[test_df['date'] == test_date].copy()
                
                if len(day_df) < self.top_k:
                    continue
                
                # Filter out rows with NaN in feature columns before prediction
                feature_cols = model.feature_cols if model.feature_cols else FEATURE_COLS
                valid_features = [c for c in feature_cols if c in day_df.columns]
                day_df = day_df.dropna(subset=valid_features)
                
                if len(day_df) < self.top_k:
                    continue
                
                # Get top-k predictions
                pred_df = model.predict(day_df)
                # Filter out any rows where prediction failed (NaN rank_score)
                pred_df = pred_df.dropna(subset=['rank_score'])
                
                if len(pred_df) < self.top_k:
                    continue
                    
                top_k = pred_df.nlargest(self.top_k, 'rank_score')
                
                # Calculate metrics for this day
                result = self._evaluate_day(top_k, test_date)
                self.results.append(result)
            
            n_folds += 1
            test_start_idx += self.test_window
        
        logger.info(f"Backtest complete: {n_folds} folds, {len(self.results)} trading days")
        
        # Compile results
        results_df = pd.DataFrame(self.results)
        return results_df
    
    def _evaluate_day(self, top_k: pd.DataFrame, date: pd.Timestamp) -> Dict:
        """Evaluate predictions for a single day"""
        
        # Actual returns of top-k stocks
        actual_returns = top_k[TARGET_COL].values
        pred_returns = top_k['pred_ret_5'].values
        
        # Metrics
        avg_return = np.mean(actual_returns) if len(actual_returns) > 0 else 0
        median_return = np.median(actual_returns) if len(actual_returns) > 0 else 0
        
        # Hit rate (% of positive returns)
        hit_rate = np.mean(actual_returns > 0) if len(actual_returns) > 0 else 0
        
        # Direction accuracy
        direction_correct = np.sum((pred_returns > 0) == (actual_returns > 0))
        direction_accuracy = direction_correct / len(actual_returns) if len(actual_returns) > 0 else 0
        
        # Rank correlation (suppress warning when predictions are constant)
        if len(actual_returns) > 1:
            from scipy.stats import spearmanr
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                warnings.filterwarnings('ignore', message='An input array is constant')
                rank_corr, _ = spearmanr(pred_returns, actual_returns)
                if np.isnan(rank_corr):
                    rank_corr = 0
        else:
            rank_corr = 0
        
        return {
            'date': date,
            'n_stocks': len(top_k),
            'avg_return': avg_return,
            'median_return': median_return,
            'hit_rate': hit_rate,
            'direction_accuracy': direction_accuracy,
            'rank_correlation': rank_corr,
            'best_return': np.max(actual_returns) if len(actual_returns) > 0 else 0,
            'worst_return': np.min(actual_returns) if len(actual_returns) > 0 else 0,
            'symbols': ','.join(top_k['symbol'].tolist()),
        }
    
    def compute_summary_stats(self, results_df: pd.DataFrame = None) -> Dict:
        """Compute summary statistics from backtest results"""
        
        if results_df is None:
            results_df = pd.DataFrame(self.results)
        
        if results_df.empty:
            return {}
        
        # Cumulative returns (assuming daily rebalance)
        results_df = results_df.sort_values('date')
        results_df['cum_return'] = (1 + results_df['avg_return']).cumprod() - 1
        
        # Annualized metrics
        n_days = len(results_df)
        total_return = results_df['cum_return'].iloc[-1]
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        
        # Volatility
        daily_vol = results_df['avg_return'].std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio (assuming 3% risk-free rate for China)
        rf_daily = 0.03 / 252
        excess_returns = results_df['avg_return'] - rf_daily
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Max drawdown
        cum_max = (1 + results_df['cum_return']).cummax()
        drawdown = (1 + results_df['cum_return']) / cum_max - 1
        max_drawdown = drawdown.min()
        
        # Win rate metrics
        positive_days = (results_df['avg_return'] > 0).sum()
        negative_days = (results_df['avg_return'] < 0).sum()
        win_rate = positive_days / n_days if n_days > 0 else 0
        
        # Average win/loss
        wins = results_df[results_df['avg_return'] > 0]['avg_return']
        losses = results_df[results_df['avg_return'] < 0]['avg_return']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')
        
        return {
            'start_date': results_df['date'].min().strftime('%Y-%m-%d'),
            'end_date': results_df['date'].max().strftime('%Y-%m-%d'),
            'n_trading_days': n_days,
            'total_return': f"{total_return:.2%}",
            'annualized_return': f"{annualized_return:.2%}",
            'annualized_volatility': f"{annualized_vol:.2%}",
            'sharpe_ratio': f"{sharpe:.2f}",
            'max_drawdown': f"{max_drawdown:.2%}",
            'win_rate': f"{win_rate:.2%}",
            'avg_win': f"{avg_win:.2%}",
            'avg_loss': f"{avg_loss:.2%}",
            'profit_factor': f"{profit_factor:.2f}",
            'avg_hit_rate': f"{results_df['hit_rate'].mean():.2%}",
            'avg_direction_accuracy': f"{results_df['direction_accuracy'].mean():.2%}",
            'avg_rank_correlation': f"{results_df['rank_correlation'].mean():.3f}",
        }
    
    def save_results(self, results_df: pd.DataFrame, filepath: str = None):
        """Save backtest results"""
        if filepath is None:
            filepath = OUTPUTS_DIR / "backtest_results.parquet"
        
        results_df.to_parquet(filepath, index=False)
        logger.info(f"Saved backtest results to {filepath}")
        
        # Also save summary
        summary = self.compute_summary_stats(results_df)
        summary_path = OUTPUTS_DIR / "backtest_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved backtest summary to {summary_path}")


def run_backtest(features_file: str = None,
                 train_window: int = 252,
                 test_window: int = 21,
                 start_date: str = None,
                 end_date: str = None,
                 max_train_symbols: int = None,
                 fast_mode: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full backtest and return results
    
    Args:
        features_file: Path to features file
        train_window: Number of days for training
        test_window: Number of days for each test fold
        start_date: Backtest start date
        end_date: Backtest end date
        max_train_symbols: Max symbols for training (speeds up backtest)
        fast_mode: Use faster model settings
        
    Returns:
        Tuple of (results_df, summary_dict)
    """
    # Load data
    features_file = features_file or FEATURES_Z_FILE
    logger.info(f"Loading features from {features_file}")
    df = pd.read_parquet(features_file)
    
    # Run backtest
    backtester = Backtester(
        train_window=train_window,
        test_window=test_window,
        max_train_symbols=max_train_symbols,
        fast_mode=fast_mode
    )
    
    results_df = backtester.run_backtest(
        df, 
        start_date=start_date,
        end_date=end_date
    )
    
    # Compute summary
    summary = backtester.compute_summary_stats(results_df)
    
    # Save results
    backtester.save_results(results_df)
    
    return results_df, summary


def print_backtest_summary(summary: Dict):
    """Pretty print backtest summary"""
    print("\n" + "="*60)
    print("           BACKTEST SUMMARY")
    print("="*60)
    print(f"Period: {summary.get('start_date', 'N/A')} to {summary.get('end_date', 'N/A')}")
    print(f"Trading Days: {summary.get('n_trading_days', 0)}")
    print("-"*60)
    print(f"Total Return:        {summary.get('total_return', 'N/A')}")
    print(f"Annualized Return:   {summary.get('annualized_return', 'N/A')}")
    print(f"Annualized Vol:      {summary.get('annualized_volatility', 'N/A')}")
    print(f"Sharpe Ratio:        {summary.get('sharpe_ratio', 'N/A')}")
    print(f"Max Drawdown:        {summary.get('max_drawdown', 'N/A')}")
    print("-"*60)
    print(f"Win Rate:            {summary.get('win_rate', 'N/A')}")
    print(f"Avg Win:             {summary.get('avg_win', 'N/A')}")
    print(f"Avg Loss:            {summary.get('avg_loss', 'N/A')}")
    print(f"Profit Factor:       {summary.get('profit_factor', 'N/A')}")
    print("-"*60)
    print(f"Avg Hit Rate:        {summary.get('avg_hit_rate', 'N/A')}")
    print(f"Avg Direction Acc:   {summary.get('avg_direction_accuracy', 'N/A')}")
    print(f"Avg Rank Corr:       {summary.get('avg_rank_correlation', 'N/A')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run backtest for CN stock predictor")
    parser.add_argument('--train-window', type=int, default=252, help="Training window in days")
    parser.add_argument('--test-window', type=int, default=21, help="Test window in days")
    parser.add_argument('--start-date', type=str, default=None, help="Backtest start date")
    parser.add_argument('--end-date', type=str, default=None, help="Backtest end date")
    parser.add_argument('--features', type=str, default=None, help="Path to features file")
    parser.add_argument('--max-symbols', type=int, default=None, help="Max symbols to use for training (faster)")
    parser.add_argument('--fast', action='store_true', help="Use fast mode (fewer trees, shallower)")
    
    args = parser.parse_args()
    
    results_df, summary = run_backtest(
        features_file=args.features,
        train_window=args.train_window,
        test_window=args.test_window,
        start_date=args.start_date,
        end_date=args.end_date,
        max_train_symbols=args.max_symbols,
        fast_mode=args.fast
    )
    
    print_backtest_summary(summary)
