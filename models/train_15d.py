"""
Two-Stage Model for Chinese Stock Top-10 Prediction - 15-Day Horizon
Stage 1: RandomForest Classifier - ranks stocks by likelihood to outperform
Stage 2: GradientBoosting Regressor - predicts actual 15-day returns

This is a parallel model to the 5-day predictor, optimized for medium-term returns.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import List, Tuple, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURES_Z_FILE, FEATURE_COLS, TARGET_COL_15D,
    RANKER_15D_MODEL_FILE, REGRESSOR_15D_MODEL_FILE,
    TOP_K, MIN_RANK_SCORE, MAX_STOCKS_PER_SECTOR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageModel15D:
    """Two-stage model for 15-day predictions: Ranker (RandomForest) + Regressor (GradientBoosting)"""
    
    def __init__(self):
        self.ranker = None
        self.regressor = None
        self.feature_cols = None
        
    def _prepare_ranking_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for RandomForest ranker with sample weights
        
        Returns:
            X: Features
            y: Relevance labels (0-4 quintiles)
            sample_weights: Higher weights for stocks with trend initiation signals
        """
        # Create relevance labels based on forward returns
        # Group stocks into quintiles by date
        df = df.copy()
        df['relevance'] = df.groupby('date')[TARGET_COL_15D].transform(
            lambda x: pd.qcut(x.rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
        ).astype(int)
        
        # Get features
        X = df[self.feature_cols].values
        y = df['relevance'].values
        
        # Create sample weights - give higher weight to samples with trend initiation signals
        sample_weights = np.ones(len(df))
        
        # Boost weight for stocks with trend initiation signals
        if 'trend_initiation_score' in df.columns:
            high_ti = df['trend_initiation_score'] > df['trend_initiation_score'].quantile(0.8)
            sample_weights[high_ti.values] *= 2.0
        
        # Boost weight for golden cross signals (important for 15-day horizon)
        if 'golden_cross_10_60' in df.columns:
            gc_signal = df['golden_cross_10_60'] == 1
            sample_weights[gc_signal.values] *= 1.8  # Stronger weight for 15d
        
        # Boost weight for breakout signals
        if 'breakout_with_volume' in df.columns:
            breakout = df['breakout_with_volume'] == 1
            sample_weights[breakout.values] *= 1.5
        
        # Boost weight for MA bullish alignment (important for medium-term)
        if 'ma_bullish_align' in df.columns:
            ma_align = df['ma_bullish_align'] > df['ma_bullish_align'].quantile(0.75)
            sample_weights[ma_align.values] *= 1.7
        
        # Boost weight for strong stocks with positive outcomes
        if 'strong_stock_score' in df.columns:
            strong_positive = (df['strong_stock_score'] > df['strong_stock_score'].quantile(0.8)) & (df[TARGET_COL_15D] > 0)
            sample_weights[strong_positive.values] *= 1.5
        
        return X, y, sample_weights
    
    def train(self, df: pd.DataFrame, 
              feature_cols: List[str] = None,
              validation_split: float = 0.2,
              fast_mode: bool = False) -> dict:
        """Train both ranker and regressor for 15-day horizon"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
        
        # Model parameters based on fast_mode
        if fast_mode:
            rf_n_estimators = 30
            rf_max_depth = 6
            gb_n_estimators = 30
            gb_max_depth = 3
            logger.info("FAST MODE: Using reduced model complexity")
        else:
            rf_n_estimators = 100
            rf_max_depth = 10
            gb_n_estimators = 100
            gb_max_depth = 5
        
        # Set feature columns
        self.feature_cols = feature_cols or [c for c in FEATURE_COLS if c in df.columns]
        
        # Ensure we have the target
        if TARGET_COL_15D not in df.columns:
            raise ValueError(f"Target column {TARGET_COL_15D} not found in data")
        
        # Drop rows with NaN
        clean = df[['date', 'symbol'] + self.feature_cols + [TARGET_COL_15D]].dropna()
        logger.info(f"Training 15D model on {len(clean)} samples, {clean['symbol'].nunique()} symbols")
        
        # Split by date
        dates = sorted(clean['date'].unique())
        split_idx = int(len(dates) * (1 - validation_split))
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]
        
        train_df = clean[clean['date'].isin(train_dates)]
        val_df = clean[clean['date'].isin(val_dates)]
        
        logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
        
        # Stage 1: Train Ranker (RandomForest Classifier)
        logger.info("Training Stage 1: RandomForest Ranker for 15-day...")
        
        X_train, y_train, train_weights = self._prepare_ranking_data(train_df)
        X_val, y_val, _ = self._prepare_ranking_data(val_df)
        
        logger.info(f"Sample weights - Mean: {train_weights.mean():.2f}, Max: {train_weights.max():.2f}, "
                    f"Boosted samples: {(train_weights > 1).sum()} ({100*(train_weights > 1).mean():.1f}%)")
        
        self.ranker = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        )
        self.ranker.fit(X_train, y_train, sample_weight=train_weights)
        
        train_acc = self.ranker.score(X_train, y_train)
        val_acc = self.ranker.score(X_val, y_val)
        logger.info(f"Ranker accuracy - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        
        # Stage 2: Train Regressor
        logger.info("Training Stage 2: GradientBoostingRegressor for 15-day...")
        
        X_reg_train = train_df[self.feature_cols].values
        y_reg_train = train_df[TARGET_COL_15D].values
        X_reg_val = val_df[self.feature_cols].values
        y_reg_val = val_df[TARGET_COL_15D].values
        
        self.regressor = GradientBoostingRegressor(
            n_estimators=gb_n_estimators,
            max_depth=gb_max_depth,
            learning_rate=0.05,
            random_state=42
        )
        self.regressor.fit(X_reg_train, y_reg_train, sample_weight=train_weights)
        
        # Evaluate
        metrics = self._evaluate(val_df)
        
        logger.info(f"Validation metrics (15D): {metrics}")
        
        return metrics
    
    def _evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate model on validation data"""
        df = df.copy()
        
        # Filter out rows with NaN in features
        mask_valid = df[self.feature_cols].notna().all(axis=1)
        df = df[mask_valid].copy()
        
        if df.empty:
            return {'avg_top10_return': 0, 'avg_direction_accuracy': 0, 'positive_days_pct': 0}
        
        # Get predictions
        X = df[self.feature_cols].values
        
        # Get rank score (probability of being in top quintile)
        proba = self.ranker.predict_proba(X)
        weights = np.arange(proba.shape[1])
        df['rank_score'] = (proba * weights).sum(axis=1)
        
        df['pred_ret_15'] = self.regressor.predict(X)
        
        # Calculate metrics by date
        results = []
        for date, group in df.groupby('date'):
            top10 = group.nlargest(10, 'rank_score')
            
            # Average actual return of top 10
            avg_ret = top10[TARGET_COL_15D].mean()
            
            # Direction accuracy
            direction_acc = ((top10['pred_ret_15'] > 0) == (top10[TARGET_COL_15D] > 0)).mean()
            
            results.append({
                'date': date,
                'avg_ret': avg_ret,
                'direction_acc': direction_acc
            })
        
        results_df = pd.DataFrame(results)
        
        return {
            'avg_top10_return': results_df['avg_ret'].mean(),
            'avg_direction_accuracy': results_df['direction_acc'].mean(),
            'positive_days_pct': (results_df['avg_ret'] > 0).mean()
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict rankings and returns for 15-day horizon"""
        if self.ranker is None or self.regressor is None:
            raise ValueError("Model not trained. Call train() first or load().")
        
        df = df.copy()
        
        # Ensure feature columns are present
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        
        # Handle NaN values
        feature_cols_present = [c for c in self.feature_cols if c in df.columns]
        mask_valid = df[feature_cols_present].notna().all(axis=1)
        
        # Initialize output columns with NaN
        df['rank_score_15d'] = np.nan
        df['pred_ret_15'] = np.nan
        
        if mask_valid.sum() == 0:
            logger.warning("No valid rows for prediction (all have NaN)")
            return df
        
        X = df.loc[mask_valid, self.feature_cols].values
        
        # Get rank score (weighted probability)
        proba = self.ranker.predict_proba(X)
        weights = np.arange(proba.shape[1])
        df.loc[mask_valid, 'rank_score_15d'] = (proba * weights).sum(axis=1)
        
        df.loc[mask_valid, 'pred_ret_15'] = self.regressor.predict(X)
        
        return df
    
    def select_top10(self, df: pd.DataFrame, date: str = None, 
                     max_per_sector: int = None) -> pd.DataFrame:
        """Select top 10 stocks based on rank score with sector diversification."""
        if max_per_sector is None:
            max_per_sector = MAX_STOCKS_PER_SECTOR
            
        if date is None:
            date = df['date'].max()
        
        day_df = df[df['date'] == date].copy()
        
        if day_df.empty:
            logger.warning(f"No data for date {date}")
            return pd.DataFrame()
        
        # Predict if not already done
        if 'rank_score_15d' not in day_df.columns:
            day_df = self.predict(day_df)
        
        # Add sector if not present
        if 'sector' not in day_df.columns:
            try:
                from data.sectors import get_stock_sector
                day_df['sector'] = day_df['symbol'].apply(lambda x: get_stock_sector(str(x)))
            except ImportError:
                logger.warning("Sector module not available, skipping sector diversification")
                day_df['sector'] = 'other'
        
        # Sort by rank score descending
        day_df = day_df.sort_values('rank_score_15d', ascending=False)
        
        # Select top stocks with sector diversification
        selected = []
        sector_counts = {}
        
        for _, row in day_df.iterrows():
            sector = row.get('sector', 'other')
            current_count = sector_counts.get(sector, 0)
            
            if current_count < max_per_sector:
                selected.append(row)
                sector_counts[sector] = current_count + 1
                
                if len(selected) >= TOP_K:
                    break
        
        if len(selected) < TOP_K:
            logger.warning(f"Only found {len(selected)} stocks meeting sector constraints")
        
        top = pd.DataFrame(selected)
        
        # Add rank and sector info
        if not top.empty:
            top['rank'] = range(1, len(top) + 1)
            
            # Add Chinese sector names
            try:
                from data.sectors import get_sector_name
                top['sector_cn'] = top['sector'].apply(lambda x: get_sector_name(x, chinese=True))
                top['sector_en'] = top['sector'].apply(lambda x: get_sector_name(x, chinese=False))
            except ImportError:
                top['sector_cn'] = top['sector']
                top['sector_en'] = top['sector']
        
        return top
    
    def save(self, ranker_path: str = None, regressor_path: str = None):
        """Save models to disk"""
        ranker_path = ranker_path or RANKER_15D_MODEL_FILE
        regressor_path = regressor_path or REGRESSOR_15D_MODEL_FILE
        
        Path(ranker_path).parent.mkdir(parents=True, exist_ok=True)
        Path(regressor_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save ranker
        with open(ranker_path, 'wb') as f:
            pickle.dump({
                'model': self.ranker,
                'feature_cols': self.feature_cols,
                'timestamp': datetime.now().isoformat(),
                'horizon': 15
            }, f)
        logger.info(f"Saved 15D ranker to {ranker_path}")
        
        # Save regressor
        with open(regressor_path, 'wb') as f:
            pickle.dump({
                'model': self.regressor,
                'feature_cols': self.feature_cols,
                'timestamp': datetime.now().isoformat(),
                'horizon': 15
            }, f)
        logger.info(f"Saved 15D regressor to {regressor_path}")
    
    def load(self, ranker_path: str = None, regressor_path: str = None):
        """Load models from disk"""
        ranker_path = ranker_path or RANKER_15D_MODEL_FILE
        regressor_path = regressor_path or REGRESSOR_15D_MODEL_FILE
        
        # Load ranker
        with open(ranker_path, 'rb') as f:
            ranker_data = pickle.load(f)
            self.ranker = ranker_data['model']
            self.feature_cols = ranker_data['feature_cols']
        logger.info(f"Loaded 15D ranker from {ranker_path}")
        
        # Load regressor
        with open(regressor_path, 'rb') as f:
            regressor_data = pickle.load(f)
            self.regressor = regressor_data['model']
        logger.info(f"Loaded 15D regressor from {regressor_path}")


def train_model_15d(df: pd.DataFrame = None, 
                    features_file: str = None,
                    fast_mode: bool = False) -> TwoStageModel15D:
    """Train the two-stage model for 15-day predictions"""
    if df is None:
        features_file = features_file or FEATURES_Z_FILE
        logger.info(f"Loading features from {features_file}")
        df = pd.read_parquet(features_file)
    
    model = TwoStageModel15D()
    metrics = model.train(df, fast_mode=fast_mode)
    
    # Save
    model.save()
    
    return model


def load_model_15d() -> TwoStageModel15D:
    """Load a pre-trained 15-day model"""
    model = TwoStageModel15D()
    model.load()
    return model


def generate_predictions_15d(features_file: str = None, output_file: str = None):
    """Generate top-10 predictions for 15-day horizon"""
    from config import FEATURES_Z_FILE, TOP10_LATEST_15D_FILE, TOP10_HISTORY_15D_FILE
    from data.sectors import get_stock_sector
    
    features_file = features_file or FEATURES_Z_FILE
    output_file = output_file or TOP10_LATEST_15D_FILE
    
    logger.info(f"Loading features from {features_file}")
    df = pd.read_parquet(features_file)
    
    # Add sector info
    logger.info("Adding sector information...")
    df['sector'] = df['symbol'].apply(lambda x: get_stock_sector(str(x)))
    
    # Load model
    logger.info("Loading 15D model...")
    model = load_model_15d()
    
    # Get latest date
    latest_date = df['date'].max()
    logger.info(f"Generating 15D predictions for {latest_date}")
    
    # Filter to latest date and predict
    day_df = df[df['date'] == latest_date].copy()
    day_df = model.predict(day_df)
    
    # Select top 10 with sector diversification
    top10 = model.select_top10(day_df, max_per_sector=MAX_STOCKS_PER_SECTOR)
    
    # Add Chinese selection reasons (15-day specific)
    logger.info("Adding selection reasons for 15D predictions...")
    try:
        from models.selection_reasons import add_selection_reasons
        top10 = add_selection_reasons(top10, include_news=True, horizon=15)
    except Exception as e:
        logger.warning(f"Could not add selection reasons: {e}")
        top10['reason_cn'] = '模型预测15日入选'
    
    # Save predictions
    top10.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(top10)} 15D predictions to {output_file}")
    
    # Update history
    logger.info("Updating 15D prediction history...")
    if TOP10_HISTORY_15D_FILE.exists():
        history = pd.read_parquet(TOP10_HISTORY_15D_FILE)
        history['date'] = pd.to_datetime(history['date'])
        # Remove existing entries for this date
        history = history[history['date'] != latest_date]
        history = pd.concat([history, top10], ignore_index=True)
    else:
        history = top10
    history = history.sort_values('date', ascending=False)
    history.to_parquet(TOP10_HISTORY_15D_FILE, index=False)
    logger.info(f"Updated 15D history: {history['date'].nunique()} dates")
    
    return top10


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train 15-day stock prediction model")
    parser.add_argument('--features', type=str, default=None, help="Path to features file")
    parser.add_argument('--fast', action='store_true', help="Use fast mode for quick training")
    parser.add_argument('--predict', action='store_true', help="Generate predictions after training")
    
    args = parser.parse_args()
    
    # Train
    model = train_model_15d(features_file=args.features, fast_mode=args.fast)
    print("15D Model training complete!")
    
    # Generate predictions if requested
    if args.predict:
        print("Generating 15D predictions...")
        top10 = generate_predictions_15d()
        print(f"Generated {len(top10)} predictions")
        print(top10[['symbol', 'pred_ret_15', 'sector_cn']].to_string())
