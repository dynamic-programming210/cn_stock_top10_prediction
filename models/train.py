"""
Two-Stage Model for Chinese Stock Top-10 Prediction
Stage 1: RandomForest Classifier - ranks stocks by likelihood to outperform
Stage 2: XGBoost Regressor - predicts actual 5-day returns

Uses sklearn RandomForest instead of LightGBM for better compatibility.
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
    FEATURES_Z_FILE, FEATURE_COLS, TARGET_COL,
    RANKER_MODEL_FILE, REGRESSOR_MODEL_FILE,
    RANKER_PARAMS, REGRESSOR_PARAMS,
    TOP_K, MIN_RANK_SCORE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageModel:
    """Two-stage model: Ranker (RandomForest) + Regressor (XGBoost/sklearn)"""
    
    def __init__(self):
        self.ranker = None
        self.regressor = None
        self.feature_cols = None
        
    def _prepare_ranking_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for RandomForest ranker"""
        # Create relevance labels based on forward returns
        # Group stocks into quintiles by date
        df = df.copy()
        df['relevance'] = df.groupby('date')[TARGET_COL].transform(
            lambda x: pd.qcut(x.rank(method='first'), q=5, labels=[0, 1, 2, 3, 4])
        ).astype(int)
        
        # Get features
        X = df[self.feature_cols].values
        y = df['relevance'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, 
              feature_cols: List[str] = None,
              validation_split: float = 0.2,
              fast_mode: bool = False) -> dict:
        """Train both ranker and regressor
        
        Args:
            df: Training data
            feature_cols: Feature columns to use
            validation_split: Fraction for validation
            fast_mode: If True, use faster model settings (fewer trees, shallower)
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
        
        # Use sklearn only for better compatibility
        use_xgboost = False
        
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
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column {TARGET_COL} not found in data")
        
        # Drop rows with NaN
        clean = df[['date', 'symbol'] + self.feature_cols + [TARGET_COL]].dropna()
        logger.info(f"Training on {len(clean)} samples, {clean['symbol'].nunique()} symbols")
        
        # Split by date
        dates = sorted(clean['date'].unique())
        split_idx = int(len(dates) * (1 - validation_split))
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]
        
        train_df = clean[clean['date'].isin(train_dates)]
        val_df = clean[clean['date'].isin(val_dates)]
        
        logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
        
        # Stage 1: Train Ranker (RandomForest Classifier)
        logger.info("Training Stage 1: RandomForest Ranker...")
        
        X_train, y_train = self._prepare_ranking_data(train_df)
        X_val, y_val = self._prepare_ranking_data(val_df)
        
        self.ranker = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        )
        self.ranker.fit(X_train, y_train)
        
        train_acc = self.ranker.score(X_train, y_train)
        val_acc = self.ranker.score(X_val, y_val)
        logger.info(f"Ranker accuracy - Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        
        # Stage 2: Train Regressor
        logger.info("Training Stage 2: Regressor...")
        
        X_reg_train = train_df[self.feature_cols].values
        y_reg_train = train_df[TARGET_COL].values
        X_reg_val = val_df[self.feature_cols].values
        y_reg_val = val_df[TARGET_COL].values
        
        if use_xgboost:
            self.regressor = XGBRegressor(**REGRESSOR_PARAMS)
            self.regressor.fit(
                X_reg_train, y_reg_train,
                eval_set=[(X_reg_val, y_reg_val)],
                verbose=False
            )
        else:
            self.regressor = GradientBoostingRegressor(
                n_estimators=gb_n_estimators,
                max_depth=gb_max_depth,
                learning_rate=0.05,
                random_state=42
            )
            self.regressor.fit(X_reg_train, y_reg_train)
        
        # Evaluate
        metrics = self._evaluate(val_df)
        
        logger.info(f"Validation metrics: {metrics}")
        
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
        # Weight by quintile (higher quintiles are better)
        weights = np.arange(proba.shape[1])
        df['rank_score'] = (proba * weights).sum(axis=1)
        
        df['pred_ret_5'] = self.regressor.predict(X)
        
        # Calculate metrics by date
        results = []
        for date, group in df.groupby('date'):
            top10 = group.nlargest(10, 'rank_score')
            
            # Average actual return of top 10
            avg_ret = top10[TARGET_COL].mean()
            
            # Direction accuracy
            direction_acc = ((top10['pred_ret_5'] > 0) == (top10[TARGET_COL] > 0)).mean()
            
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
        """Predict rankings and returns"""
        if self.ranker is None or self.regressor is None:
            raise ValueError("Model not trained. Call train() first or load().")
        
        df = df.copy()
        
        # Ensure feature columns are present
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        
        # Handle NaN values - only predict for rows with complete features
        feature_cols_present = [c for c in self.feature_cols if c in df.columns]
        mask_valid = df[feature_cols_present].notna().all(axis=1)
        
        # Initialize output columns with NaN
        df['rank_score'] = np.nan
        df['pred_ret_5'] = np.nan
        
        if mask_valid.sum() == 0:
            logger.warning("No valid rows for prediction (all have NaN)")
            return df
        
        X = df.loc[mask_valid, self.feature_cols].values
        
        # Get rank score (weighted probability)
        proba = self.ranker.predict_proba(X)
        weights = np.arange(proba.shape[1])
        df.loc[mask_valid, 'rank_score'] = (proba * weights).sum(axis=1)
        
        df.loc[mask_valid, 'pred_ret_5'] = self.regressor.predict(X)
        
        return df
    
    def select_top10(self, df: pd.DataFrame, date: str = None, 
                     max_per_sector: int = None) -> pd.DataFrame:
        """
        Select top 10 stocks based on rank score with sector diversification.
        
        Args:
            df: DataFrame with predictions
            date: Date to select for (default: latest)
            max_per_sector: Maximum stocks per sector (default: from config)
        
        Returns:
            DataFrame with top 10 stocks, diversified across sectors
        """
        from config import MAX_STOCKS_PER_SECTOR
        
        if max_per_sector is None:
            max_per_sector = MAX_STOCKS_PER_SECTOR
            
        if date is None:
            date = df['date'].max()
        
        day_df = df[df['date'] == date].copy()
        
        if day_df.empty:
            logger.warning(f"No data for date {date}")
            return pd.DataFrame()
        
        # Predict if not already done
        if 'rank_score' not in day_df.columns:
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
        day_df = day_df.sort_values('rank_score', ascending=False)
        
        # Select top stocks with sector diversification
        selected = []
        sector_counts = {}
        
        for _, row in day_df.iterrows():
            sector = row.get('sector', 'other')
            current_count = sector_counts.get(sector, 0)
            
            # Check if we can add more from this sector
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
        ranker_path = ranker_path or RANKER_MODEL_FILE
        regressor_path = regressor_path or REGRESSOR_MODEL_FILE
        
        Path(ranker_path).parent.mkdir(parents=True, exist_ok=True)
        Path(regressor_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save ranker
        with open(ranker_path, 'wb') as f:
            pickle.dump({
                'model': self.ranker,
                'feature_cols': self.feature_cols,
                'timestamp': datetime.now().isoformat()
            }, f)
        logger.info(f"Saved ranker to {ranker_path}")
        
        # Save regressor
        with open(regressor_path, 'wb') as f:
            pickle.dump({
                'model': self.regressor,
                'feature_cols': self.feature_cols,
                'timestamp': datetime.now().isoformat()
            }, f)
        logger.info(f"Saved regressor to {regressor_path}")
    
    def load(self, ranker_path: str = None, regressor_path: str = None):
        """Load models from disk"""
        ranker_path = ranker_path or RANKER_MODEL_FILE
        regressor_path = regressor_path or REGRESSOR_MODEL_FILE
        
        # Load ranker
        with open(ranker_path, 'rb') as f:
            ranker_data = pickle.load(f)
            self.ranker = ranker_data['model']
            self.feature_cols = ranker_data['feature_cols']
        logger.info(f"Loaded ranker from {ranker_path}")
        
        # Load regressor
        with open(regressor_path, 'rb') as f:
            regressor_data = pickle.load(f)
            self.regressor = regressor_data['model']
        logger.info(f"Loaded regressor from {regressor_path}")


def train_model(df: pd.DataFrame = None, 
                features_file: str = None) -> TwoStageModel:
    """Train the two-stage model"""
    if df is None:
        features_file = features_file or FEATURES_Z_FILE
        logger.info(f"Loading features from {features_file}")
        df = pd.read_parquet(features_file)
    
    model = TwoStageModel()
    metrics = model.train(df)
    
    # Save
    model.save()
    
    return model


def load_model() -> TwoStageModel:
    """Load a pre-trained model"""
    model = TwoStageModel()
    model.load()
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument('--features', type=str, default=None, help="Path to features file")
    
    args = parser.parse_args()
    
    model = train_model(features_file=args.features)
    print("Model training complete!")
