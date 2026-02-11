"""
Tests for Stock Watcher predictions
Verifies that predictions are unique and reasonable for watched stocks
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from config import (
    FEATURES_Z_FILE, WATCHER_OUTPUT_FILE, WATCHER_INPUT_FILE,
    FEATURE_COLS, RANKER_MODEL_FILE, REGRESSOR_MODEL_FILE
)


class TestWatcherPredictions:
    """Tests for stock watcher prediction quality"""
    
    def test_predictions_are_unique(self):
        """Predictions should be unique for different stocks, not all the same value"""
        if not WATCHER_OUTPUT_FILE.exists():
            pytest.skip("No watcher output file")
        
        df = pd.read_parquet(WATCHER_OUTPUT_FILE)
        
        if len(df) < 2:
            pytest.skip("Need at least 2 stocks to test uniqueness")
        
        # Check for duplicate prediction values
        pred_5_unique = df['pred_ret_5'].nunique()
        pred_15_unique = df['pred_ret_15'].nunique()
        
        # At least 50% of predictions should be unique
        min_unique_ratio = 0.5
        actual_ratio_5 = pred_5_unique / len(df)
        actual_ratio_15 = pred_15_unique / len(df)
        
        assert actual_ratio_5 >= min_unique_ratio, (
            f"pred_ret_5: Only {actual_ratio_5:.0%} unique ({pred_5_unique}/{len(df)}). "
            f"This indicates a bug - predictions should be different for different stocks"
        )
        assert actual_ratio_15 >= min_unique_ratio, (
            f"pred_ret_15: Only {actual_ratio_15:.0%} unique ({pred_15_unique}/{len(df)}). "
            f"This indicates a bug - predictions should be different for different stocks"
        )
        
        print(f"✅ Predictions are sufficiently unique:")
        print(f"   pred_ret_5: {pred_5_unique}/{len(df)} unique values ({actual_ratio_5:.0%})")
        print(f"   pred_ret_15: {pred_15_unique}/{len(df)} unique values ({actual_ratio_15:.0%})")
    
    def test_predictions_in_reasonable_range(self):
        """Predictions should be in a reasonable range (not extreme values)"""
        if not WATCHER_OUTPUT_FILE.exists():
            pytest.skip("No watcher output file")
        
        df = pd.read_parquet(WATCHER_OUTPUT_FILE)
        
        # Check 5-day predictions are reasonable (between -50% and +100%)
        assert df['pred_ret_5'].min() > -0.5, f"pred_ret_5 min too low: {df['pred_ret_5'].min()}"
        assert df['pred_ret_5'].max() < 1.0, f"pred_ret_5 max too high: {df['pred_ret_5'].max()}"
        
        # Check 15-day predictions are reasonable (between -50% and +150%)
        assert df['pred_ret_15'].min() > -0.5, f"pred_ret_15 min too low: {df['pred_ret_15'].min()}"
        assert df['pred_ret_15'].max() < 1.5, f"pred_ret_15 max too high: {df['pred_ret_15'].max()}"
        
        print(f"✅ Predictions in reasonable range:")
        print(f"   pred_ret_5: {df['pred_ret_5'].min():.2%} to {df['pred_ret_5'].max():.2%}")
        print(f"   pred_ret_15: {df['pred_ret_15'].min():.2%} to {df['pred_ret_15'].max():.2%}")
    
    def test_watched_stocks_have_features(self):
        """All watched stocks should have valid features in the features file"""
        if not WATCHER_INPUT_FILE.exists():
            pytest.skip("No watcher input file")
        if not FEATURES_Z_FILE.exists():
            pytest.skip("No features file")
        
        # Read watchlist
        with open(WATCHER_INPUT_FILE, 'r') as f:
            watchlist = [x.strip() for x in f.read().strip().replace('\n', ',').split(',') if x.strip() and x.strip().isdigit()]
        
        if not watchlist:
            pytest.skip("Watchlist is empty")
        
        # Load features
        df = pd.read_parquet(FEATURES_Z_FILE)
        latest_date = df['date'].max()
        
        # Check each stock
        missing = []
        low_quality = []
        
        for sym in watchlist:
            stock = df[(df['symbol'] == sym) & (df['date'] == latest_date)]
            if stock.empty:
                missing.append(sym)
                continue
            
            # Check feature quality - count NaN/zero values
            feat_cols = [c for c in FEATURE_COLS if c in stock.columns]
            row = stock.iloc[0][feat_cols]
            nan_ratio = row.isna().sum() / len(row)
            zero_ratio = (row == 0).sum() / len(row)
            
            if nan_ratio > 0.3 or zero_ratio > 0.5:
                low_quality.append((sym, nan_ratio, zero_ratio))
        
        if missing:
            print(f"⚠️ Stocks missing from features: {missing}")
        if low_quality:
            print(f"⚠️ Stocks with low quality features: {low_quality}")
        
        # Fail if too many stocks are missing
        max_missing_ratio = 0.2  # Allow up to 20% missing
        actual_missing_ratio = len(missing) / len(watchlist)
        
        assert actual_missing_ratio <= max_missing_ratio, (
            f"{len(missing)}/{len(watchlist)} stocks missing from features ({actual_missing_ratio:.0%}). "
            f"Missing: {missing}"
        )
        
        print(f"✅ {len(watchlist) - len(missing)}/{len(watchlist)} stocks have features")


class TestFeatureQuality:
    """Tests for feature quality of watched stocks"""
    
    def test_features_are_different(self):
        """Features should be different for different stocks"""
        if not FEATURES_Z_FILE.exists():
            pytest.skip("No features file")
        if not WATCHER_INPUT_FILE.exists():
            pytest.skip("No watcher input file")
        
        # Read watchlist
        with open(WATCHER_INPUT_FILE, 'r') as f:
            watchlist = [x.strip() for x in f.read().strip().replace('\n', ',').split(',') if x.strip() and x.strip().isdigit()]
        
        if len(watchlist) < 2:
            pytest.skip("Need at least 2 stocks")
        
        # Load features
        df = pd.read_parquet(FEATURES_Z_FILE)
        latest_date = df['date'].max()
        
        # Get features for watchlist stocks
        feat_cols = [c for c in FEATURE_COLS if c in df.columns][:10]  # First 10 features
        features = {}
        
        for sym in watchlist:
            stock = df[(df['symbol'] == sym) & (df['date'] == latest_date)]
            if not stock.empty:
                features[sym] = tuple(stock.iloc[0][feat_cols].fillna(0).round(4).values)
        
        # Check for duplicate feature vectors
        unique_features = set(features.values())
        
        if len(unique_features) < len(features):
            duplicates = {}
            for sym, feat in features.items():
                if feat not in duplicates:
                    duplicates[feat] = []
                duplicates[feat].append(sym)
            
            duplicate_groups = {k: v for k, v in duplicates.items() if len(v) > 1}
            print(f"⚠️ Stocks with identical features: {duplicate_groups}")
        
        # At least 80% should be unique
        unique_ratio = len(unique_features) / len(features) if features else 0
        assert unique_ratio >= 0.8, (
            f"Only {unique_ratio:.0%} unique feature vectors. "
            "This indicates a data quality issue."
        )
        
        print(f"✅ {len(unique_features)}/{len(features)} unique feature vectors ({unique_ratio:.0%})")


class TestModelPredictions:
    """Test model prediction consistency"""
    
    def test_different_inputs_give_different_outputs(self):
        """Model should give different predictions for different feature inputs"""
        import pickle
        
        if not REGRESSOR_MODEL_FILE.exists():
            pytest.skip("No regressor model file")
        
        with open(REGRESSOR_MODEL_FILE, 'rb') as f:
            data = pickle.load(f)
            model = data['model'] if isinstance(data, dict) and 'model' in data else data
        
        # Create random feature inputs
        np.random.seed(42)
        n_features = len([c for c in FEATURE_COLS])
        
        # Generate different feature vectors
        X1 = np.random.randn(1, n_features)
        X2 = np.random.randn(1, n_features)
        X3 = np.zeros((1, n_features))  # All zeros
        X4 = np.ones((1, n_features))   # All ones
        
        pred1 = model.predict(X1)[0]
        pred2 = model.predict(X2)[0]
        pred3 = model.predict(X3)[0]
        pred4 = model.predict(X4)[0]
        
        # Predictions should be different
        predictions = [pred1, pred2, pred3, pred4]
        unique_preds = len(set(predictions))
        
        print(f"Predictions for different inputs:")
        print(f"  Random 1: {pred1:.6f}")
        print(f"  Random 2: {pred2:.6f}")
        print(f"  All zeros: {pred3:.6f}")
        print(f"  All ones: {pred4:.6f}")
        
        assert unique_preds >= 3, (
            f"Only {unique_preds} unique predictions for 4 different inputs. "
            "Model might be ignoring input features."
        )
        
        print(f"✅ Model gives {unique_preds}/4 unique predictions for different inputs")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
