"""
Daily Update Pipeline for Chinese Stock Top-10 Predictor
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BARS_FILE, FEATURES_Z_FILE, RANKER_MODEL_FILE,
    TOP10_LATEST_FILE, TOP10_HISTORY_FILE, QUALITY_REPORT_FILE,
    OUTPUTS_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_data(full_refresh: bool = False, max_symbols: int = None):
    """Fetch/update price data"""
    from data.fetch_eodhd import update_data
    
    logger.info("Fetching data from EODHD...")
    bars, universe = update_data(full_refresh=full_refresh, max_symbols=max_symbols)
    
    return bars, universe


def build_features():
    """Build features from price data"""
    from features.build_features import build_features as _build_features
    
    logger.info("Building features...")
    df = _build_features()
    
    return df


def train_model(df=None):
    """Train the prediction model"""
    from models.train import train_model as _train_model
    
    logger.info("Training model...")
    model = _train_model(df)
    
    return model


def generate_predictions(model=None, df=None):
    """Generate top-10 predictions"""
    import pandas as pd
    from models.train import load_model
    from models.confidence import add_confidence_intervals
    
    if model is None:
        model = load_model()
    
    if df is None:
        df = pd.read_parquet(FEATURES_Z_FILE)
    
    # Get latest date
    latest_date = df['date'].max()
    logger.info(f"Generating predictions for {latest_date}")
    
    # Select top 10
    top10 = model.select_top10(df, date=latest_date)
    
    # Add confidence intervals
    top10 = add_confidence_intervals(top10)
    
    # Add metadata
    top10['model_version'] = 'cn-v1.0.0'
    top10['generated_at'] = datetime.now().isoformat()
    
    # Save latest
    top10.to_parquet(TOP10_LATEST_FILE, index=False)
    logger.info(f"Saved latest predictions to {TOP10_LATEST_FILE}")
    
    # Append to history
    if TOP10_HISTORY_FILE.exists():
        history = pd.read_parquet(TOP10_HISTORY_FILE)
        # Remove any existing entries for this date
        history = history[history['date'] != latest_date]
        history = pd.concat([history, top10], ignore_index=True)
    else:
        history = top10
    
    history.to_parquet(TOP10_HISTORY_FILE, index=False)
    logger.info(f"Updated history file: {TOP10_HISTORY_FILE}")
    
    return top10


def generate_quality_report(bars=None, features=None):
    """Generate data quality report"""
    import pandas as pd
    
    if bars is None:
        bars = pd.read_parquet(BARS_FILE)
    
    if features is None:
        features = pd.read_parquet(FEATURES_Z_FILE)
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'asof_date': str(bars['date'].max().date()),
        'data': {
            'total_bars': len(bars),
            'unique_symbols': bars['symbol'].nunique(),
            'date_range': f"{bars['date'].min().date()} to {bars['date'].max().date()}",
            'exchanges': bars['exchange'].unique().tolist()
        },
        'features': {
            'total_rows': len(features),
            'unique_symbols': features['symbol'].nunique()
        },
        'coverage': {
            'asof_date_rate': len(bars[bars['date'] == bars['date'].max()]) / bars['symbol'].nunique()
        }
    }
    
    with open(QUALITY_REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Generated quality report: {QUALITY_REPORT_FILE}")
    
    return report


def run_full_pipeline(full_refresh: bool = False, 
                      skip_data: bool = False,
                      skip_train: bool = False,
                      max_symbols: int = None):
    """Run the full daily update pipeline"""
    
    logger.info("=" * 50)
    logger.info("Chinese Stock Top-10 Predictor - Daily Update")
    logger.info("=" * 50)
    
    # Step 1: Fetch data
    if not skip_data:
        bars, universe = fetch_data(full_refresh=full_refresh, max_symbols=max_symbols)
    else:
        logger.info("Skipping data fetch...")
    
    # Step 2: Build features
    if not skip_data:
        df = build_features()
    else:
        import pandas as pd
        df = pd.read_parquet(FEATURES_Z_FILE)
    
    # Step 3: Train model (if needed or requested)
    if not skip_train and (not RANKER_MODEL_FILE.exists() or full_refresh):
        model = train_model(df)
    else:
        logger.info("Skipping model training (using existing model)...")
        model = None
    
    # Step 4: Generate predictions
    top10 = generate_predictions(model, df)
    
    # Step 5: Generate quality report
    generate_quality_report()
    
    # Summary
    logger.info("=" * 50)
    logger.info("Pipeline Complete!")
    logger.info(f"Top 10 predictions for {top10['date'].iloc[0]}:")
    for _, row in top10.iterrows():
        logger.info(f"  {row['symbol']}: {row['pred_ret_5']:.2%} (conf: {row.get('confidence_score', 0):.0%})")
    logger.info("=" * 50)
    
    return top10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chinese Stock Daily Update Pipeline")
    parser.add_argument("--full-refresh", action="store_true", help="Full data refresh")
    parser.add_argument("--skip-data", action="store_true", help="Skip data fetching")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--max-symbols", type=int, help="Limit symbols for testing")
    parser.add_argument("--test", action="store_true", help="Test mode with 20 symbols")
    parser.add_argument("--setup", action="store_true", help="Initial setup (full refresh + train)")
    
    args = parser.parse_args()
    
    if args.test:
        args.max_symbols = 20
        args.full_refresh = True
    
    if args.setup:
        args.full_refresh = True
        args.skip_train = False
    
    run_full_pipeline(
        full_refresh=args.full_refresh,
        skip_data=args.skip_data,
        skip_train=args.skip_train,
        max_symbols=args.max_symbols
    )
