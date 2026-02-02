#!/usr/bin/env python3
"""
Daily Update Pipeline for Chinese Stock Predictor
Can be run locally or via GitHub Actions

Usage:
    python run_daily_update.py              # Normal update (incremental data)
    python run_daily_update.py --full       # Full data refresh
    python run_daily_update.py --skip-fetch # Skip data fetching
    python run_daily_update.py --skip-train # Skip model training
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"❌ {description} failed with code {result.returncode}")
        return False
    
    print(f"✅ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Daily update pipeline")
    parser.add_argument("--full", action="store_true", help="Full data refresh")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature building")
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"\n🚀 Starting daily update pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    steps = []
    
    # Step 1: Fetch data
    if not args.skip_fetch:
        fetch_cmd = [sys.executable, "data/fetch_eodhd.py"]
        if args.full:
            fetch_cmd.append("--full-refresh")
        steps.append((fetch_cmd, "Fetching stock data"))
    
    # Step 2: Build features
    if not args.skip_features:
        steps.append(([sys.executable, "features/build_features.py"], "Building features"))
    
    # Step 3: Train model
    if not args.skip_train:
        steps.append(([sys.executable, "models/train.py", "--retrain"], "Training model"))
    
    # Step 4: Generate predictions (always run)
    predict_script = """
import sys
sys.path.insert(0, '.')
from models.train import generate_predictions
generate_predictions()
"""
    steps.append(([sys.executable, "-c", predict_script], "Generating predictions"))
    
    # Step 5: Show predictions
    steps.append(([sys.executable, "show_predictions.py"], "Displaying predictions"))
    
    # Run all steps
    failed_steps = []
    for cmd, description in steps:
        if not run_command(cmd, description):
            failed_steps.append(description)
            # Continue with other steps even if one fails
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("📊 PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    
    if failed_steps:
        print(f"\n❌ Failed steps ({len(failed_steps)}):")
        for step in failed_steps:
            print(f"   - {step}")
        return 1
    else:
        print("\n✅ All steps completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
