#!/usr/bin/env python3
"""Script to fix the quality report and history files"""
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BARS_FILE, FEATURES_Z_FILE, TOP10_LATEST_FILE, TOP10_HISTORY_FILE,
    QUALITY_REPORT_FILE
)

def update_quality_report():
    """Update the quality report with latest data"""
    print("Updating quality report...")
    
    bars = pd.read_parquet(BARS_FILE)
    features = pd.read_parquet(FEATURES_Z_FILE)
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'asof_date': str(bars['date'].max().date()),
        'data': {
            'total_bars': len(bars),
            'unique_symbols': int(bars['symbol'].nunique()),
            'date_range': f"{bars['date'].min().date()} to {bars['date'].max().date()}",
            'exchanges': bars['exchange'].unique().tolist()
        },
        'features': {
            'total_rows': len(features),
            'unique_symbols': int(features['symbol'].nunique())
        },
        'coverage': {
            'asof_date_rate': float(len(bars[bars['date'] == bars['date'].max()]) / bars['symbol'].nunique())
        }
    }
    
    with open(QUALITY_REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Quality report updated: asof_date = {report['asof_date']}")
    return report


def update_history():
    """Add latest predictions to history if not already there"""
    print("Updating history...")
    
    if not TOP10_LATEST_FILE.exists():
        print("No latest predictions file found")
        return
    
    latest = pd.read_parquet(TOP10_LATEST_FILE)
    latest['date'] = pd.to_datetime(latest['date'])
    latest_date = latest['date'].iloc[0]
    
    print(f"Latest prediction date: {latest_date.date()}")
    
    if TOP10_HISTORY_FILE.exists():
        history = pd.read_parquet(TOP10_HISTORY_FILE)
        history['date'] = pd.to_datetime(history['date'])
        
        # Check if this date already exists
        if latest_date in history['date'].values:
            print(f"Date {latest_date.date()} already in history, replacing...")
            history = history[history['date'] != latest_date]
        
        # Append latest
        history = pd.concat([history, latest], ignore_index=True)
    else:
        history = latest
    
    # Sort by date descending
    history = history.sort_values('date', ascending=False)
    
    history.to_parquet(TOP10_HISTORY_FILE, index=False)
    
    print(f"History updated: {history['date'].nunique()} unique dates")
    print(f"Date range: {history['date'].min().date()} to {history['date'].max().date()}")


if __name__ == "__main__":
    update_quality_report()
    print()
    update_history()
    print("\n✅ Done!")
