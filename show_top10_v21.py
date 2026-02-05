#!/usr/bin/env python3
"""Show top 10 predictions with trend initiation signals"""
import pandas as pd
from models.train import TwoStageModel
from config import FEATURES_Z_FILE, TOP10_LATEST_FILE, FEATURE_COLS
from data.sectors import get_stock_sector

print('Loading features...')
df = pd.read_parquet(FEATURES_Z_FILE)
print(f'Total features: {len([c for c in FEATURE_COLS if c in df.columns])}')

print('Adding sector info...')
df['sector'] = df['symbol'].apply(lambda x: get_stock_sector(str(x)))

print('Loading trained model...')
model = TwoStageModel()
model.load()

latest_date = df['date'].max()
print(f'Generating predictions for {latest_date}...')

day_df = df[df['date'] == latest_date].copy()
day_df = model.predict(day_df)

print('Selecting top 10 with sector diversification...')
top10 = model.select_top10(day_df, max_per_sector=2)
top10.to_parquet(TOP10_LATEST_FILE, index=False)

# Show results with trend initiation features
print(f'\n{"="*80}')
print(f'📈 TOP 10 STOCKS FOR {latest_date} (v2.1 - Trend Initiation Model)')
print(f'{"="*80}')

for idx, (i, row) in enumerate(top10.iterrows()):
    print(f"\n{idx+1}. {row['symbol']} ({row.get('sector', 'Unknown')})")
    print(f"   Rank Score: {row['rank_score']:.4f} | Predicted 5D Return: {row['pred_ret_5']*100:.2f}%")
    
    # Show trend initiation signals
    signals = []
    if 'trend_initiation_score' in row.index and pd.notna(row['trend_initiation_score']) and row['trend_initiation_score'] > 0:
        signals.append(f"TrendInit={row['trend_initiation_score']:.1f}")
    if 'golden_cross_5d' in row.index and pd.notna(row['golden_cross_5d']) and row['golden_cross_5d'] > 0:
        signals.append(f"GoldenCross(5d)={int(row['golden_cross_5d'])}")
    if 'breakout_with_volume' in row.index and pd.notna(row['breakout_with_volume']) and row['breakout_with_volume'] > 0:
        signals.append('BREAKOUT+VOL')
    if 'reversal_pattern' in row.index and pd.notna(row['reversal_pattern']) and row['reversal_pattern'] > 0:
        signals.append('REVERSAL_PATTERN')
    if 'morning_star' in row.index and pd.notna(row['morning_star']) and row['morning_star'] > 0:
        signals.append('MorningStar')
    if 'bullish_engulfing' in row.index and pd.notna(row['bullish_engulfing']) and row['bullish_engulfing'] > 0:
        signals.append('BullishEngulf')
    if 'higher_low_count_20' in row.index and pd.notna(row['higher_low_count_20']) and row['higher_low_count_20'] > 1:
        signals.append(f"HigherLows={int(row['higher_low_count_20'])}")
    if 'strong_stock_score' in row.index and pd.notna(row['strong_stock_score']) and row['strong_stock_score'] > 5:
        signals.append(f"StrongStock={row['strong_stock_score']:.1f}")
        
    if signals:
        print(f"   🚀 Signals: {', '.join(signals)}")
    else:
        print(f"   (No active trend initiation signals)")

print(f'\n{"="*80}')
print(f'Model version: cn-v2.1.0 with trend initiation features')
print(f'Features used: {len(model.feature_cols)}')
