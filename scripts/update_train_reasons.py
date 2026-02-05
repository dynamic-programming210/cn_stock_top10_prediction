#!/usr/bin/env python3
"""Script to update train.py to add selection reasons"""
import re

# Read the file
with open('models/train.py', 'r') as f:
    content = f.read()

# Find and replace to add selection reasons
old_text = '''    # Select top 10 with sector diversification
    top10 = model.select_top10(day_df, max_per_sector=MAX_STOCKS_PER_SECTOR)
    
    # Save predictions
    top10.to_parquet(output_file, index=False)'''

new_text = '''    # Select top 10 with sector diversification
    top10 = model.select_top10(day_df, max_per_sector=MAX_STOCKS_PER_SECTOR)
    
    # Add Chinese selection reasons
    logger.info("Adding selection reasons...")
    try:
        from models.selection_reasons import add_selection_reasons
        top10 = add_selection_reasons(top10, include_news=True)
    except Exception as e:
        logger.warning(f"Could not add selection reasons: {e}")
        top10['reason_cn'] = '模型预测入选'
    
    # Save predictions
    top10.to_parquet(output_file, index=False)'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('models/train.py', 'w') as f:
        f.write(content)
    print('Successfully updated models/train.py')
else:
    print('Could not find the text to replace - may already be updated')
