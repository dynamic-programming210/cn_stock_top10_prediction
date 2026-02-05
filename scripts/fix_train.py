#!/usr/bin/env python3
"""Fix the _update_quality_report function"""

with open('models/train.py', 'r') as f:
    content = f.read()

# Remove the errant 'return top10' from _update_quality_report
old = '''        logger.warning(f"Could not update quality report: {e}")
    
    return top10


if __name__ == "__main__":'''

new = '''        logger.warning(f"Could not update quality report: {e}")


if __name__ == "__main__":'''

if old in content:
    content = content.replace(old, new)
    with open('models/train.py', 'w') as f:
        f.write(content)
    print("Fixed _update_quality_report function")
else:
    print("Pattern not found")
