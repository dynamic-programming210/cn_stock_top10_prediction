#!/usr/bin/env python3
"""Update web.py to use reason_cn column"""

with open('app/web.py', 'r') as f:
    content = f.read()

# Replace all occurrences of reason_human with reason_cn
content = content.replace("'reason_human'", "'reason_cn'")
content = content.replace('"reason_human"', '"reason_cn"')

with open('app/web.py', 'w') as f:
    f.write(content)

print("Updated web.py to use reason_cn")
