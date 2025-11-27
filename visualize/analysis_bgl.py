import argparse
import sys
from pathlib import Path
from math import ceil
from collections import Counter

path = Path('./datasets/BGL/raw_log_seqs.txt')
if not path.exists():
    print(f"File not found: {path}", file=sys.stderr)
    sys.exit(2)
    
counts = Counter()
with path.open("r", encoding="utf-8", errors="replace") as f:
    for line in f:
        tokens = line.strip().split(':')
        if len(tokens) > 0:
            node = tokens[0]
            if node:
                counts[node] += 1

for node, cnt in counts.most_common(10):
    print(f"{node}\t{cnt}")