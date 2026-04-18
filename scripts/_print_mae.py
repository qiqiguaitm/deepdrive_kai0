#!/usr/bin/env python3
"""Print MAE summary from eval_val.json. Usage: _print_mae.py STEP PATH"""
import json, sys
step = sys.argv[1]
data = json.load(open(sys.argv[2]))
m = data["mae"]
print(f"[eval] v2 step={step}  MAE@1={m['1']:.4f}  @10={m['10']:.4f}  @25={m['25']:.4f}  @50={m['50']:.4f}")
