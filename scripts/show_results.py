#!/usr/bin/env python
"""Print the accuracy-vs-K table from outputs/module_e/real/exp1.csv (mean over seeds)."""
import csv, sys
from collections import defaultdict
from pathlib import Path

f = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/module_e/real/exp1.csv")
rows = list(csv.DictReader(open(f)))
models = [k[4:] for k in rows[0] if k.startswith("acc_") and not k.endswith(("_lo", "_hi"))]
for m in models:
    d = defaultdict(list)
    for r in rows:
        d[(r["arm"], int(r["k"]))].append(float(r[f"acc_{m}"]))
    arms = sorted({a for a, _ in d}); ks = sorted({k for _, k in d})
    print(f"\n{m} accuracy (mean over seeds; {rows[0]['n_images']} images per cell)")
    print(f"{'K':>4}  " + "  ".join(f"{a:>14}" for a in arms))
    for k in ks:
        print(f"{k:>4}  " + "  ".join(f"{sum(d[(a, k)]) / len(d[(a, k)]):>14.3f}" if (a, k) in d else f"{'-':>14}" for a in arms))
