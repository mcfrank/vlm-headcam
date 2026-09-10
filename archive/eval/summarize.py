"""Summarize experiment runs pulled from ccn2 into runs/. Prints a table of best/final
4AFC accuracy and val-loss trajectory per arm, and the top per-category accuracies."""
import json
import sys
from pathlib import Path

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")

print(f"{'arm':22} {'ep':>3} {'train':>7} {'val':>7} {'4afc':>6}  {'best4afc':>8}")
for d in sorted(runs.glob("*/")):
    lg = d / "log.json"
    if not lg.exists():
        continue
    log = json.loads(lg.read_text())
    last = log[-1]
    best = max(r["acc_4afc"] for r in log)
    print(f"{d.name:22} {last['epoch']:>3} {last['train_loss']:>7.3f} "
          f"{last['val_loss']:>7.3f} {last['acc_4afc']:>6.3f}  {best:>8.3f}")

for d in sorted(runs.glob("*/")):
    pc = d / "per_cat_final.json"
    if not pc.exists():
        continue
    cats = json.loads(pc.read_text())
    if not cats:
        continue
    top = sorted(cats.items(), key=lambda kv: -kv[1])[:12]
    bot = sorted(cats.items(), key=lambda kv: kv[1])[:6]
    print(f"\n[{d.name}] top: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
    print(f"[{d.name}] bottom: " + ", ".join(f"{k} {v:.2f}" for k, v in bot))
