"""Age coverage of the 2026.1 release.

A: recorded hours by child age at recording.
B: longitudinal span, one cumulative-hours line per child, which shows both how long each
   child was followed and how unevenly the hours accumulate.
Ported from supplement.qmd (fig-age); source is the committed release diagnostics.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

D = __import__("pathlib").Path(__file__).resolve().parent.parent / "diagnostics" / "2026.1"
V = pd.read_parquet(D / "video_level.parquet")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.4), gridspec_kw=dict(wspace=0.26))
ax.hist(V.age_months, bins=40, weights=V.hours, color=T.FREE, zorder=3)
ax.set_xlabel("age at recording (months)"); ax.set_ylabel("hours recorded")
T.clean(ax)

for _, g in V.sort_values("age_months").groupby("child"):
    bx.plot(g.age_months, g.hours.cumsum(), lw=0.7, alpha=0.7, color=T.FREE, zorder=3)
bx.set_xlabel("age (months)"); bx.set_ylabel("cumulative hours per child")
T.clean(bx)
w = V.dropna(subset=["age_months"])
q = np.percentile(np.repeat(w.age_months, np.maximum(w.hours * 10, 0).astype(int)), [10, 50, 90])
print(f"  NOTE figS_corpus_age: hours-weighted age 10/50/90 pct = "
      f"{q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} months; span "
      f"{V.age_months.min():.0f}-{V.age_months.max():.0f}")
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.14, dy=1.09)
T.save(fig, "figS_corpus_age")
