"""Per-child recording effort in the 2026.1 release.

Unevenness across children is expected, but it is what the diversity sweep manipulates, so
it should be visible: hours per child, sorted, with the largest contributor's share named.
Ported from supplement.qmd (fig-child-hours) onto the project theme; source data is the
committed release diagnostics, not a live cluster read.
"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import pandas as pd
import matplotlib.pyplot as plt

D = __import__("pathlib").Path(__file__).resolve().parent.parent / "diagnostics" / "2026.1"
C = pd.read_parquet(D / "child_level.parquet").sort_values("hours", ascending=False)
prov = json.loads((D / "provenance.json").read_text())

fig, ax = plt.subplots(figsize=(T.W2, 2.3))
ax.bar(range(len(C)), C.hours, color=T.FREE, width=0.75, zorder=3)
ax.set_xticks(range(len(C)))
ax.set_xticklabels(C.child, rotation=90, fontsize=3.8)
ax.set_xlim(-0.8, len(C) - 0.2)
ax.set_ylabel("hours recorded")
top, tot = C.hours.iloc[0], C.hours.sum()
med = C.hours.median()
ax.axhline(med, color=T.SUB, lw=0.6, ls=(0, (3, 2)), zorder=2)
ax.text(len(C) - 0.6, med + 2, f"median {med:.0f} h", fontsize=5.2, color=T.SUB, ha="right")
ax.text(0.985, 0.95, f"{len(C)} children · {tot:,.0f} h total\nlargest contributes "
        f"{100 * top / tot:.1f}%", transform=ax.transAxes, fontsize=5.6, color=T.INK,
        ha="right", va="top", linespacing=1.4)
T.clean(ax)
print(f"  NOTE figS_corpus_effort: release {prov['release']}, {len(C)} children, "
      f"{tot:,.0f} h; top {100 * top / tot:.1f}%, median {med:.1f} h, "
      f"min {C.hours.min():.1f} h")
T.save(fig, "figS_corpus_effort")
