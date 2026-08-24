"""Display item 8 — does the model find hard what children find hard?

Item-level comparison: per-category 4AFC accuracy of the free (base) learner across all 177
Konkle categories (200 trials x 3 seeds each; results/item_eval_b26.csv) against a scale-free
age of acquisition — Rasch item difficulties fit with mirt to ALL English (American) Wordbank
data (production = WG+WS pooled; comprehension = WG). Two panels: comprehension, production.
Higher b = acquired later. Trend line, correlation, and word labels on the outside of the cloud.

Per-rung correlations print to the build log (which training signal is most child-like).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
ev = pd.read_csv(R / "item_eval_b26.csv")
wb = pd.read_csv(R / "wordbank_rasch.csv")

# Konkle category -> CDI word (normalized item_definition); only clean equivalents
ALT = {"bike": "bicycle", "tv": "tv", "phone": "telephone", "socks": "sock", "boot": "boots",
       "leaves": "leaf", "glasses": "glasses", "fridge": "refrigerator", "pants": "pants"}
wbw = set(wb.word)
def match(c):
    for cand in (ALT.get(c), c, c + "s", c[:-1] if c.endswith("s") else None):
        if cand and cand in wbw:
            return cand
    return None

acc = (ev.groupby(["rung", "category"], as_index=False)
         .agg(acc=("acc", "mean"), sd=("acc", "std"), in_vocab=("in_vocab", "all")))
acc = acc[acc.in_vocab]                       # a word absent from training text can't be learned
acc["word"] = acc.category.map(match)
m = acc.merge(wb, on="word")
matched = m[m.rung == "base"]
print(f"  NOTE fig8: {matched.word.nunique()} of {ev.category.nunique()} categories matched to CDI items")
for meas in ["b_comprehend", "b_produce"]:
    rows = []
    for rung in ["base", "filtnat", "t15", "t2"]:
        d = m[(m.rung == rung)].dropna(subset=[meas])
        rows.append(f"{rung} r={spearmanr(d.acc, d[meas]).correlation:+.2f} (n={len(d)})")
    print(f"  NOTE fig8 {meas}: " + "  ".join(rows))

fig, axes = plt.subplots(1, 2, figsize=(T.W2, 2.7))
base = m[m.rung == "base"]
for ax, meas, title_col in [(axes[0], "b_comprehend", "comprehension"),
                            (axes[1], "b_produce", "production")]:
    d = base.dropna(subset=[meas]).copy()
    r_s = spearmanr(d.acc, d[meas]).correlation
    r_p = pearsonr(d.acc, d[meas]).statistic
    ax.scatter(d[meas], d.acc, s=11, color=T.FREE, alpha=0.75, edgecolors="none", zorder=3)
    z = np.polyfit(d[meas], d.acc, 1)
    xx = np.linspace(d[meas].min(), d[meas].max(), 20)
    ax.plot(xx, np.polyval(z, xx), color=T.INK, lw=0.9, zorder=4)
    ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=1)
    # label points on the outside of the cloud: largest residuals + the axis extremes
    d["resid"] = d.acc - np.polyval(z, d[meas])
    picks = pd.concat([d.nlargest(3, "resid"), d.nsmallest(4, "resid"),
                       d.nsmallest(2, meas), d.nlargest(2, meas)]).drop_duplicates("word")
    ceil_rank = 0
    for p_ in picks.sort_values(meas).itertuples():
        dy = 1.6
        if p_.acc > 94:                        # stagger labels crowding the ceiling
            dy = (2.2, 5.6)[ceil_rank % 2]; ceil_rank += 1
        ax.annotate(p_.word, (getattr(p_, meas), p_.acc),
                    (getattr(p_, meas) + 0.12, p_.acc + dy),
                    fontsize=4.6, color=T.SUB, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=T.GRID, lw=0.4, shrinkA=0, shrinkB=1))
    ax.text(0.03, 0.06, f"ρ = {r_s:+.2f}   (r = {r_p:+.2f}, n = {len(d)})",
            transform=ax.transAxes, fontsize=6, color=T.INK)
    ax.set_xlabel(f"children's {title_col} difficulty  (Rasch $b$, Wordbank)")
    ax.set_ylim(0, 108)
    T.clean(ax)
axes[0].set_ylabel("model item accuracy (%)  ·  free learner")
axes[1].set_yticklabels([])
for a, l in zip(axes, "AB"):
    T.panel(a, l, dx=-0.12)
T.save(fig, "fig8_items")
