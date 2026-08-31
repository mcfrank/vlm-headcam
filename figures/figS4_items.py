"""Display item 8 — does the model find hard what children find hard, and when?

A/B: item-level scatters at 300k pairs (mid-curve, widest item spread): per-category 4AFC
accuracy of the free learner against children's Rasch difficulty (mirt on all English
(American) Wordbank data; production = WG+WS pooled, comprehension = WG).
C: how that rank correlation changes with training scale (base rung; sign flipped so up =
more child-like). The profile is most child-aligned mid-curve and washes out at full scale,
where the model is at ceiling on most CDI-matched items. n grows with scale because a small
subsample's vocabulary covers fewer of the 74 CDI-matched categories (composition caveat).

Per-rung correlations at the ladder scales print to the build log.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
ie = pd.read_csv(R / "item_eval_final.csv")
ie = ie[ie.encoder == "dinov3b"]
wb = pd.read_csv(R / "wordbank_rasch.csv")

ALT = {"bike": "bicycle", "tv": "tv", "phone": "telephone", "socks": "sock", "boot": "boots",
       "leaves": "leaf", "glasses": "glasses", "fridge": "refrigerator", "pants": "pants"}
wbw = set(wb.word)
def match(c):
    for cand in (ALT.get(c), c, c + "s", c[:-1] if c.endswith("s") else None):
        if cand and cand in wbw:
            return cand

acc = (ie.groupby(["scale", "rung", "category"], as_index=False)
         .agg(acc=("acc", "mean"), in_vocab=("in_vocab", "all")))
acc = acc[acc.in_vocab]
acc["word"] = acc.category.map(match)
m = acc.merge(wb, on="word")
SCATTER_SCALE = "300000"

for sc in ["100000", "300000", "full"]:
    rows = []
    for rung in ["base", "filtnat", "t15", "t2"]:
        d = m[(m.scale == sc) & (m.rung == rung)].dropna(subset=["b_produce"])
        rows.append(f"{rung} r={spearmanr(d.acc, d.b_produce).correlation:+.2f} (n={len(d)})")
    print(f"  NOTE figS4 production @{sc}: " + "  ".join(rows))

fig, axes = plt.subplots(1, 3, figsize=(T.W2, 2.5), gridspec_kw=dict(wspace=0.32))
base = m[(m.rung == "base") & (m.scale == SCATTER_SCALE)]
for ax, meas, ttl in [(axes[0], "b_comprehend", "comprehension"),
                      (axes[1], "b_produce", "production")]:
    d = base.dropna(subset=[meas]).copy()
    r_s = spearmanr(d.acc, d[meas]).correlation
    ax.scatter(d[meas], d.acc, s=10, color=T.FREE, alpha=0.75, edgecolors="none", zorder=3)
    z = np.polyfit(d[meas], d.acc, 1)
    xx = np.linspace(d[meas].min(), d[meas].max(), 20)
    ax.plot(xx, np.polyval(z, xx), color=T.INK, lw=0.9, zorder=4)
    ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=1)
    d["resid"] = d.acc - np.polyval(z, d[meas])
    picks = pd.concat([d.nlargest(3, "resid"), d.nsmallest(3, "resid"),
                       d.nsmallest(2, meas), d.nlargest(2, meas)]).drop_duplicates("word")
    ceil_rank = 0
    for p_ in picks.sort_values(meas).itertuples():
        dy = 1.8
        if p_.acc > 92:
            dy = (2.4, 6.4)[ceil_rank % 2]; ceil_rank += 1
        ax.annotate(p_.word, (getattr(p_, meas), p_.acc),
                    (getattr(p_, meas) + 0.12, p_.acc + dy),
                    fontsize=4.6, color=T.SUB, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=T.GRID, lw=0.4, shrinkA=0, shrinkB=1))
    ax.text(0.04, 0.05, f"ρ = {r_s:+.2f}  (n = {len(d)})", transform=ax.transAxes,
            fontsize=5.8, color=T.INK)
    ax.set_xlabel(f"children's {ttl}\ndifficulty (Rasch $b$)")
    ax.set_ylim(0, 108)
    T.clean(ax)
axes[0].set_ylabel(f"model item accuracy (%)\nfree learner · 300k pairs")
axes[1].set_yticklabels([])

# ---- C: alignment across scale --------------------------------------------------
cx = axes[2]
SCALES = [("10000", 1e4), ("30000", 3e4), ("100000", 1e5), ("300000", 3e5),
          ("1000000", 1e6), ("full", 1.82e6)]
for meas, mk, ls, lab in [("b_produce", "^", "-", "production"),
                          ("b_comprehend", "o", (0, (2, 1.5)), "comprehension")]:
    xs, ys, ns = [], [], []
    for sc, n in SCALES:
        d = m[(m.rung == "base") & (m.scale == sc)].dropna(subset=[meas])
        if len(d) < 10:
            continue
        xs.append(n); ys.append(-spearmanr(d.acc, d[meas]).correlation); ns.append(len(d))
    cx.plot(xs, ys, marker=mk, ms=3, lw=1.0, ls=ls, color=T.FREE, zorder=3)
    if meas == "b_produce":
        cx.text(2.2e4, 0.355, lab, fontsize=5.4, color=T.FREE, ha="center", va="bottom")
    else:
        cx.text(1.25e6, 0.075, lab, fontsize=5.4, color=T.FREE, ha="right", va="bottom")
    if meas == "b_produce":
        for x_, y_, n_ in zip(xs, ys, ns):
            cx.text(x_, y_ + 0.04, str(n_), fontsize=4.4, color=T.SUB, ha="center")
cx.axhline(0, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
cx.text(1.1e4, 0.012, "no relation", fontsize=5.2, color=T.SUB, ha="left")
cx.axvline(float(SCATTER_SCALE), color=T.GRID, lw=0.7, zorder=0)
cx.text(float(SCATTER_SCALE) * 1.12, -0.06, "A, B", fontsize=5.2, color=T.SUB, ha="left")
cx.set_xscale("log")
cx.set_xlabel("training pairs")
cx.set_ylabel("child-alignment of item profile  (−ρ)")
cx.set_ylim(-0.08, 0.48)
cx.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
T.clean(cx)

for a, l in zip(axes, "ABC"):
    T.panel(a, l, dx=-0.13)
T.save(fig, "figS4_items")
