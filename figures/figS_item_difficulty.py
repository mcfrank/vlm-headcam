"""Does the model find hard the words children find hard?

Item-level comparison between per-category 4AFC accuracy (200 trials x seeds, in-vocabulary
categories only) and children's age of acquisition, indexed by Rasch item difficulty fit
with mirt to all English (American) Wordbank data (production = WG+WS pooled, comprehension
= WG). Higher difficulty = acquired later, so a NEGATIVE correlation is the child-like
direction; panel A plots -rho so that up is more child-like.

A: the correspondence depends on training scale, and the reason is measurement, not
   development: as the model saturates, its accuracy on CDI-matched items piles up against
   the ceiling and the item variance that any correlation needs disappears (grey, right
   axis: the share of items scoring above 90%). We therefore show the scatters at 300k
   pairs, where nearly all matched items are available (73 of 74) but only 23% are at
   ceiling. Points at the smallest scales rest on few items and are marked with their n.
B, C: the scatters at that scale, with the outermost points labelled.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
ENC, SHOW = "dinov3l", "300000"
ie = pd.read_csv(R / "item_eval_final.csv")
ie = ie[(ie.encoder == ENC) & (ie.rung == "base")]
wb = pd.read_csv(R / "wordbank_rasch.csv")
ALT = {"bike": "bicycle", "tv": "tv", "phone": "telephone", "socks": "sock", "boot": "boots",
       "leaves": "leaf", "glasses": "glasses", "fridge": "refrigerator", "pants": "pants"}
W = set(wb.word)
def match(c):
    for k in (ALT.get(c), c, c + "s", c[:-1] if c.endswith("s") else None):
        if k and k in W:
            return k
acc = ie.groupby(["scale", "category"], as_index=False).agg(acc=("acc", "mean"),
                                                            iv=("in_vocab", "all"))
acc = acc[acc.iv].copy(); acc["word"] = acc.category.map(match)
J = acc.merge(wb, on="word")

fig = plt.figure(figsize=(T.W2, 2.6))
ax = fig.add_axes([0.075, 0.20, 0.275, 0.72])
bx = fig.add_axes([0.505, 0.20, 0.22, 0.72])
cx = fig.add_axes([0.775, 0.20, 0.22, 0.72])

# ---- A: correspondence vs scale, with the ceiling that explains it ---------------
SC = [("10000", 1e4), ("30000", 3e4), ("100000", 1e5), ("300000", 3e5),
      ("1000000", 1e6), ("full", 1.686e6)]
rows = []
for s, n in SC:
    d = J[J.scale == s]
    dp = d.dropna(subset=["b_produce"]); dc = d.dropna(subset=["b_comprehend"])
    rows.append(dict(n=n, scale=s, np_=len(dp),
                     pr=-spearmanr(dp.acc, dp.b_produce).correlation,
                     cm=-spearmanr(dc.acc, dc.b_comprehend).correlation,
                     ceil=100 * (d.acc > 90).mean()))
S = pd.DataFrame(rows)
axc = ax.twinx()
axc.plot(S.n, S.ceil, color=T.NEUTRAL, lw=1.0, ls=(0, (2, 1.5)), zorder=1)
axc.set_ylim(0, 100); axc.set_ylabel("% of items scoring >90%", fontsize=6, color=T.SUB)
axc.tick_params(labelsize=6, colors=T.SUB)
for sp in axc.spines.values():
    sp.set_visible(False)
axc.text(2.6e6, 78, "items at ceiling\n(right axis)", fontsize=5.0, color=T.SUB,
         ha="right", va="center", linespacing=1.3)
for k, lab, mk in [("pr", "production", "^"), ("cm", "comprehension", "o")]:
    ax.plot(S.n, S[k], marker=mk, ms=3, lw=1.1, color=T.FREE,
            ls="-" if k == "pr" else (0, (2, 1.5)), zorder=3)
    ax.text(S.n.iloc[-1] * 1.35, S[k].iloc[-1], lab, fontsize=5.2, color=T.FREE, va="center")
for n_, k in zip(S.n, S.np_):
    ax.text(n_, -0.085, str(k), fontsize=4.6, color=T.SUB, ha="center")
ax.text(7.5e3, -0.108, "items compared", fontsize=4.6, color=T.SUB, ha="left")
ax.axhline(0, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=1)
ax.axvline(3e5, color=T.GRID, lw=1.4, zorder=0)
ax.text(3e5, 0.60, "shown in\nB, C", fontsize=5.2, color=T.SUB, ha="center", va="top",
        linespacing=1.3)
ax.set_xscale("log"); ax.set_xlim(6.5e3, 4.5e6); ax.set_ylim(-0.12, 0.62)
ax.set_xlabel("training pairs")
ax.set_ylabel("child-alignment of item profile (−ρ)")
ax.set_zorder(axc.get_zorder() + 1); ax.patch.set_visible(False)
T.clean(ax)

# ---- B, C: the scatters at the chosen scale -------------------------------------
for a, meas, ttl in [(bx, "b_comprehend", "comprehension"), (cx, "b_produce", "production")]:
    d = J[J.scale == SHOW].dropna(subset=[meas]).copy()
    r_s = spearmanr(d.acc, d[meas]).correlation
    a.scatter(d[meas], d.acc, s=10, color=T.FREE, alpha=0.75, edgecolors="none", zorder=3)
    z = np.polyfit(d[meas], d.acc, 1)
    xx = np.linspace(d[meas].min(), d[meas].max(), 20)
    a.plot(xx, np.polyval(z, xx), color=T.INK, lw=0.9, zorder=4)
    a.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=1)
    d["resid"] = d.acc - np.polyval(z, d[meas])
    picks = pd.concat([d.nsmallest(3, "resid"), d.nsmallest(1, meas),
                       d.nlargest(1, meas)]).drop_duplicates("word")
    for p_ in picks.itertuples():
        a.annotate(p_.word, (getattr(p_, meas), p_.acc),
                   (getattr(p_, meas) + 0.15, p_.acc - 4.0), fontsize=4.8, color=T.SUB,
                   zorder=5, arrowprops=dict(arrowstyle="-", color=T.GRID, lw=0.4,
                                             shrinkA=0, shrinkB=1))
    a.text(0.04, 0.05, f"ρ = {r_s:+.2f}  (n = {len(d)})", transform=a.transAxes,
           fontsize=5.8, color=T.INK)
    a.set_xlabel(f"children's {ttl}\ndifficulty (Rasch $b$)")
    a.set_ylim(0, 112)
    T.clean(a)
bx.set_ylabel("item accuracy (%)\nmodel trained on 300k pairs", fontsize=6)
cx.set_yticklabels([])
print(f"  NOTE figS_item_difficulty: {ENC}, scatters at {SHOW}; " +
      "  ".join(f"{r.scale}:prod{r.pr:+.2f}/comp{r.cm:+.2f}(n={r.np_},ceil{r.ceil:.0f}%)"
                for r in S.itertuples()))
for a, l in zip((ax, bx, cx), "ABC"):
    T.panel(a, l, dx=-0.22, dy=1.06)
T.save(fig, "figS_item_difficulty")
