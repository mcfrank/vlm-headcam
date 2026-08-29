"""Display item 11 — what the best free model learned about words.

A (large): t-SNE of the learned noun lexicon (B26_lad_base_s0; words above the learned-
   structure null), colored by MacArthur CDI semantic category; grey = nouns outside the
   CDI. Konkle-60 eval words ringed. Labels: strongest exemplars per category.
B: human relatedness across scale on noun-noun pairs — the two-tower vs word2vec trained
   on the identical utterances, same pairs at each scale.
C: the unique contributions (rank-based partial correlations): language-beyond-vision vs
   vision-beyond-language on the same pairs.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
d = pd.read_csv(R / "lexicon_tsne_B26_lad_base_s0_NOUN.csv")
cdi = pd.read_csv(R / "cdi_categories.csv").set_index("word").category
ws = pd.read_csv(R / "lexicon_ws_scaling.csv")
pt = pd.read_csv(R / "lexicon_partial.csv")

CAT_COL = {"animals": "#117733", "food_drink": "#CC6677", "vehicles": "#332288",
           "toys": "#AA4499", "clothing": "#88CCEE", "body_parts": "#44AA99",
           "household": "#999933", "furniture_rooms": "#DDCC77", "outside": "#6699CC",
           "places": "#888888"}
d["cat"] = d.word.map(cdi)
d.loc[~d.cat.isin(CAT_COL), "cat"] = np.nan

fig = plt.figure(figsize=(T.W2, 4.4))
ax = fig.add_axes([0.005, 0.01, 0.615, 0.97])
bx = fig.add_axes([0.72, 0.585, 0.27, 0.36])
cx = fig.add_axes([0.72, 0.10, 0.27, 0.36])

# ---- A: the noun map -------------------------------------------------------------
bg = d[d.cat.isna()]
ax.scatter(bg.x, bg.y, s=2.5, c="#d5d4cd", lw=0, zorder=1)
for cat, col in CAT_COL.items():
    dd = d[d.cat == cat]
    ax.scatter(dd.x, dd.y, s=8, c=col, lw=0, zorder=3)
k = d[d.konkle60]
ax.scatter(k.x, k.y, s=16, facecolors="none", edgecolors=T.INK, lw=0.35, zorder=4)
labelled = []
for cat in CAT_COL:
    labelled.append(d[d.cat == cat].nlargest(8, "nn_cos"))
labelled.append(bg.nlargest(22, "nn_cos"))
for r in pd.concat(labelled).itertuples():
    col = CAT_COL.get(r.cat, "#8a8a86")
    ax.annotate(r.word, (r.x, r.y), (r.x + 0.25, r.y + 0.25), fontsize=3.9, color=col, zorder=5)
hs = [plt.Line2D([], [], marker="o", lw=0, ms=3.2, color=c,
                 label=f"{n.replace('_', ' / ')}") for n, c in CAT_COL.items()]
ax.legend(handles=hs, fontsize=4.6, loc="lower right", frameon=False, ncol=2,
          handletextpad=0.15, columnspacing=0.7, labelspacing=0.35, borderaxespad=0)
ax.axis("off")

# ---- B: human relatedness across scale, noun pairs -------------------------------
wn = ws[(ws.category == "noun") & (ws.scale >= 1e4)]
m = wn[wn.kind == "model"].groupby("scale").agg(y=("spearman", "mean"), e=("spearman", "std"))
w = wn[wn.kind == "w2v"].groupby("scale").spearman.mean()
bx.errorbar(m.index, m.y, yerr=m.e, fmt="-o", color=T.FREE, ms=2.6, lw=1.0,
            elinewidth=0.6, capsize=1.5, zorder=3)
bx.plot(w.index, w.values, "--s", color=T.SUB, ms=2.4, lw=0.9, zorder=3)
bx.axhline(0, color=T.GRID, lw=0.6)
bx.text(0.05, 0.95, "word2vec (same utterances)", fontsize=5.4, color=T.SUB,
        transform=bx.transAxes, va="top")
bx.text(0.05, 0.86, "two-tower (grounded)", fontsize=5.4, color=T.FREE,
        transform=bx.transAxes, va="top")
bx.set_xscale("log"); bx.set_xlim(6e3, 3e6); bx.set_ylim(-0.12, 0.5)
bx.set_xticklabels([])
bx.set_ylabel("ρ with human relatedness\n(noun–noun pairs)", fontsize=6)
T.clean(bx)

# ---- C: unique contributions (partials) ------------------------------------------
pn = pt[(pt.category == "noun") & (pt.scale >= 1e4)]
g = pn.groupby("scale").agg(m=("partial_model", "mean"), me=("partial_model", "std"),
                            v=("partial_w2v", "mean"), ve=("partial_w2v", "std"))
cx.errorbar(g.index, g.v, yerr=g.ve, fmt="--s", color=T.SUB, ms=2.4, lw=0.9,
            elinewidth=0.6, capsize=1.5, zorder=3)
cx.errorbar(g.index, g.m, yerr=g.me, fmt="-o", color=T.FREE, ms=2.6, lw=1.0,
            elinewidth=0.6, capsize=1.5, zorder=3)
cx.axhline(0, color=T.GRID, lw=0.6)
cx.text(0.05, 0.95, "language beyond vision", fontsize=5.4, color=T.SUB,
        transform=cx.transAxes, va="top")
cx.text(0.05, 0.86, "vision beyond language", fontsize=5.4, color=T.FREE,
        transform=cx.transAxes, va="top")
cx.set_xscale("log"); cx.set_xlim(6e3, 3e6); cx.set_ylim(-0.12, 0.5)
cx.set_xlabel("training pairs")
cx.set_ylabel("unique contribution\n(partial ρ)", fontsize=6)
T.clean(cx)

T.panel(ax, "A", dx=0.01, dy=0.995)
T.panel(bx, "B", dx=-0.32)
T.panel(cx, "C", dx=-0.32)
print("  NOTE fig11: map = B26_lad_base_s0 nouns above 4x null; B/C shared noun pairs, "
      "w2v = SGNS s0 per scale")
T.save(fig, "fig11_lexicon")
