"""Display item 11 — what the best free model learned about words.

A (large): t-SNE of the learned noun lexicon (B26_lad_base_s0; words above the learned-
   structure null), colored by MacArthur CDI semantic category; grey = nouns outside the
   CDI. Konkle-60 eval words ringed. Labels: strongest exemplars per category.
B: human relatedness across scale on noun-noun pairs — word2vec trained on the identical
   utterances, the two-tower, and the two-tower's UNIQUE contribution (partial rho
   controlling word2vec), same pairs at each scale.
C: CDI-category structure in the embedding space: within- vs between-category cosine per
   category, against the label-permutation null.
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
cs = pd.read_csv(R / "lexicon_category_structure.csv")

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
ax.scatter(bg.x, bg.y, s=4.5, c="#d5d4cd", lw=0, zorder=1)
for cat, col in CAT_COL.items():
    dd = d[d.cat == cat]
    ax.scatter(dd.x, dd.y, s=15, c=col, lw=0, zorder=3)
k = d[d.konkle60]
ax.scatter(k.x, k.y, s=30, facecolors="none", edgecolors=T.INK, lw=0.45, zorder=4)
labelled = []
for cat in CAT_COL:
    labelled.append(d[d.cat == cat].nlargest(8, "nn_cos"))
labelled.append(bg.nlargest(14, "nn_cos"))
placed, seen_stem = [], set()
for r in pd.concat(labelled).sort_values("nn_cos", ascending=False).itertuples():
    stem = r.word.rstrip("s")
    if stem in seen_stem:                       # one of dog/dogs is enough
        continue
    if any(abs(r.x - px) < 3.2 and abs(r.y - py) < 1.4 for px, py in placed):
        continue                                # keep labels off each other
    col = CAT_COL.get(r.cat, "#8a8a86")
    ax.annotate(r.word, (r.x, r.y), (r.x + 0.3, r.y + 0.3), fontsize=5.6, color=col, zorder=5)
    placed.append((r.x, r.y)); seen_stem.add(stem)
hs = [plt.Line2D([], [], marker="o", lw=0, ms=4.0, color=c,
                 label=f"{n.replace('_', ' / ')}") for n, c in CAT_COL.items()]
ax.legend(handles=hs, fontsize=5.6, loc="lower right", frameon=False, ncol=2,
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
pn = pt[(pt.category == "noun") & (pt.scale >= 1e4)]
g = pn.groupby("scale").agg(m=("partial_model", "mean"), me=("partial_model", "std"))
bx.errorbar(g.index, g.m, yerr=g.me, fmt=":o", color=T.FREE, ms=2.4, lw=0.9,
            elinewidth=0.5, capsize=1.3, markerfacecolor="white", zorder=2)
bx.text(0.05, 0.97, "word2vec (same utterances)", fontsize=5.4, color=T.SUB,
        transform=bx.transAxes, va="top")
bx.text(0.05, 0.885, "two-tower (grounded)", fontsize=5.4, color=T.FREE,
        transform=bx.transAxes, va="top")
bx.text(0.05, 0.80, "· · vision beyond language (partial)", fontsize=5.4, color=T.FREE,
        alpha=0.8, transform=bx.transAxes, va="top")
bx.set_xscale("log"); bx.set_xlim(6e3, 3e6); bx.set_ylim(-0.12, 0.5)
bx.set_xlabel("training pairs")
bx.set_ylabel("ρ with human relatedness\n(noun–noun pairs)", fontsize=6)
T.clean(bx)

# ---- C: CDI category structure --------------------------------------------------
cs = cs.sort_values("within", ascending=True).reset_index(drop=True)
ys = np.arange(len(cs))
for i, r in cs.iterrows():
    col = CAT_COL[r["category"]]
    cx.plot([r["between"], r["within"]], [i, i], color=col, lw=1.0, zorder=2)
    cx.scatter([r["within"]], [i], s=16, color=col, zorder=3)
    cx.scatter([r["between"]], [i], s=13, facecolors="white", edgecolors=col, lw=0.8, zorder=3)
nl = cs.null_sd.iloc[0]
cx.axvspan(-2 * nl, 2 * nl, color=T.NEUTRAL, alpha=0.25, lw=0, zorder=1)
cx.axvline(0, color=T.SUB, lw=0.5, zorder=1)
cx.set_yticks(ys)
cx.set_yticklabels([c.replace("_", " / ") for c in cs.category], fontsize=5.4)
cx.set_xlabel("mean cosine to category members\n(filled = within, open = between)", fontsize=6)
cx.set_xlim(-0.012, 0.1)
T.clean(cx, grid_axis="x")

T.panel(ax, "A", dx=0.01, dy=0.995)
T.panel(bx, "B", dx=-0.32)
T.panel(cx, "C", dx=-0.32)
print("  NOTE fig4: map = B26_lad_base_s0 nouns above 4x null; B/C shared noun pairs, "
      "w2v = SGNS s0 per scale")
T.save(fig, "fig4_lexicon")
