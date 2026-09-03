"""Display item 3 — what the best-performing model learned about words.

Best encoder = DINOv3-L off-the-shelf (81.7 full-corpus 4AFC vs 78.6 for the DINOv3-B
workhorse), so its lexicon is the one mapped and scored here. Final audio-filtered
corpus (F_ families, bv26a manifests) — the same corpus as figs 1-3, including the
word2vec topline in C. Panels run
qualitative -> quantified -> external comparison.

A (large): t-SNE of the learned noun lexicon (nouns whose neighbourhoods rise above the
   random-init null), colored by MacArthur CDI semantic category; grey = nouns with no
   CDI item or in a non-object CDI category. Konkle-60 eval words ringed.
B: the same structure quantified — each CDI category's cohesion gap (mean within-category
   cosine minus mean cosine to every other noun, full embedding space), with its OWN
   label-permutation null (5,000 shuffles per category per seed) and across-seed spread.
C: external comparison. Human relatedness across training scale on noun-noun pairs:
   word2vec trained on the identical utterances, the two-tower, and the two-tower's
   UNIQUE contribution (partial rho controlling word2vec); same pairs at each scale.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
RUN, FAM = "F_dinov3l_base_s0", "F-dinov3l"            # top encoder, FINAL corpus
MODEL = T.OTHER                                        # L-OTS keeps its fig2 colour
d = pd.read_csv(R / f"lexicon_tsne_{RUN}_NOUN.csv")
cdi = pd.read_csv(R / "cdi_categories.csv").set_index("word").category
ws = pd.read_csv(R / f"lexicon_ws_scaling_{FAM}.csv")
pt = pd.read_csv(R / f"lexicon_partial_{FAM}.csv")
cs = pd.read_csv(R / f"lexicon_category_structure_{RUN.rsplit(chr(95), 1)[0]}.csv")

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
    if any(abs(r.x - px) < 4.0 and abs(r.y - py) < 1.7 for px, py in placed):
        continue                                # keep labels off each other
    col = CAT_COL.get(r.cat, "#8a8a86")
    ax.annotate(r.word, (r.x, r.y), (r.x + 0.3, r.y + 0.3), fontsize=5.6, color=col, zorder=5)
    placed.append((r.x, r.y)); seen_stem.add(stem)
hs = [plt.Line2D([], [], marker="o", lw=0, ms=4.0, color=c,
                 label=f"{n.replace('_', ' / ')}") for n, c in CAT_COL.items()]
ax.legend(handles=hs, fontsize=5.6, loc="lower right", frameon=False, ncol=2,
          handletextpad=0.15, columnspacing=0.7, labelspacing=0.35, borderaxespad=0)
ax.axis("off")

# ---- B: CDI category structure --------------------------------------------------
cs = cs.sort_values("within", ascending=True).reset_index(drop=True)
cg = (cs[cs.category != "ALL"].groupby("category")
         .agg(n=("n", "first"), gap=("gap", "mean"), gsd=("gap", "std"),
              lo=("null_lo", "mean"), hi=("null_hi", "mean"), p=("p_perm", "max"))
         .sort_values("gap").reset_index())
ys = np.arange(len(cg))
for i, r in cg.iterrows():                     # per-category null band, then the observed gap
    bx.plot([r["lo"], r["hi"]], [i, i], color=T.NEUTRAL, lw=2.6, solid_capstyle="butt",
            zorder=1, alpha=0.9)
for i, r in cg.iterrows():                     # same category colours as panel A
    bx.errorbar([r["gap"]], [i], xerr=[r["gsd"]], fmt="o", color=CAT_COL[r["category"]],
                ms=3.4, lw=0, elinewidth=0.8, capsize=1.6, zorder=3)
bx.axvline(0, color=T.SUB, lw=0.5, zorder=1)
for i, r in cg.iterrows():
    if r["p"] >= 0.05:
        bx.text(r["gap"] + 0.004, i, "n.s.", fontsize=5.0, color=T.SUB, va="center")
bx.set_yticks(ys)
bx.set_yticklabels([f'{c.replace("_", "/")} ({n})' for c, n in zip(cg.category, cg.n)],
                   fontsize=5.2)
bx.set_ylim(-0.7, len(cg) - 0.3)
bx.set_xlabel("category cohesion gap\n(grey = permutation null, 95%)", fontsize=6)
bx.set_xlim(-0.015, 0.115)
T.clean(bx, grid_axis="x")

# ---- C: human relatedness across scale, noun pairs -------------------------------
wn = ws[(ws.category == "noun") & (ws.scale >= 1e4)]
m = wn[wn.kind == "model"].groupby("scale").agg(y=("spearman", "mean"), e=("spearman", "std"))
w = wn[wn.kind == "w2v"].groupby("scale").spearman.mean()
cx.errorbar(m.index, m.y, yerr=m.e, fmt="-o", color=MODEL, ms=2.6, lw=1.0,
            elinewidth=0.6, capsize=1.5, zorder=3)
cx.plot(w.index, w.values, "--s", color=T.SUB, ms=2.4, lw=0.9, zorder=3)
cx.axhline(0, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=1)   # chance: rho = 0
cx.text(2.7e6, -0.025, "chance", fontsize=5.4, color=T.SUB, ha="right", va="top")
pn = pt[(pt.category == "noun") & (pt.scale >= 1e4)]
g = pn.groupby("scale").agg(m=("partial_model", "mean"), me=("partial_model", "std"))
cx.errorbar(g.index, g.m, yerr=g.me, fmt=":o", color=MODEL, ms=2.4, lw=0.9,
            elinewidth=0.5, capsize=1.3, markerfacecolor="white", zorder=2)
cx.text(0.05, 0.97, "word2vec (same utterances)", fontsize=5.4, color=T.SUB,
        transform=cx.transAxes, va="top")
cx.text(0.05, 0.885, "L-OTS model", fontsize=5.4, color=MODEL,
        transform=cx.transAxes, va="top")
cx.text(0.05, 0.80, "· · L-OTS, word2vec partialled out", fontsize=5.4, color=MODEL,
        alpha=0.8, transform=cx.transAxes, va="top")
cx.set_xscale("log"); cx.set_xlim(6e3, 3e6); cx.set_ylim(-0.12, 0.5)
cx.set_xlabel("training pairs")
cx.set_ylabel("ρ with human relatedness\n(noun–noun pairs)", fontsize=6)
T.clean(cx)

T.panel(ax, "A", dx=0.01, dy=0.995)
T.panel(bx, "B", dx=-0.32)
T.panel(cx, "C", dx=-0.32)
al = cs[cs.category == "ALL"]
print(f"  NOTE fig3: {RUN}, {len(d)} nouns above 4x null; pooled gap "
      f"{al.gap.mean():.4f} (null {al.null_mean.mean():+.4f}, p<={al.p_perm.max():.4f}); "
      f"per-category p<.05 in {(cg.p < 0.05).sum()}/{len(cg)} "
      f"(n.s.: {', '.join(cg.category[cg.p >= 0.05]) or 'none'}); "
      f"{cs.seed.nunique()} seeds x {int(cs.n_perm.iloc[0]):,} permutations; "
      f"C on {ws[ws.category == 'noun'].n_pairs.max()} shared noun pairs")
T.save(fig, "fig3_lexicon")
