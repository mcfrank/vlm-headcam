"""S5 — the learned noun lexicon under all four encoders.

fig4A for every encoder: t-SNE of the noun embeddings that clear the random-vector null,
coloured by MacArthur CDI category. Each map is its own t-SNE, so positions are NOT
comparable across panels — only the degree to which same-category words group is. Each
panel is annotated with that encoder's 4AFC and its pooled category cohesion gap (the
statistic quantified per category in fig4B), so the visual impression can be checked
against a number.
"""
import sys, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
ENC = [("dinov3l", "L-OTS", "DINOv3-L off-the-shelf"),
       ("dinov3b", "B-OTS", "DINOv3-B off-the-shelf"),
       ("vitb_bv", "B-BV", "ViT-B BabyView-trained"),
       ("vits_bv", "S-BV", "ViT-S BabyView-trained")]
CAT_COL = {"animals": "#117733", "food_drink": "#CC6677", "vehicles": "#332288",
           "toys": "#AA4499", "clothing": "#88CCEE", "body_parts": "#44AA99",
           "household": "#999933", "furniture_rooms": "#DDCC77", "outside": "#6699CC",
           "places": "#888888"}
cdi = pd.read_csv(R / "cdi_categories.csv").set_index("word").category

fig, axes = plt.subplots(2, 2, figsize=(T.W2, 6.0))
for a, (enc, key, lab) in zip(axes.ravel(), ENC):
    d = pd.read_csv(R / f"lexicon_tsne_F_{enc}_base_s0_NOUN.csv")
    d["cat"] = d.word.map(cdi)
    d.loc[~d.cat.isin(CAT_COL), "cat"] = np.nan
    grey = d[d.cat.isna()]
    a.scatter(grey.x, grey.y, s=2.0, color="#dddad2", lw=0, zorder=1)
    for c, col in CAT_COL.items():
        m = d[d.cat == c]
        a.scatter(m.x, m.y, s=7.0, color=col, lw=0, zorder=2)
    acc = D.family(f"F_{enc}_base")["mean"]
    cs = pd.read_csv(R / f"lexicon_category_structure_F_{enc}_base.csv")
    gap = cs[cs.category == "ALL"].gap.mean()
    ns = cs[(cs.category != "ALL")].groupby("category").p_perm.max()
    a.set_title(f"{lab}   ·   {acc:.1f}% 4AFC   ·   cohesion gap {gap:.3f}",
                fontsize=6.2, color=T.INK, pad=3)
    a.text(0.015, 0.015, f"{len(d)} nouns above null   ·   "
           f"{(ns < 0.05).sum()}/{len(ns)} categories above chance",
           transform=a.transAxes, fontsize=5.2, color=T.SUB)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(T.GRID)
    print(f"  NOTE figS5 {key}: {len(d)} nouns, pooled gap {gap:.4f}, "
          f"{(ns < 0.05).sum()}/{len(ns)} categories p<.05"
          f" (n.s.: {', '.join(ns.index[ns >= 0.05]) or 'none'})")

handles = [plt.Line2D([], [], marker="o", lw=0, ms=3.0, color=c,
                      label=k.replace("_", " / ")) for k, c in CAT_COL.items()]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=5.8, frameon=False,
           handletextpad=0.3, columnspacing=1.1, bbox_to_anchor=(0.5, 0.028))
for a, l in zip(axes.ravel(), "ABCD"):
    T.panel(a, l, dx=-0.03, dy=1.045)
fig.subplots_adjust(wspace=0.06, hspace=0.14, bottom=0.085)
T.save(fig, "figS5_tsne_encoders")
