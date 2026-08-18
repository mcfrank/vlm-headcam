"""Talk/book replots that need only cached data (no model, no frames):
  (1) fig_item_analysis.png  — adds a linear trend line to the frequency panel
  (2) fig_confusion.png      — RDM with diagonal masked, colored cluster brackets + names, legible labels
      fig_confusion_mds.png  — 2-D MDS map of the same confusion, items labeled, colored by cluster
  (3) fig_cues.png           — every accessible cue's information about alignment on one scale, vs the
                               ignition band from the titration
Inputs: book_figs/item_analysis.parquet + book_figs/confusion_M.npz (from make_item_analysis.py /
make_confusion.py on ccn2). usage: python src/plot_item_confusion_cues.py <datadir> <outdir>"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from sklearn.manifold import MDS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
INK, SUB, GRID = "#2c2c2a", "#6b6a66", "#e1e0d9"
GREEN, BLUE, RED, AMBER = "#1d9e75", "#185fa5", "#b0655a", "#b8860b"


def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=SUB, length=0)


# ------------------------------------------------------------------ (1) item analysis + trend line
a = pd.read_parquet(f"{D}/item_analysis.parquet")
x = np.log1p(a.freq.values); y = a.acc.values
lr = linregress(x, y)
rho_f = spearmanr(y, x).correlation; rho_v = spearmanr(y, a.proto).correlation
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
ax[0].hist(a.proto, bins=np.linspace(20, 100, 33), color=GREEN)
ax[0].axvline(a.acc.mean(), color=RED, lw=1.5, ls="--")
ax[0].text(a.acc.mean() - 1, ax[0].get_ylim()[1] * 0.8, f"model mean {a.acc.mean():.0f}", color=RED, ha="right", fontsize=9)
ax[0].set_xlabel("vision prototype 4AFC (frozen features)"); ax[0].set_ylabel("# categories")
ax[0].set_title("Frozen features separate every category", fontsize=11, loc="left")
ax[1].scatter(x, y, s=22, color=BLUE, alpha=0.85, zorder=3)
xx = np.linspace(x.min(), x.max(), 50)
ax[1].plot(xx, lr.intercept + lr.slope * xx, color=INK, lw=1.8, zorder=4)
ax[1].text(0.98, 0.04, f"linear fit  r = {lr.rvalue:.2f}", transform=ax[1].transAxes, ha="right", va="bottom", fontsize=9.5, color=INK)
ax[1].axhline(25, color=SUB, lw=1, ls=(0, (5, 4)))
ax[1].text(x.min(), 26.5, "chance", color=SUB, fontsize=8)
ax[1].set_xlabel("log(1 + training frequency)"); ax[1].set_ylabel("model 4AFC")
ax[1].set_title(f"Learning tracks frequency (ρ = {rho_f:.2f}), not vision (ρ = {rho_v:.2f})", fontsize=10.5, loc="left")
for q in ax: clean(q)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_item_analysis.png", bbox_inches="tight"); print("wrote fig_item_analysis.png")

# ------------------------------------------------------------------ (2) confusion RDM, legible
z = np.load(f"{D}/confusion_M.npz", allow_pickle=True)
M, cats, order, clusters = z["M"], list(z["cats"]), z["order"], z["clusters"]
n = len(cats)
# name the clusters we can read (from the printed membership); small/incoherent ones stay unnamed
NAMES = {2: "tools & vehicles", 3: "toys, clothing & household", 6: "animals & nature",
         8: "kitchen & tableware", 4: "furniture", 7: "round / flat"}
CCOL = {2: BLUE, 3: "#7b5ea7", 6: GREEN, 8: RED, 4: AMBER, 7: "#4c8f9d"}
Mo = M[np.ix_(order, order)].copy()
np.fill_diagonal(Mo, np.nan)                       # the diagonal swamps the scale; mask it
lab = [cats[i] for i in order]; cl = clusters[order]
fig = plt.figure(figsize=(14, 12), dpi=150)
gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 10], height_ratios=[30, 1], wspace=0.02, hspace=0.16)
ax = fig.add_subplot(gs[0, 1]); axb = fig.add_subplot(gs[0, 0], sharey=ax); axc = fig.add_subplot(gs[1, 1])
cmap = plt.get_cmap("magma").copy(); cmap.set_bad("#f2f1ec")
vmax = np.nanpercentile(Mo, 99)
im = ax.imshow(Mo, cmap=cmap, vmin=np.nanpercentile(Mo, 1), vmax=vmax)
ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=90, fontsize=7.2)
ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=7.2); ax.yaxis.tick_right()   # labels right; brackets left
ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
for t, k in zip(ax.get_yticklabels(), cl): t.set_color(CCOL.get(k, SUB))
for t, k in zip(ax.get_xticklabels(), cl): t.set_color(CCOL.get(k, SUB))
# bracket axis: shares the row coordinate; bar at x=1, name to its left
axb.set_xlim(0, 1.0); axb.axis("off")
i = 0
while i < n:
    j = i
    while j + 1 < n and cl[j + 1] == cl[i]: j += 1
    k = cl[i]
    if k in NAMES and j - i >= 2:
        c = CCOL[k]
        axb.plot([0.97, 0.97], [i - 0.4, j + 0.4], color=c, lw=6, solid_capstyle="butt")
        axb.text(0.90, (i + j) / 2, NAMES[k], color=c, fontsize=12, fontweight="bold", ha="right", va="center")
        ax.plot([i - 0.5, j + 0.5, j + 0.5, i - 0.5, i - 0.5], [i - 0.5, i - 0.5, j + 0.5, j + 0.5, i - 0.5], color=c, lw=1.6)
    i = j + 1
ax.set_title("Confusion among learned categories (clustered; diagonal masked)", fontsize=13, loc="left", color=INK, pad=14)
cb = fig.colorbar(im, cax=axc, orientation="horizontal")
cb.set_label("word–image score (off-diagonal; brighter = more confusable)", color=SUB, fontsize=10)
fig.savefig(f"{OUT}/fig_confusion.png", bbox_inches="tight"); print("wrote fig_confusion.png")

# MDS map of the same confusion — the legible "semantic structure" view
S = (M + M.T) / 2
Zs = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
Dm = 1 - np.corrcoef(Zs); np.fill_diagonal(Dm, 0); Dm = np.clip((Dm + Dm.T) / 2, 0, None)
xy = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=8).fit_transform(Dm)
fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
for k in sorted(set(clusters)):
    idx = np.where(clusters == k)[0]
    c = CCOL.get(k, "#b8b6ae")
    ax.scatter(xy[idx, 0], xy[idx, 1], s=42, color=c, alpha=0.9, zorder=3, label=NAMES.get(k))
    for i in idx:
        ax.annotate(cats[i], xy[i], xytext=(4, 3), textcoords="offset points", fontsize=8.5, color=c if k in NAMES else SUB)
ax.legend(frameon=False, fontsize=10, loc="upper left", title="confusion clusters", title_fontsize=10)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color(GRID)
ax.set_title("Semantic structure of the learned vocabulary (MDS of the confusion matrix)", fontsize=12.5, loc="left", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_confusion_mds.png", bbox_inches="tight"); print("wrote fig_confusion_mds.png")

# ------------------------------------------------------------------ (3) cue information vs ignition
# From ch5 Condition-0 table. AUC → rank-biserial ρ = 2·AUC − 1 (the mapping ch5 uses: ρ 0.3 ≈ AUC 0.65).
CUES = [  # (label, rho, kind)
    ("discourse continuity", 0.14, "language"),
    ("combined language predictor", 0.11, "language"),
    ("prosody (energy range)", 0.05, "language"),
    ("person present", 2 * 0.55 - 1, "social"),
    ("caregiver speaking", 2 * 0.54 - 1, "social"),
    ("pose gesture (point / show)", 2 * 0.53 - 1, "social"),
    ("child's hand present", 2 * 0.53 - 1, "social"),
    ("visible face", 2 * 0.51 - 1, "social"),
    ("caregiver gaze direction", 2 * 0.50 - 1, "social"),
    ("adult hand / reach / pointing", 2 * 0.50 - 1, "social"),
]
CUES.sort(key=lambda t: t[1])
KC = {"language": GREEN, "social": BLUE}
fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=150)
ax.axvspan(0.3, 0.5, color=RED, alpha=0.10, zorder=1)
ax.text(0.4, len(CUES) - 0.4, "ignition band\n(titration: cue used as a hard gate)", color=RED, ha="center", va="top", fontsize=9.5)
ax.axvline(0.3, color=RED, lw=1, ls=(0, (4, 3)))
for i, (lab_, r, k) in enumerate(CUES):
    ax.plot([0, r], [i, i], color=KC[k], lw=2.2, zorder=2)
    ax.scatter([r], [i], s=70, color=KC[k], zorder=3)
    ax.text(r + 0.012, i, f"{r:.2f}", va="center", fontsize=9, color=KC[k])
ax.set_yticks(range(len(CUES))); ax.set_yticklabels([c[0] for c in CUES], fontsize=10.5)
for t, (_, _, k) in zip(ax.get_yticklabels(), CUES): t.set_color(KC[k])
ax.set_xlim(-0.01, 0.55); ax.set_ylim(-0.7, len(CUES) - 0.3)
ax.set_xlabel("information about referential alignment  (rank correlation ρ with Gemini gold)", fontsize=10.5, color=SUB)
ax.set_title("No accessible cue carries enough information to select referential moments", fontsize=12, loc="left", color=INK)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color=GREEN, lw=0, ms=8, label="language cues"),
                   Line2D([], [], marker="o", color=BLUE, lw=0, ms=8, label="social / visual cues (AUC → ρ)")],
          frameon=False, loc="lower right", fontsize=9.5)
clean(ax); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_cues.png", bbox_inches="tight"); print("wrote fig_cues.png")
