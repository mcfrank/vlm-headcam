"""ch6 figures: the random-vs-aligned scaling curves, and the whose-data (diversity) bars.
Aggregate numbers only, safe to commit."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/data2/mcfrank/vlm-headcam/book_figs")
INK, SUB, GRID = "#2c2c2a", "#6b6a66", "#e1e0d9"
RAND, ALIGN = "#b0655a", "#1d9e75"

# --- scaling curves (N, mean, sd) ---
rand = [(10000, 29.3, 3.5), (30000, 38.8, 0.8), (100000, 39.7, 3.1), (300000, 56.3, 2.6),
        (911000, 65.3, 1.7), (1145000, 65.6, 2.2)]   # 1.14M = full corpus (held-out 20% added)
align = [(10000, 63.9, 5.5), (30000, 68.2, 1.3), (85000, 71.5, 1.6)]
fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
for data, col, lab in [(rand, RAND, "random (unfiltered)"), (align, ALIGN, "Gemini-aligned")]:
    x = [d[0] for d in data]; y = [d[1] for d in data]; e = [d[2] for d in data]
    ax.errorbar(x, y, yerr=e, fmt="o-", color=col, ms=7, lw=2, capsize=3, label=lab)
ax.axhline(25, color=SUB, lw=1.1, ls=(0, (5, 4))); ax.text(9000, 25.7, "chance", color=SUB, fontsize=9)
ax.annotate("10k aligned ≈ 900k random", xy=(10000, 63.9), xytext=(30000, 50),
            color=INK, fontsize=10, arrowprops=dict(arrowstyle="->", color=SUB))
ax.set_xscale("log"); ax.set_xlabel("training pairs", fontsize=11, color=SUB)
ax.set_ylabel("Konkle 4AFC", fontsize=11, color=SUB); ax.set_ylim(22, 78)
ax.set_title("Alignment is worth ~2 orders of magnitude of data", fontsize=12.5, color=INK, loc="left", pad=10)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=SUB, length=0)
ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=10, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_scaling_curves.png", bbox_inches="tight"); plt.close(fig)
print("wrote scaling_curves")

# --- whose data ---
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
div = [("1", 31.0, 2.7), ("3", 33.0, 4.4), ("10", 36.9, 3.0), ("36", 34.4, 1.0)]
ax[0].bar(range(4), [d[1] for d in div], yerr=[d[2] for d in div], color=ALIGN, width=0.6, capsize=3)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels([d[0] for d in div])
ax[0].set_xlabel("# children (30k pairs fixed)", fontsize=10); ax[0].set_title("Diversity at fixed count", fontsize=11, color=INK, loc="left")
ax[0].axhline(25, color=SUB, lw=1, ls=(0, (5, 4)))
ax[1].bar([0, 1], [36.7, 43.0], yerr=[2.7, 3.1], color=[RAND, ALIGN], width=0.55, capsize=3)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["1 child\n(110k)", "pooled\n(110k)"])
ax[1].set_title("Within-child ceiling (matched count)", fontsize=11, color=INK, loc="left")
ax[1].axhline(25, color=SUB, lw=1, ls=(0, (5, 4)))
for a in ax:
    a.set_ylim(22, 48); a.set_ylabel("Konkle 4AFC", fontsize=10, color=SUB)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.spines["left"].set_color(GRID); a.spines["bottom"].set_color(GRID); a.tick_params(colors=SUB, length=0)
    a.yaxis.grid(True, color=GRID, lw=0.7); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "fig_whose_data.png", bbox_inches="tight"); plt.close(fig)
print("wrote whose_data")
