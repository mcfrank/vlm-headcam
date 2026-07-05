"""Waterfalls for the bootstrapping ladder (ch4). Aggregate-only, safe to commit."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/data2/mcfrank/vlm-headcam/book_figs")
OUT.mkdir(parents=True, exist_ok=True)
FREE, ORACLE, NULLC = "#1d9e75", "#185fa5", "#b0655a"
INK, SUB, GRID = "#2c2c2a", "#6b6a66", "#e1e0d9"


def waterfall(steps, out, title, oracle_from=None, chance=25, ceiling=None, ylim=(20, 88)):
    fig, ax = plt.subplots(figsize=(8.4, 4.5), dpi=150)
    prev = chance
    for i, (lab, val) in enumerate(steps):
        oracle = oracle_from is not None and i >= oracle_from
        col = ORACLE if oracle else FREE
        bot, top = (chance, val) if i == 0 else (min(prev, val), max(prev, val))
        ax.bar(i, top - bot, bottom=bot, width=0.64, color=col, zorder=3)
        d = val - (chance if i == 0 else prev)
        lbl = ("%.0f" % val) if i == 0 else ("+%.1f" % d)
        ax.text(i, bot + 0.5 * (top - bot), lbl, ha="center", va="center", color="white",
                fontsize=11.5, zorder=4, fontweight="bold")
        ax.text(i, val + 0.9, "%.1f" % val, ha="center", va="bottom", color=INK, fontsize=11, zorder=4)
        if i:
            ax.plot([i - 1 + 0.32, i - 0.32], [prev, prev], color=SUB, lw=0.8, ls=(0, (3, 3)), zorder=2)
        prev = val
    ax.axhline(chance, color=SUB, lw=1.1, ls=(0, (5, 4)))
    ax.text(-0.45, chance + 0.5, "chance", color=SUB, fontsize=9, va="bottom")
    if ceiling:
        ax.axhline(ceiling, color=SUB, lw=1.0, ls=(0, (2, 2)))
        ax.text(len(steps) - 1, ceiling + 0.4, "perfect labels can't exceed ~%.0f" % ceiling,
                color=SUB, fontsize=8.5, ha="right", va="bottom")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=10)
    ax.set_ylim(*ylim); ax.set_ylabel("Konkle 4AFC", fontsize=10, color=SUB)
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=SUB, length=0); ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    if oracle_from is not None:
        ax.bar([], [], color=FREE, label="free (no labels)")
        ax.bar([], [], color=ORACLE, label="oracle (upper bound)")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


# (a) the full ladder
waterfall([("pure\nlearning", 52.9), ("+ region\nMIL", 62.6), ("+ frame\nMIL", 65.0),
           ("+ alignment\nfilter", 68.5), ("+ word\nselection", 73.2), ("+ vision\nbinding", 81.3)],
          OUT / "fig_waterfall_ladder.png", "The bootstrapping ladder (Konkle test-60, 3 seeds)",
          oracle_from=3, ceiling=81, ylim=(20, 88))

# (b) the label decomposition (from the region-MIL baseline)
waterfall([("region-MIL\nbaseline", 62.6), ("+ filter", 68.5), ("+ word\nselection", 73.2),
           ("+ vision\nbinding", 81.3)],
          OUT / "fig_waterfall_label.png", "Where the +19 label headroom lives",
          oracle_from=1, ylim=(55, 88))

# (c) what each bootstrap buys
fig, ax = plt.subplots(figsize=(7.4, 2.9), dpi=150)
mechs = [("region MIL\n(where in frame)", 9.7, FREE),
         ("frame MIL\n(which moment)", 2.4, FREE),
         ("utterance EM\n(which pairs)", 1.2, NULLC)]
for y, (lab, d, c) in enumerate(mechs):
    ax.barh(y, d, color=c, height=0.62)
    ax.text(d + 0.15, y, "+%.1f" % d, va="center", fontsize=11, color=INK)
ax.set_yticks(range(len(mechs))); ax.set_yticklabels([m[0] for m in mechs], fontsize=10)
ax.invert_yaxis(); ax.set_xlim(0, 11); ax.set_xlabel("Konkle 4AFC gain", fontsize=10, color=SUB)
ax.set_title("What each bootstrap buys (green = free, red = null within noise)",
             fontsize=11.5, color=INK, loc="left", pad=8)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=SUB, length=0)
fig.tight_layout(); fig.savefig(OUT / "fig_waterfall_mechanisms.png", bbox_inches="tight")
print("wrote", OUT / "fig_waterfall_mechanisms.png")
