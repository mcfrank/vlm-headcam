"""Reusable accuracy-waterfall (bridge) figures — a repeating motif for showing what each
intervention contributes. Each rung floats from the previous level to its own; an optional
second series is drawn as reference dots. Aggregate numbers only (no human-subjects data)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/data2/mcfrank/vlm-headcam/book_figs")        # generate here, scp to repo
OUT.mkdir(parents=True, exist_ok=True)
GAIN = "#1d9e75"; ORACLE = "#185fa5"; DOT = "#b0655a"; INK = "#2c2c2a"; SUB = "#6b6a66"; GRID = "#e1e0d9"


def waterfall(steps, out, title, ceiling=None, ceiling_label=None, chance=25,
              primary="Konkle (clean objects)", ref="BabyView detector eval", ylim=(20, 80)):
    """steps: list of (label, primary_value, ref_value). First step is the base (drawn as a
    full bar from chance). Each later step floats from the previous primary value."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=150)
    xs = range(len(steps))
    prev = chance
    for i, (lab, pv, rv) in enumerate(steps):
        col = ORACLE if ("oracle" in lab.lower()) else GAIN
        bot, top = (chance, pv) if i == 0 else (min(prev, pv), max(prev, pv))
        ax.bar(i, top - bot, bottom=bot, width=0.62, color=col, zorder=3)
        d = pv - (chance if i == 0 else prev)
        ax.text(i, bot + 0.72 * (top - bot), "+%.0f" % d, ha="center", va="center",
                color="white", fontsize=12.5, zorder=4)
        ax.text(i, pv + 1.2, "%.0f" % pv, ha="center", va="bottom", color=INK, fontsize=12, zorder=4)
        if i:
            ax.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color=SUB, lw=0.8, ls=(0, (3, 3)), zorder=2)
        prev = pv
    # optional reference series as dots + faint line (skipped if all None)
    rvs = [s[2] for s in steps]
    has_ref = any(r is not None for r in rvs)
    if has_ref:
        rx = [i for i, r in enumerate(rvs) if r is not None]
        ax.plot(rx, [rvs[i] for i in rx], "o-", color=DOT, ms=6, lw=1.4, zorder=5)
    ax.axhline(chance, color=SUB, lw=1.2, ls=(0, (5, 4)))
    ax.text(-0.45, chance + 0.6, "chance", color=SUB, fontsize=10, va="bottom")
    if ceiling:
        ax.axhline(ceiling, color=GRID, lw=1.2)
        ax.text(len(steps) - 1, ceiling - 0.8, ceiling_label or "ceiling", color=SUB, fontsize=10, ha="center", va="top")
    ax.set_xticks(list(xs)); ax.set_xticklabels([s[0] for s in steps], fontsize=10.5)
    ax.set_ylim(*ylim); ax.set_ylabel("4AFC accuracy", fontsize=11, color=SUB)
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=SUB, length=0); ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    # legend (only show the reference entry if a reference series is drawn)
    if has_ref:
        ax.scatter([], [], color=GAIN, marker="s", s=60, label=primary)
        ax.plot([], [], "o-", color=DOT, label=ref)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False, fontsize=9.5)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# ch5 capstone: the bootstrap ladder, clean Konkle eval vs the noisy detector eval
waterfall(
    [("self-supervised\nbootstrap", 43, 34),
     ("+ language\ncue", 48, 35),
     ("oracle\n(alignment given)", 70, 49)],
    OUT / "fig_waterfall_bootstrap.png",
    "What the interventions buy, and what the eval hid",
    ceiling=72, ceiling_label="clean-label ceiling", ylim=(20, 80))

# ch3: the alignment levers that WORK (single series, detector eval)
waterfall(
    [("naive\n(random pairs)", 34, None),
     ("+ alignment\nfilter", 47, None),
     ("+ region\ngrounding", 49, None)],
    OUT / "fig_waterfall_alignment.png",
    "The levers that work: alignment and grounding",
    ceiling=72, ceiling_label="clean-label ceiling", ylim=(20, 80))

# ch6: how much data, and whose (data-matched, same held-out eval)
waterfall(
    [("single\nchild", 33, None),
     ("+ diversity\n(pool children)", 38, None),
     ("+ more data\n(15k → 110k)", 41, None)],
    OUT / "fig_waterfall_data.png",
    "How much data, and whose: diversity beats amount",
    ylim=(20, 52))
