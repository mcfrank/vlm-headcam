"""Display item 3 — the alignment ladder and what the oracle buys.

A: the ladder. Free rungs (green) stack; oracle rungs (blue) branch from the region-MIL baseline.
B: the oracle advantage decomposed into the three interventions, with what each would require
   of a learner.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

FREE = ["ladder_pure", "ladder_region", "ladder_frame2"]
ORACLE = ["ladder_filter", "ladder_word", "ladder_vision"]
SHORT = {"ladder_pure": "whole-frame\n(mean-pooled)", "ladder_region": "+ region\nMIL",
         "ladder_frame2": "+ frame\nMIL ±2s", "ladder_filter": "+ alignment\nfilter",
         "ladder_word": "+ word\nselection", "ladder_vision": "+ vision\nbinding"}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.55, 1]))

# ---- A: ladder ----------------------------------------------------------------
ids = FREE + ORACLE
vals = {i: D.claim(i) for i in ids}
base = vals["ladder_region"]["value"]           # oracle branches from region-MIL
FLOOR = 55
prev = None
for k, i in enumerate(ids):
    v, is_or = vals[i]["value"], i in ORACLE
    col = T.BLUE if is_or else T.GREEN
    bot = FLOOR if k == 0 else (base if i == "ladder_filter" else prev)
    lo, hi = min(bot, v), max(bot, v)
    ax.bar(k, hi - lo, bottom=lo, width=0.62, color=col, zorder=3,
           edgecolor=T.PROV if vals[i]["provisional"] else "none",
           linewidth=1.0, linestyle=(0, (1.5, 1)) if vals[i]["provisional"] else "-")
    if k:
        ax.text(k, hi + 0.9, f"+{v - bot:.1f}", ha="center", fontsize=6, color=T.INK)
    ax.text(k, lo + 0.6, f"{v:.1f}", ha="center", fontsize=6, color="white", fontweight="bold")
    prev = v
ax.axhline(vals["ladder_vision"]["value"], color=T.SUB, lw=0.6, ls=(0, (2, 2)))
ax.text(-0.4, vals["ladder_vision"]["value"] + 0.7, "clean-label ceiling",
        ha="left", fontsize=5.8, color=T.SUB)
ax.set_xticks(range(len(ids)))
ax.set_xticklabels([SHORT[i] for i in ids], fontsize=5.8)
ax.set_ylim(FLOOR, 86); ax.set_ylabel("Konkle 4AFC (%)")
ax.text(0.5, 0.96, "free", transform=ax.transAxes, color=T.GREEN, fontsize=6.5,
        ha="right", style="italic")
ax.text(0.56, 0.96, "oracle", transform=ax.transAxes, color=T.BLUE, fontsize=6.5,
        ha="left", style="italic")
T.clean(ax)

# ---- B: what each oracle rung supplies -----------------------------------------
gains = [(SHORT[i].replace("\n", " ").replace("+ ", ""),
          vals[i]["value"] - (base if i == "ladder_filter" else
                              vals[ORACLE[ORACLE.index(i) - 1]]["value"]),
          q) for i, q in zip(ORACLE, ["which moments", "which word", "which object"])]
ys = range(len(gains))
bx.barh(list(ys), [g for _, g, _ in gains], color=T.BLUE, height=0.55, zorder=3)
for y, (lab, g, q) in zip(ys, gains):
    bx.text(g + 0.15, y, f"+{g:.1f}", va="center", fontsize=6.5, color=T.INK)
    bx.text(0.12, y + 0.30, q, fontsize=5.6, color="white", style="italic")
bx.set_yticks(list(ys)); bx.set_yticklabels([l for l, _, _ in gains], fontsize=6.5)
bx.invert_yaxis(); bx.set_xlabel("gain over previous rung (pts)"); bx.set_xlim(0, 9.5)
T.clean(bx, grid_axis="x")
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)

note = D.provisional_note(ids)
if note:
    fig.text(0.5, -0.12, "dashed outline = " + note, ha="center", fontsize=5.6, color=T.PROV)
T.save(fig, "fig3_ladder")
