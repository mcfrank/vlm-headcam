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

fig, ax = plt.subplots(figsize=(T.W15, 2.6))

# ---- A: ladder ----------------------------------------------------------------
ids = FREE + ORACLE
vals = {i: D.claim(i) for i in ids}
base = vals["ladder_region"]["value"]           # oracle branches from region-MIL
FLOOR = 55
prev = None
for k, i in enumerate(ids):
    v, is_or = vals[i]["value"], i in ORACLE
    col = T.ORACLE if is_or else T.FREE
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
ax.set_xticklabels([SHORT[i] for i in ids], fontsize=6.2)
ax.set_ylim(FLOOR, 86); ax.set_ylabel("Konkle 4AFC (%)")
ax.text(0.5, 0.96, "free", transform=ax.transAxes, color=T.FREE, fontsize=6.5,
        ha="right", style="italic")
ax.text(0.56, 0.96, "oracle", transform=ax.transAxes, color=T.ORACLE, fontsize=6.5,
        ha="left", style="italic")
ax.annotate("", xy=(2.6, base), xytext=(2.6, vals["ladder_vision"]["value"]),
            arrowprops=dict(arrowstyle="<->", color=T.SUB, lw=0.7))
ax.text(2.72, (base + vals["ladder_vision"]["value"]) / 2,
        f"referential\nheadroom\n+{vals['ladder_vision']['value'] - base:.1f}",
        fontsize=5.8, color=T.SUB, va="center")
for k, q in zip((3, 4, 5), ("which\nmoments", "which\nword", "which\nobject")):
    ax.text(k, FLOOR + 0.8, q, ha="center", fontsize=5.4, color=T.ORACLE, style="italic")
T.clean(ax)

for a, l in zip((ax,), ("",)):
    pass

note = D.provisional_note(ids)
if note:
    fig.text(0.5, -0.12, "dashed outline = " + note, ha="center", fontsize=5.6, color=T.PROV)
T.save(fig, "fig3_ladder")
