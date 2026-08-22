"""Display item 3 — the alignment ladder: what the learner gets free, and what the oracle buys.

Free rungs (green) stack; oracle rungs (indigo) branch from the region-MIL baseline. The gap
between the top free rung and the clean-label ceiling is the referential headroom: everything the
learner would gain if it knew which moments, which word, which object. The cue analysis that
shows no accessible signal closes it is fig4.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

FREE = ["ladder_pure", "ladder_region", "ladder_frame2"]
ORACLE = ["ladder_filter", "ladder_word", "ladder_vision"]
SHORT = {"ladder_pure": "whole-\nframe", "ladder_region": "+ region\nMIL",
         "ladder_frame2": "+ frame\nMIL ±2s", "ladder_filter": "+ align\nfilter",
         "ladder_word": "+ word\nselect", "ladder_vision": "+ vision\nbind"}

fig, ax = plt.subplots(figsize=(T.W15, 2.6))

ids = FREE + ORACLE
vals = {i: D.claim(i) for i in ids}
base = vals["ladder_region"]["value"]           # oracle branches from region-MIL
ceil = vals["ladder_vision"]["value"]
FLOOR, TOP = 55, 86
X0, SPLIT, X1 = -0.62, len(FREE) - 0.5, len(ids) - 0.5

# the two regimes as background bands, so the free/oracle split is structural, not a floating label
ax.axvspan(X0, SPLIT, color=T.FREE, alpha=0.06, lw=0, zorder=0)
ax.axvspan(SPLIT, X1, color=T.ORACLE, alpha=0.06, lw=0, zorder=0)
ax.text(X0 + 0.12, TOP - 0.8, "free", color=T.FREE, fontsize=6.5, style="italic", va="top")
ax.text(SPLIT + 0.12, TOP - 0.8, "oracle", color=T.ORACLE, fontsize=6.5, style="italic", va="top")

prev = None
for k, i in enumerate(ids):
    v, is_or = vals[i]["value"], i in ORACLE
    bot = FLOOR if k == 0 else (base if i == "ladder_filter" else prev)
    lo, hi = min(bot, v), max(bot, v)
    ax.bar(k, hi - lo, bottom=lo, width=0.62, color=T.ORACLE if is_or else T.FREE, zorder=3,
           edgecolor=T.PROV if vals[i]["provisional"] else "none",
           linewidth=1.0, linestyle=(0, (1.5, 1)) if vals[i]["provisional"] else "-")
    if k:
        ax.text(k, hi + 1.0, f"+{v - bot:.1f}", ha="center", fontsize=6, color=T.INK)
    if hi - lo >= 2.2:                       # tall enough to hold the running total inside
        ax.text(k, lo + 0.6, f"{v:.1f}", ha="center", fontsize=6, color="white", fontweight="bold")
    else:                                    # a sliver: put it under the bar instead
        ax.text(k, lo - 1.5, f"{v:.1f}", ha="center", va="top", fontsize=6, color=T.INK)
    prev = v

ax.plot([X0, X1 + 0.1], [ceil, ceil], color=T.SUB, lw=0.6, ls=(0, (2, 2)), zorder=2)
ax.text(X0 + 0.12, ceil + 0.7, "clean-label ceiling", ha="left", fontsize=5.8, color=T.SUB)

# referential headroom, parked in the right margin so it cannot collide with the delta labels
HX = X1 + 0.42
ax.annotate("", xy=(HX, base), xytext=(HX, ceil),
            arrowprops=dict(arrowstyle="<->", color=T.SUB, lw=0.7, shrinkA=0, shrinkB=0))
ax.text(HX + 0.34, (base + ceil) / 2, f"referential headroom\n+{ceil - base:.1f}",
        fontsize=5.8, color=T.SUB, va="center", ha="center", rotation=90)

for k, q in zip(range(len(FREE), len(ids)), ("which\nmoments", "which\nword", "which\nobject")):
    ax.text(k, FLOOR + 0.7, q, ha="center", fontsize=5.4, color=T.ORACLE, style="italic")
ax.set_xticks(range(len(ids)))
ax.set_xticklabels([SHORT[i] for i in ids], fontsize=6.0)
ax.set_xlim(X0, HX + 0.62); ax.set_ylim(FLOOR, TOP)
ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)

note = D.provisional_note(ids)
if note:
    print("  NOTE fig3 (dashed outline):", note)
T.save(fig, "fig3_ladder")
