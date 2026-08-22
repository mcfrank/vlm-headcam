"""Display item 3 — the referential gap, and the elimination argument that nothing free closes it.

A: the ladder. Free rungs (green) stack; oracle rungs (indigo) branch from the region-MIL
   baseline. The gap between the top free rung and the clean-label ceiling is the referential
   headroom: everything the learner would gain if it knew which moments, which word, which object.
B: the elimination. Every cue we can actually read off the stream, on one information scale
   (rank correlation with the Gemini gold), against the ignition band from the titration — the
   cue quality at which hard gating starts to beat soft weighting at all. No accessible cue is
   within a factor of two of the band, so the first oracle rung is not something a learner could
   have taken.
"""
import sys, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"

FREE = ["ladder_pure", "ladder_region", "ladder_frame2"]
ORACLE = ["ladder_filter", "ladder_word", "ladder_vision"]
SHORT = {"ladder_pure": "whole-\nframe", "ladder_region": "+ region\nMIL",
         "ladder_frame2": "+ frame\nMIL ±2s", "ladder_filter": "+ align\nfilter",
         "ladder_word": "+ word\nselect", "ladder_vision": "+ vision\nbind"}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.75),
                             gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.40))

# ---- A: ladder ------------------------------------------------------------------
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

# ---- B: no accessible cue reaches the ignition band ------------------------------
cues = pd.read_csv(R / "cues.csv").sort_values("rho")
IGN_LO, IGN_HI = D.ignition_band()

bx.axvspan(IGN_LO, IGN_HI, color=T.ORACLE, alpha=0.10, lw=0, zorder=1)
bx.axvline(IGN_LO, color=T.ORACLE, lw=0.7, ls=(0, (3, 2)), zorder=2)
for i, r in enumerate(cues.itertuples()):
    bx.plot([0, r.rho], [i, i], color=T.FREE, lw=1.0, zorder=3)
    bx.scatter([r.rho], [i], s=15, color=T.FREE, zorder=4,
               marker="o" if r.kind == "language" else "s")
best = cues.iloc[-1]
bx.annotate(f"best accessible cue  ρ = {best.rho:.2f}", (best.rho, len(cues) - 1),
            (best.rho + 0.025, len(cues) - 1), fontsize=5.6, color=T.FREE,
            va="center", ha="left")
bx.text((IGN_LO + IGN_HI) / 2, (len(cues) - 1) / 2,
        "ignition\nband", color=T.ORACLE, ha="center", va="center", fontsize=6)
bx.set_yticks(range(len(cues)))
bx.set_yticklabels(cues.cue, fontsize=5.8)
bx.set_xlim(-0.008, max(0.56, IGN_HI + 0.06)); bx.set_ylim(-0.8, len(cues) - 0.2)
bx.set_xlabel("information about which moments are referential\n(ρ with the Gemini gold)")
bx.legend(handles=[Line2D([], [], marker="o", color=T.FREE, lw=0, ms=3.2, label="language"),
                   Line2D([], [], marker="s", color=T.FREE, lw=0, ms=3.2, label="social / visual")],
          loc="lower left", bbox_to_anchor=(0.14, 0.0), fontsize=5.8,
          handletextpad=0.4, borderpad=0.2)
T.clean(bx, grid_axis="x")

for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.14)

cap = ("B: the ignition band (ρ "
       f"{IGN_LO:.2f}–{IGN_HI:.2f}) is where a hard gate first beats soft weighting in the "
       "titration (Phase-2 rig; SI). No cue a learner\ncould read off the stream comes close, so "
       "the first oracle rung is not one it could have climbed.")
note = D.provisional_note(ids)
if note:
    cap += "\nA: dashed outline = " + note
fig.text(0.5, -0.16, cap, ha="center", va="top", fontsize=5.5, color=T.SUB)
T.save(fig, "fig3_ladder")
