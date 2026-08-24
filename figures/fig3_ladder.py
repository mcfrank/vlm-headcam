"""Display item 3 — the referential gap on the production rig.

The B26 ladder: four training manifests, same 1.82M-pair English-filtered corpus, same model
and eval throughout. base = every pair as spoken (what a learner gets for free). The oracle
rungs answer, in turn: which moments are referential (filtnat), which word is the referent
(t15), which object it names (t2 — text is the label itself, the clean-label ceiling).

Read from runs.parquet directly (published.csv claim ids still point at the 2025.2 ladder;
coordinate with the pipeline session before repointing).
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

RUNGS = [("B26_lad_base", "unfiltered\n(region MIL)", False),
         ("B26_lad_filtnat", "+ alignment\nfilter", True),
         ("B26_lad_t15", "+ word\nselection", True),
         ("B26_lad_t2", "+ vision\nbinding", True)]
fam = {f: D.family(f) for f, _, _ in RUNGS}
if any(str(f).startswith("B26_lad1") or str(f).startswith("B26_lad3") for f in D.runs.family.unique()):
    print("  NOTE fig3: B26 ladder-at-scale families exist (300k/100k) — panel candidate once "
          "all rungs and seeds are in")

fig, ax = plt.subplots(figsize=(T.W1, 2.5))
base = fam["B26_lad_base"]["mean"]
ceil = fam["B26_lad_t2"]["mean"]
FLOOR, TOP = 74, 88.5
X0, SPLIT, X1 = -0.62, 0.5, len(RUNGS) - 0.5

ax.axvspan(X0, SPLIT, color=T.FREE, alpha=0.06, lw=0, zorder=0)
ax.axvspan(SPLIT, X1, color=T.ORACLE, alpha=0.06, lw=0, zorder=0)
ax.text(X0 + 0.10, TOP - 0.35, "free", color=T.FREE, fontsize=6.5, style="italic", va="top")
ax.text(SPLIT + 0.10, TOP - 0.35, "oracle", color=T.ORACLE, fontsize=6.5, style="italic", va="top")

prev = None
for k, (f, lab, is_or) in enumerate(RUNGS):
    v, sd = fam[f]["mean"], fam[f]["sd"]
    bot = FLOOR if k == 0 else prev
    lo, hi = min(bot, v), max(bot, v)
    ax.bar(k, hi - lo, bottom=lo, width=0.62, color=T.ORACLE if is_or else T.FREE, zorder=3)
    ax.errorbar(k, v, yerr=sd, fmt="none", ecolor=T.INK, elinewidth=0.7, capsize=2, zorder=5)
    if k:
        d = v - bot
        if d >= 0:
            ax.text(k, max(hi, v + sd) + 0.35, f"{d:+.1f}", ha="center", fontsize=6, color=T.INK)
        else:
            ax.text(k, min(lo, v - sd) - 0.35, f"{d:+.1f}", ha="center", va="top", fontsize=6,
                    color=T.INDOM)
    ax.text(k, lo + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=6, color="white",
            fontweight="bold", zorder=4)
    prev = v

ax.plot([X0, X1 + 0.1], [ceil, ceil], color=T.SUB, lw=0.6, ls=(0, (2, 2)), zorder=2)
ax.text(X0 + 0.10, ceil - 0.35, "clean-label ceiling", ha="left", va="top", fontsize=5.8, color=T.SUB)

HX = X1 + 0.40
ax.annotate("", xy=(HX, base), xytext=(HX, ceil),
            arrowprops=dict(arrowstyle="<->", color=T.SUB, lw=0.7, shrinkA=0, shrinkB=0))
ax.text(HX + 0.28, (base + ceil) / 2, f"referential headroom\n+{ceil - base:.1f}",
        fontsize=5.6, color=T.SUB, va="center", ha="center", rotation=90, linespacing=1.3)

for k, q in zip((1, 2, 3), ("which\nmoments", "which\nword", "which\nobject")):
    ax.text(k, FLOOR + 0.4, q, ha="center", fontsize=5.4, color=T.ORACLE, style="italic")
ax.set_xticks(range(len(RUNGS)))
ax.set_xticklabels([lab for _, lab, _ in RUNGS], fontsize=6.0)
ax.set_xlim(X0, HX + 0.55); ax.set_ylim(FLOOR, TOP)
ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
print("  NOTE fig3: B26 ladder, n per rung =", [fam[f]["n"] for f, _, _ in RUNGS],
      "· word selection is a negative step on this rig")
T.save(fig, "fig3_ladder")
