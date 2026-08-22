"""Display item 4 — no accessible cue carries enough information to select referential moments.

A: every cue we can read from the stream, on one information scale (rank correlation with the
   Gemini gold), against the ignition band.
B: the titration that defines the band — a synthetic cue of controlled rho, used as a hard gate
   vs a soft weight. Gating is worse than soft weighting for a noisy cue and only overtakes it
   once rho crosses ~0.3–0.5 (data.ignition_band). The best real cue is 0.14.

NB panel B is Phase-2 (old rig, CLIP-era detector eval), so its ABSOLUTE accuracies are not
comparable to the Konkle ladder. The crossover is an internal comparison and survives.
"""
import sys, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
cues = pd.read_csv(R / "cues.csv").sort_values("rho")
tit = pd.read_csv(R / "titration.csv")
IGN_LO, IGN_HI = D.ignition_band()
best = cues.rho.max()

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.30))

# ---- A: cue information ----------------------------------------------------------
ax.axvspan(IGN_LO, IGN_HI, color=T.ORACLE, alpha=0.10, lw=0, zorder=1)
ax.axvline(IGN_LO, color=T.ORACLE, lw=0.7, ls=(0, (3, 2)), zorder=2)
for i, r in enumerate(cues.itertuples()):
    ax.plot([0, r.rho], [i, i], color=T.FREE, lw=1.0, zorder=3)
    ax.scatter([r.rho], [i], s=15, color=T.FREE, zorder=4,
               marker="o" if r.kind == "language" else "s")
ax.set_yticks(range(len(cues)))
ax.set_yticklabels(cues.cue, fontsize=5.8)
ax.set_xlim(-0.008, max(0.56, IGN_HI + 0.06)); ax.set_ylim(-0.8, len(cues) - 0.2)
ax.set_xlabel("information about alignment  (ρ with Gemini gold)")
ax.text((IGN_LO + IGN_HI) / 2, (len(cues) - 1) / 2, "ignition\nband", color=T.ORACLE,
        ha="center", va="center", fontsize=6)
ax.legend(handles=[Line2D([], [], marker="o", color=T.FREE, lw=0, ms=3.2, label="language"),
                   Line2D([], [], marker="s", color=T.FREE, lw=0, ms=3.2, label="social / visual")],
          loc="lower left", bbox_to_anchor=(0.14, 0.0), fontsize=5.8,
          handletextpad=0.4, borderpad=0.2)
T.clean(ax, grid_axis="x")

# ---- B: titration ----------------------------------------------------------------
bx.axvspan(IGN_LO, IGN_HI, color=T.ORACLE, alpha=0.10, lw=0, zorder=1)
for mode, col, mk, lab in [("gate", T.ORACLE, "o", "gate (hard filter)"),
                           ("soft", T.NEUTRAL, "s", "soft (soft weight)")]:
    d = tit[tit["mode"] == mode].sort_values("rho")
    bx.plot(d.rho, 100 * d.acc_4afc, color=col, marker=mk, ms=2.8, lw=1.0, label=lab, zorder=3)
bx.axvline(best, color=T.FREE, lw=0.8, zorder=2)
bx.text(best + 0.02, 37.9, f"best accessible\ncue  ρ = {best:.2f}", color=T.FREE, fontsize=5.8,
        va="top")
bx.set_xlabel("cue quality (ρ with true alignment)")
bx.set_ylabel("4AFC (%)  ·  Phase-2 rig")
bx.set_xlim(-0.03, 1.03)
bx.legend(loc="upper left", fontsize=6, handletextpad=0.5, borderpad=0.2)
T.clean(bx)

for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.14 if a is bx else -0.55)
print("  NOTE fig4: panel B is Phase-2 rig; absolute values not comparable to the Konkle ladder")
T.save(fig, "fig4_cues")
