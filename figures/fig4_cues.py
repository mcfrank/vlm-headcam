"""Display item 4 — no accessible cue carries enough information to select referential moments.

A: every cue we can read from the stream, on one information scale (rank correlation with the
   Gemini gold), against the ignition band established by the titration.
B: the titration itself — a synthetic cue of controlled rho, used as a hard gate vs a soft weight.
   Soft weighting never ignites; gating needs rho ~ 0.3-0.5. The best real cue is 0.14.

NB panel B is Phase-2 (old rig, CLIP-era detector eval), so its ABSOLUTE accuracies are not
comparable to the Konkle ladder. The threshold claim is an internal comparison and survives.
"""
import sys, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
cues = pd.read_csv(R / "cues.csv").sort_values("rho")
tit = pd.read_csv(R / "titration.csv")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
KC = {"language": T.GREEN, "social": T.BLUE}

# ---- A: cue information --------------------------------------------------------
ax.axvspan(0.30, 0.50, color=T.RED, alpha=0.10, zorder=1)
ax.axvline(0.30, color=T.RED, lw=0.7, ls=(0, (3, 2)))
for i, r in enumerate(cues.itertuples()):
    c = KC[r.kind]
    ax.plot([0, r.rho], [i, i], color=c, lw=1.1, zorder=2)
    ax.scatter([r.rho], [i], s=16, color=c, zorder=3)
ax.set_yticks(range(len(cues)))
ax.set_yticklabels(cues.cue, fontsize=6)
for t, k in zip(ax.get_yticklabels(), cues.kind):
    t.set_color(KC[k])
ax.set_xlim(-0.01, 0.56); ax.set_ylim(-0.8, len(cues) - 0.2)
ax.set_xlabel("information about alignment  (ρ with Gemini gold)")
ax.text(0.40, len(cues) - 0.6, "ignition\nband", color=T.RED, ha="center", va="top", fontsize=6)
ax.legend(handles=[Line2D([], [], marker="o", color=T.GREEN, lw=0, ms=3.5, label="language"),
                   Line2D([], [], marker="o", color=T.BLUE, lw=0, ms=3.5, label="social / visual")],
          loc="lower right", fontsize=6)
T.clean(ax, grid_axis="x")

# ---- B: titration --------------------------------------------------------------
bx.axvspan(0.30, 0.50, color=T.RED, alpha=0.10, zorder=1)
for mode, col, mk in [("gate", T.BLUE, "o"), ("soft", T.SUB, "s")]:
    d = tit[tit["mode"] == mode]
    bx.plot(d.rho, 100 * d.acc_4afc, color=col, marker=mk, ms=3, lw=1.1,
            label=f"{mode} ({'hard filter' if mode=='gate' else 'soft weight'})", zorder=3)
best_real = cues.rho.max()
bx.axvline(best_real, color=T.GREEN, lw=0.8)
bx.text(best_real + 0.02, 32.9, f"best real cue (ρ={best_real:.2f})", color=T.GREEN, fontsize=5.8, va="bottom")
bx.set_xlabel("cue quality (ρ with true alignment)")
bx.set_ylabel("4AFC (%)  ·  Phase-2 rig")
bx.legend(loc="upper left", fontsize=6)
T.clean(bx)
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)
fig.text(0.5, -0.10, "Panel B is old-rig (CLIP-era eval); absolute values are not comparable to "
         "the Konkle ladder — the threshold is an internal comparison.",
         ha="center", fontsize=5.6, color=T.SUB)
T.save(fig, "fig4_cues")
