"""SI display item — the titration that defines the ignition band cited in fig3B.

A synthetic cue of controlled quality (rho with the true alignment) is used two ways: as a hard
gate that keeps only the pairs it flags, and as a soft weight on the loss. Soft weighting never
ignites — it drifts up almost linearly and stays low. Gating is WORSE than soft weighting for a
noisy cue (it throws away good pairs to act on a bad signal) and only overtakes it once the cue
crosses rho ~ 0.3-0.5. That crossover is the ignition band. The best cue we can actually read off
the stream is 0.14, well below it (fig3B).

NB this is the Phase-2 rig (old CLIP-era detector eval), so the ABSOLUTE accuracies here are not
comparable to the Konkle ladder. The crossover is an internal comparison and survives the rig
change; that is the only thing the figure claims.
"""
import sys, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
tit = pd.read_csv(R / "titration.csv")
cues = pd.read_csv(R / "cues.csv")
IGN_LO, IGN_HI = D.ignition_band()

fig, ax = plt.subplots(figsize=(T.W1, 2.5))

ax.axvspan(IGN_LO, IGN_HI, color=T.ORACLE, alpha=0.10, lw=0, zorder=1)
for mode, col, mk, lab in [("gate", T.ORACLE, "o", "gate (hard filter)"),
                           ("soft", T.NEUTRAL, "s", "soft (soft weight)")]:
    d = tit[tit["mode"] == mode].sort_values("rho")
    ax.plot(d.rho, 100 * d.acc_4afc, color=col, marker=mk, ms=2.8, lw=1.0, label=lab, zorder=3)

best = cues.rho.max()
ax.axvline(best, color=T.FREE, lw=0.8, zorder=2)
ax.text(best + 0.02, 37.9, f"best accessible\ncue  ρ = {best:.2f}", color=T.FREE, fontsize=5.8,
        va="top")
ax.text((IGN_LO + IGN_HI) / 2, 32.6, "ignition\nband", color=T.ORACLE, ha="center", va="bottom",
        fontsize=5.8)
ax.set_xlabel("cue quality (ρ with true alignment)")
ax.set_ylabel("4AFC (%)  ·  Phase-2 rig")
ax.set_xlim(-0.03, 1.03)
ax.legend(loc="upper left", fontsize=6, handletextpad=0.5, borderpad=0.2)
T.clean(ax)

fig.text(0.5, -0.14, "Phase-2 rig (CLIP-era eval): absolute values are not comparable to the "
         "Konkle ladder.\nGating only overtakes soft weighting above ρ ≈ "
         f"{IGN_LO:.2f}–{IGN_HI:.2f} — the ignition band used in fig3B.",
         ha="center", va="top", fontsize=5.5, color=T.SUB)
T.save(fig, "fig4_cues")
