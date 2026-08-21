"""Display item 2 — scaling, in two currencies.

A: 4AFC vs number of training pairs (the engineering axis).
B: the SAME observations rescaled to developmental time (the developmental axis) — because
   aligned pairs are ~9.3% of utterances, buying N aligned pairs costs ~11x more waking hours
   than N random ones, which partly offsets alignment's apparent data-efficiency.

Reconciles the book's fig_scaling_curves (pairs axis) and fig_dev_time (time axis) into one
figure with one set of numbers, drawn from results/ rather than hardcoded.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

# child-input rate constants (as in src/make_dev_figure.py)
UTT_PER_HR, HR_PER_YEAR, ALIGNED_RATE = 820, 4000, 0.093

RAND = [("scale_rand_10k", 1e4), ("scale_rand_30k", 3e4), ("scale_rand_100k", 1e5),
        ("scale_rand_300k", 3e5), ("scale_rand_911k", 9.11e5)]
ALIGN = [("scale_align_10k", 1e4), ("scale_align_30k", 3e4), ("scale_align_85k", 8.5e4)]


def series(spec):
    x, y, e, prov = [], [], [], []
    for cid, n in spec:
        c = D.claim(cid)
        x.append(n); y.append(c["value"]); e.append(c["sd"] or 0); prov.append(c["provisional"])
    return np.array(x), np.array(y), np.array(e), np.array(prov)

rx, ry, re_, rp = series(RAND)
ax_, ay, ae, ap = series(ALIGN)

fig, axes = plt.subplots(1, 2, figsize=(T.W2, 2.5), sharey=True)

for ax, mode in zip(axes, ["pairs", "time"]):
    if mode == "pairs":
        X_r, X_a = rx, ax_
        ax.set_xlabel("training pairs")
    else:
        X_r = rx / UTT_PER_HR / HR_PER_YEAR
        X_a = ax_ / (UTT_PER_HR * ALIGNED_RATE) / HR_PER_YEAR
        ax.set_xlabel("developmental time (years of waking input)")
    for X, Y, E, P, col, lab in [(X_r, ry, re_, rp, T.GREEN, "random (unfiltered)"),
                                 (X_a, ay, ae, ap, T.BLUE, "referentially aligned")]:
        ax.errorbar(X, Y, yerr=E, color=col, lw=1.2, marker="o", ms=3.2, capsize=1.6,
                    elinewidth=0.6, label=lab, zorder=3)
        if P.any():   # provisional points get a hollow ring
            ax.scatter(X[P], Y[P], s=44, facecolors="none", edgecolors=T.PROV,
                       lw=1.0, zorder=4)
    ax.set_xscale("log")
    ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
    T.clean(ax)

axes[0].set_ylabel("Konkle 4AFC (%)")
axes[0].set_ylim(20, 80)
axes[0].legend(loc="upper left")
axes[0].annotate("10k aligned ≈ 900k random", xy=(1e4, ay[0]), xytext=(4e4, 34),
                 fontsize=6, color=T.SUB,
                 arrowprops=dict(arrowstyle="-", color=T.SUB, lw=0.5))
axes[1].text(0.97, 0.13, "aligned pairs cost ~11× the hours\n(only ~9% of utterances align)",
             transform=axes[1].transAxes, ha="right", va="bottom", fontsize=6, color=T.SUB)
for a, l in zip(axes, "AB"):
    T.panel(a, l)

note = D.provisional_note([c for c, _ in RAND + ALIGN])
if note:
    fig.text(0.5, -0.10, "○ " + note, ha="center", fontsize=5.6, color=T.PROV)
fig.text(0.5, -0.17, "Points 10k–300k are old-rig (final-epoch, post-hoc eval); 911k is clean-rig "
         "(best-epoch). See notes/PROVENANCE.md D1 — re-run before submission.",
         ha="center", fontsize=5.6, color=T.SUB)
T.save(fig, "fig2_scaling")
