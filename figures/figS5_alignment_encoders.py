"""S5 — the alignment result across all four encoders.

fig2B for every encoder: the aligned arm (independent subsamples of the referential pairs)
against that encoder's unfiltered curve. Aligned arms are points joined, not fitted — four
points cannot identify a ceiling. Arrows give the data equivalence read off the unfiltered
fit alone, where the aligned level is reachable at all.
"""
import sys, re, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scaling_fit import fit, logistic, CHANCE

ENC = [("dinov3l", "L-OTS", "DINOv3-L off-the-shelf", T.OTHER),
       ("dinov3b", "B-OTS", "DINOv3-B off-the-shelf", T.FREE),
       ("vitb_bv", "B-BV", "ViT-B BabyView-trained", T.INDOM),
       ("vits_bv", "S-BV", "ViT-S BabyView-trained", T.INDOM2)]
rng = np.random.default_rng(0)
grid = np.logspace(3.3, 6.6, 220)

fig, axes = plt.subplots(2, 2, figsize=(T.W2, 4.4), sharex=True, sharey=True)
for a, (enc, key, lab, col) in zip(axes.ravel(), ENC):
    F = fit(rng, enc=enc); m_, s_, A_ = F["popt"]
    a.fill_between(grid, *F["band"](grid), color=col, alpha=0.13, lw=0, zorder=1)
    a.plot(grid, logistic(grid, m_, s_, A_), color=col, lw=1.0, zorder=2)
    a.errorbar(F["x"], F["y"], yerr=F["e"], fmt="o", color=col, ms=2.4, lw=0,
               elinewidth=0.6, capsize=1.2, zorder=3)
    af = sorted(((f, D.family(f)) for f in D.runs.family.unique()
                 if re.fullmatch(rf"F_{enc}_align_\d+", str(f))), key=lambda t: t[1]["n_pairs"])
    xa = np.array([f["n_pairs"] for _, f in af]); ya = np.array([f["mean"] for _, f in af])
    ea = np.array([f["sd"] for _, f in af])
    a.errorbar(xa, ya, yerr=ea, color=T.ORACLE, marker="o", ms=3.0, lw=1.2,
               elinewidth=0.7, capsize=1.5, zorder=5)
    a.axvline(xa.max(), color=T.SUB, lw=0.6, ls=(0, (1, 2)), zorder=1)
    reach = lambda yv: 10 ** (m_ - s_ * np.log((A_ - CHANCE) / (yv - CHANCE) - 1))
    labs = []
    for i in range(len(ya)):
        if ya[i] >= A_ - 0.3:
            labs.append("—"); continue
        xu = reach(ya[i]); labs.append(f"{xu / xa[i]:.0f}×")
        if i < 2:
            a.annotate("", xy=(min(xu, 3.6e6), ya[i]), xytext=(xa[i], ya[i]),
                       arrowprops=dict(arrowstyle="-|>", color=T.SUB, lw=0.7,
                                       shrinkA=2, shrinkB=1))
            a.text(np.sqrt(xa[i] * min(xu, 3.6e6)), ya[i] + 1.0, labs[-1], fontsize=5.6,
                   color=T.SUB, ha="center", va="bottom", fontweight="bold")
    a.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
    a.text(0.03, 0.94, lab, transform=a.transAxes, fontsize=6.2, color=col, va="top")
    a.text(0.03, 0.86, "aligned only", transform=a.transAxes, fontsize=5.6,
           color=T.ORACLE, va="top")
    a.set_xscale("log"); a.set_xlim(3e3, 4e6); a.set_ylim(18, 95)
    T.clean(a)
    n_un = int((ya >= A_ - 0.3).sum())
    print(f"  NOTE figS5 {key}: unfiltered asymptote {A_:.1f}; aligned {np.round(ya,1)}; "
          f"equivalence {labs}; {n_un} point(s) unreachable by unfiltered data")
for a in axes[1]:
    a.set_xlabel("training pairs")
for a in axes[:, 0]:
    a.set_ylabel("Konkle 4AFC (%)")
for a, l in zip(axes.ravel(), "ABCD"):
    T.panel(a, l, dx=-0.13)
T.save(fig, "figS5_alignment_encoders")
