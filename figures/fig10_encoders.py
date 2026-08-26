"""Display item 10 — fig2's scaling curve under three encoders.

Same conventions as fig2 (free-asymptote logistic in log N, seed-noise band, chance floor),
one curve per encoder on the SAME training draws: DINOv3-B off-the-shelf (the fig2 curve),
DINOv3-L off-the-shelf, and DINOv3-L retrained from scratch on BabyView — the in-domain
encoder, and the negative result. Full-corpus points are the ladder base runs.

Preview corpus (transcript-filter): these families re-land under new names after the
audio-filter rerun — repoint the regexes then.
"""
import sys, re, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scaling_fit import logistic, CHANCE

ENCODERS = [  # label, scaling-family regex, full-corpus family, color
    ("DINOv3-B\noff-the-shelf", r"B26_rand_(\d+)", "B26_lad_base", T.FREE),
    ("DINOv3-L\noff-the-shelf", r"C8_dinov3l_grid4x4_rand_(\d+)", "C8_dinov3l_grid4x4_base", T.OTHER),
    ("DINOv3-L\nBabyView-trained", r"C8_dinov3l_bv_grid4x4_rand_(\d+)", "C8_dinov3l_bv_grid4x4_base", T.INDOM),
]
rng = np.random.default_rng(0)
grid = np.logspace(3.3, 7.8, 260)

fig, ax = plt.subplots(figsize=(T.W15, 2.6))
for lab, rx, full, col in ENCODERS:
    fams = sorted((int(m.group(1)), f) for f in D.runs.family.unique()
                  if (m := re.fullmatch(rx, str(f))))
    fams.append((None, full))
    fam = [D.family(f) for _, f in fams]
    x = np.array([fm["n_pairs"] if not np.isnan(fm.get("n_pairs", np.nan)) else n
                  for (n, _), fm in zip(fams, fam)], float)
    y = np.array([fm["mean"] for fm in fam]); e = np.array([fm["sd"] for fm in fam])
    popt, pcov = curve_fit(logistic, x, y, p0=(5, 0.8, 85), sigma=np.maximum(e, 0.5),
                           bounds=([3, 0.05, 26], [9, 3, 100]), maxfev=60000)
    draws = []
    for _ in range(400):
        try:
            p_, _ = curve_fit(logistic, x, y + rng.normal(0, np.maximum(e, 0.5)), p0=popt,
                              bounds=([3, 0.05, 26], [9, 3, 100]), maxfev=60000)
            draws.append(logistic(grid, *p_))
        except Exception:
            pass
    lo, hi = np.percentile(np.array(draws), [10, 90], axis=0)
    ax.fill_between(grid, lo, hi, color=col, alpha=0.13, lw=0, zorder=1)
    ax.plot(grid, logistic(grid, *popt), color=col, lw=1.0, zorder=2)
    ax.errorbar(x, y, yerr=e, fmt="o", color=col, ms=2.9, lw=0, elinewidth=0.7,
                capsize=1.5, zorder=4)
    A = popt[2]
    va = "top" if "B\n" in lab else "bottom"                  # B-OTS labels below its curve
    ax.text(1.6e7, A + (-2.6 if va == "top" else 2.2), lab.replace("\n", " "), fontsize=5.6,
            color=col, ha="right", va=va)
    print(f"  NOTE fig10 {lab.replace(chr(10), ' ')}: asymptote "
          f"{popt[2]:.1f} ± {np.sqrt(pcov[2, 2]):.1f}, n per point {[fm['n'] for fm in fam]}")

ax.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(3.5e6, 22.2, "chance", fontsize=5.6, color=T.SUB, ha="right")
ax.set_xscale("log"); ax.set_xlim(3e3, 2e7); ax.set_ylim(18, 95)
ax.set_xlabel("training pairs"); ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
T.save(fig, "fig10_encoders")
