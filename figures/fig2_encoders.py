"""Display item 2 — the scaling curve under four encoders (FINAL audio-filtered corpus).

Same conventions as fig2 (free-asymptote logistic in log N, seed-noise band, chance floor),
one curve per encoder on the SAME training draws: DINOv3-B off-the-shelf (the fig2 curve),
DINOv3-L off-the-shelf, and DINOv3-L retrained from scratch on BabyView — the in-domain
encoder, and the negative result. Full-corpus points are the ladder base runs.

"""
import sys, re, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scaling_fit import logistic, CHANCE

ENCODERS = [  # label, scaling-family regex, full-corpus family, color
    ("DINOv3-L off-the-shelf", r"F_dinov3l_rand_(\d+)", "F_dinov3l_base", T.OTHER),
    ("DINOv3-B off-the-shelf", r"F_dinov3b_rand_(\d+)", "F_dinov3b_base", T.FREE),
    ("ViT-B BabyView-trained", r"F_vitb_bv_rand_(\d+)", "F_vitb_bv_base", T.INDOM),
    ("ViT-S BabyView-trained", r"F_vits_bv_rand_(\d+)", "F_vits_bv_base", "#d99aa7"),
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
    # anchor labels at the full-corpus point, staggered per encoder (asymptotes converge)
    DY = {"DINOv3-L off-the-shelf": 3.6, "DINOv3-B off-the-shelf": -3.8,
          "ViT-B BabyView-trained": 4.2, "ViT-S BabyView-trained": -3.6}[lab]
    ax.text(x[-1] * 1.25, y[-1] + DY, lab, fontsize=5.6, color=col, ha="left", va="center")
    print(f"  NOTE fig2 {lab.replace(chr(10), ' ')}: asymptote "
          f"{popt[2]:.1f} ± {np.sqrt(pcov[2, 2]):.1f}, n per point {[fm['n'] for fm in fam]}")

ax.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(3.5e6, 22.2, "chance", fontsize=5.6, color=T.SUB, ha="right")
ax.set_xscale("log"); ax.set_xlim(3e3, 3e7); ax.set_ylim(18, 95)
ax.set_xlabel("training pairs"); ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
T.save(fig, "fig2_encoders")
