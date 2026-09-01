"""Display item 2 — what moves the scaling curve.

A: the encoder moves it up — four encoders on the same training draws (FINAL corpus),
   free-asymptote logistic per encoder. Off-the-shelf DINOv3 ceilings converge (~85);
   BabyView-trained encoders saturate far lower.
B: alignment moves it left — training only on the referential pairs (independent aligned
   subsamples, oracle-filtered) against the unfiltered B-OTS curve from panel A. Same
   ceiling, ~29x less data at matched accuracy; the aligned supply ends where the corpus's
   referential moments run out.
"""
import sys, re, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scaling_fit import fit, logistic, CHANCE

rng = np.random.default_rng(0)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6))

# ================================ A: encoders ====================================
ENCODERS = [  # label, scaling-family regex, full-corpus family, color
    ("DINOv3-L off-the-shelf", r"F_dinov3l_rand_(\d+)", "F_dinov3l_base", T.OTHER),
    ("DINOv3-B off-the-shelf", r"F_dinov3b_rand_(\d+)", "F_dinov3b_base", T.FREE),
    ("ViT-B BabyView-trained", r"F_vitb_bv_rand_(\d+)", "F_vitb_bv_base", T.INDOM),
    ("ViT-S BabyView-trained", r"F_vits_bv_rand_(\d+)", "F_vits_bv_base", T.INDOM2),
]
rng = np.random.default_rng(0)
grid = np.logspace(3.3, 7.8, 260)

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
    KEY = {"DINOv3-L off-the-shelf": "L-OTS", "DINOv3-B off-the-shelf": "B-OTS",
           "ViT-B BabyView-trained": "B-BV", "ViT-S BabyView-trained": "S-BV"}[lab]
    ax.text(x[-1] * 1.3, y[-1] + DY, KEY, fontsize=5.6, color=col, ha="left", va="center")
    print(f"  NOTE fig2 {lab.replace(chr(10), ' ')}: asymptote "
          f"{popt[2]:.1f} ± {np.sqrt(pcov[2, 2]):.1f}, n per point {[fm['n'] for fm in fam]}")

ax.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(3.5e6, 22.2, "chance", fontsize=5.6, color=T.SUB, ha="right")
ax.set_xscale("log"); ax.set_xlim(3e3, 3e7); ax.set_ylim(18, 92)
ax.set_xlabel("training pairs"); ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)

# published single-child SAYCam models at their utterance counts (same Konkle-60 items)
import pandas as pd
LIT = pd.read_csv(__import__("pathlib").Path(__file__).resolve().parent.parent / "results" / "literature.csv")
lit = LIT.groupby(["source", "split"], sort=False).agg(n=("n_utterances", "first"),
                                                        acc=("konkle_acc", "mean")).reset_index()
lit = lit.sort_values("acc").reset_index(drop=True)
dy = np.zeros(len(lit))
for i in range(1, len(lit)):
    if lit.acc[i] - lit.acc[i - 1] < 1.5 and abs(np.log10(lit.n[i] / lit.n[i - 1])) < 0.1:
        dy[i - 1] -= 1.6; dy[i] += 1.2
# hand-placed so no label touches a curve, a marker or another label
# only the child ids sit at the points; the two sources are keyed in a small legend
POS = {"S": (1.20, 0.0, "left", "center"), "A": (0.82, 0.7, "right", "center"),
       "Y": (0.82, -1.5, "right", "center")}
for r in lit.itertuples():
    filled = "2026" in r.source
    ax.scatter([r.n], [r.acc], marker="D", s=15, zorder=6,
               facecolors=T.INK if filled else "white", edgecolors=T.INK, lw=0.8)
    if filled:
        fx, dy_, ha, va = POS[r.split.split("-")[0]]
        ax.text(r.n * fx, r.acc + dy_, r.split.split("-")[0], fontsize=5.4, color=T.INK,
                ha=ha, va=va, zorder=7)
ax.text(4.0e3, 74, "single-child models (SAYCam)", fontsize=5.2, color=T.INK, va="center")
for yy, fc, lab in [(69.0, "white", "CVCL (Vong et al. 2024)"),
                    (64.5, T.INK, "Vong & Lake 2026")]:
    ax.scatter([4.9e3], [yy], marker="D", s=15, facecolors=fc, edgecolors=T.INK, lw=0.8,
               zorder=6, clip_on=False)
    ax.text(6.4e3, yy, lab, fontsize=5.2, color=T.INK, va="center")

# ================================ B: alignment ===================================
F = fit(rng)
popt = F["popt"]; A = popt[2]
afams = sorted(((f, D.family(f)) for f in D.runs.family.unique()
                if re.fullmatch(r"F_dinov3b_align_\d+", str(f))), key=lambda t: t[1]["n_pairs"])
ax_ = np.array([f["n_pairs"] for _, f in afams])
ay = np.array([f["mean"] for _, f in afams]); ae = np.array([f["sd"] for _, f in afams])
apopt, _ = curve_fit(lambda N, m, s_: logistic(N, m, s_, A), ax_, ay, p0=(4.3, 0.5),
                     sigma=ae, maxfev=40000)
agrid = np.logspace(3.6, np.log10(ax_.max()), 140)
adraws = []
for _ in range(400):
    try:
        pa, _ = curve_fit(lambda N, m, s_: logistic(N, m, s_, A), ax_,
                          ay + rng.normal(0, np.maximum(ae, 0.5)), p0=apopt, maxfev=40000)
        adraws.append(logistic(agrid, *pa, A))
    except Exception:
        pass
alo, ahi = np.percentile(np.array(adraws), [10, 90], axis=0)


# the unfiltered curve as a known reference (it is fig2's B-OTS curve)
grid = np.logspace(3.3, 6.6, 220)
bx.plot(grid, logistic(grid, *popt), color=T.NEUTRAL, lw=1.0, zorder=2)
bx.errorbar(F["x"], F["y"], yerr=F["e"], fmt="o", color=T.NEUTRAL, ms=2.6, lw=0,
            elinewidth=0.6, capsize=1.3, zorder=3)
bx.text(3.7e6, 62, "unfiltered\n(B-OTS, as in A)", fontsize=5.6, color=T.NEUTRAL, ha="right",
        va="top", linespacing=1.35)

# the aligned arm
bx.fill_between(agrid, alo, ahi, color=T.ORACLE, alpha=0.14, lw=0, zorder=1)
bx.plot(agrid, logistic(agrid, *apopt, A), color=T.ORACLE, lw=1.2, zorder=4)
bx.errorbar(ax_, ay, yerr=ae, fmt="o", color=T.ORACLE, ms=3.2, lw=0, elinewidth=0.7,
            capsize=1.6, zorder=5)
aend = logistic(agrid[-1], *apopt, A)
bx.plot([agrid[-1]] * 2, [aend - 2.6, aend + 2.6], color=T.ORACLE, lw=0.9, zorder=5)
bx.text(agrid[-1] * 1.35, aend - 1.2, "all referential pairs\nin the corpus", fontsize=5.2,
        color=T.ORACLE, ha="left", va="top", linespacing=1.35)
bx.text(3.4e3, 76, "aligned only\n(oracle filter)", fontsize=6.0, color=T.ORACLE, ha="left",
        va="top", linespacing=1.35)

# data equivalence: at each aligned point's accuracy, how much unfiltered data matches it?
inv = lambda y, m, s_: 10 ** (m - s_ * np.log((A - CHANCE) / (y - CHANCE) - 1))
for y_lev in (ay[0], ay[1]):
    xa = inv(y_lev, *apopt)
    xu = inv(y_lev, popt[0], popt[1])
    bx.annotate("", xy=(xu, y_lev), xytext=(xa, y_lev),
                arrowprops=dict(arrowstyle="-|>", color=T.INK, lw=0.8, shrinkA=2, shrinkB=1))
    bx.text(np.sqrt(xa * xu), y_lev + 1.0, f"{xu / xa:.0f}×", fontsize=6.2, color=T.INK,
            ha="center", va="bottom", fontweight="bold")
print(f"  NOTE fig3: data-equivalence {inv(ay[0], *apopt):,.0f} vs "
      f"{inv(ay[0], popt[0], popt[1]):,.0f} pairs at {ay[0]:.1f}%; shift "
      f"{popt[0] - apopt[0]:.2f} decades at midpoint")

bx.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
bx.text(3.6e6, 22.3, "chance", fontsize=5.6, color=T.SUB, ha="right")
bx.set_xscale("log"); bx.set_xlim(3e3, 4e6); bx.set_ylim(18, 92)
bx.set_ylabel("")
bx.set_xlabel("training pairs"); bx.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)

bx.set_yticklabels([])
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.14)
T.save(fig, "fig2_scaling")
