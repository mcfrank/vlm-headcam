"""Display item 4 — both evals in developmental time, all four encoders, beside children.

A: Konkle. The fitted scaling curve for each encoder (fig2's fits) re-expressed as years of
   waking input, faded beyond the observed range, with Wordbank CDI trajectories (predicted
   4AFC over the same 60 words: know-it-or-guess; WG comprehension, WS production as a lower
   bound) plotted at child age.
B: LEVANTE. The fair score per encoder (all 159 items, chance credited out-of-vocab) under
   the same logistic — a guess, since the fair ceiling depends on vocabulary growth beyond
   this corpus — against children's MEASURED accuracy by age on the same items (levante-bench
   trials; macro over items, bootstrap 10-90% CI). Bands omitted with four curves; seed sd is
   in fig2 / figS5.
"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scaling_fit import fit, logistic, CHANCE

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
WB = pd.read_csv(R / "wordbank_anchors.csv")
KIDS = pd.read_csv(R / "levante_child_by_age.csv")
lev = pd.read_csv(R / "lev_scaling_final.csv")
lev = lev[lev.N <= 1_686_105]
UTT_PER_HR, HR_PER_YEAR = 820, 4000
CDI_INK = "#8a6d1f"
to_yr = lambda n: n / UTT_PER_HR / HR_PER_YEAR
tgrid = np.logspace(np.log10(3e-3), np.log10(14), 260)
ngrid = tgrid * UTT_PER_HR * HR_PER_YEAR
OBS_END = to_yr(1_686_105)

ENC = [("dinov3l", "L-OTS", "DINOv3-L off-the-shelf", T.OTHER),
       ("dinov3b", "B-OTS", "DINOv3-B off-the-shelf", T.FREE),
       ("vitb_bv", "B-BV", "ViT-B BabyView-trained", T.INDOM),
       ("vits_bv", "S-BV", "ViT-S BabyView-trained", "#d99aa7")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6))


def curve_with_fade(a, popt, col):
    obs = tgrid <= OBS_END; ext = tgrid >= OBS_END
    a.plot(tgrid[obs], logistic(ngrid[obs], *popt), color=col, lw=1.1, zorder=2)
    a.plot(tgrid[ext], logistic(ngrid[ext], *popt), color=col, lw=1.1, alpha=0.4, zorder=2)


# ---- A: Konkle -------------------------------------------------------------------
KDY = {"L-OTS": 2.6, "B-OTS": -2.6, "B-BV": 2.2, "S-BV": -2.4}
for enc, key, lab, col in ENC:
    F = fit(enc=enc)
    curve_with_fade(ax, F["popt"], col)
    ax.text(13.5, logistic(ngrid[-1], *F["popt"]) + KDY[key], key, fontsize=5.4, color=col,
            ha="right", va="center")
for form, meas, mk in [("WG", "understands", "o"), ("WS", "produces", "^")]:
    d = WB[WB.form == form].sort_values("age")
    ax.plot(d.age / 12, d.pred_4afc, marker=mk, ms=2.6, lw=0.9, color=T.CHILD,
            markeredgecolor=CDI_INK, markeredgewidth=0.4, zorder=5)
    end = d.iloc[-1]
    if form == "WS":
        ax.text(end.age / 12 * 1.06, end.pred_4afc - 4.5, meas, fontsize=5.2, color=CDI_INK,
                va="top", ha="left")
    else:
        ax.text(end.age / 12 * 0.90, end.pred_4afc + 2.2, meas, fontsize=5.2, color=CDI_INK,
                va="bottom", ha="right")
d0 = WB.sort_values("age").iloc[0]
ax.text(d0.age / 12 * 0.92, d0.pred_4afc + 1.5, "children\n(Wordbank CDI)", fontsize=5.2,
        color=CDI_INK, ha="right", va="center", linespacing=1.4)

# ---- B: LEVANTE ------------------------------------------------------------------
lev["fair"] = np.where(lev.playable, lev.correct, 0.25)
LDY = {"L-OTS": -2.6, "B-OTS": -2.2, "B-BV": 2.4, "S-BV": -2.2}
for enc, key, lab, col in ENC:
    per = lev[lev.encoder == key].groupby(["N", "seed"]).fair.mean().mul(100).reset_index()
    S = per.groupby("N").agg(m=("fair", "mean"), sd=("fair", "std")).reset_index()
    x, y, e = S.N.values.astype(float), S.m.values, np.maximum(S.sd.values, 0.5)
    popt, _ = curve_fit(logistic, x, y, p0=(6, 0.8, 45), sigma=e,
                        bounds=([3, 0.1, 25.5], [10, 3, 100]), maxfev=60000)
    curve_with_fade(bx, popt, col)
    bx.text(13.5, logistic(ngrid[-1], *popt) + LDY[key], key, fontsize=5.4, color=col,
            ha="right", va="center")
    print(f"  NOTE fig4 LEVANTE {key}: fair asymptote {popt[2]:.1f}")
bx.errorbar(KIDS.age_yr, 100 * KIDS.acc_macro,
            yerr=[100 * (KIDS.acc_macro - KIDS.lo), 100 * (KIDS.hi - KIDS.acc_macro)],
            fmt="-o", color=T.CHILD, ms=2.8, lw=0.9, elinewidth=0.6, capsize=1.4,
            markeredgecolor=CDI_INK, markeredgewidth=0.4, zorder=5)
bx.text(4.4, 74, "children, same items\n(LEVANTE trials)", fontsize=5.2, color=CDI_INK,
        ha="right", va="center", linespacing=1.4)

for a, ylab in [(ax, "Konkle 4AFC (%)"), (bx, "LEVANTE vocabulary 4AFC (%)")]:
    for yr in (1, 2, 3):
        a.axvline(yr, color=T.GRID, lw=0.7, zorder=0)
        a.text(yr, 19.3, f"{yr} yr" if yr == 1 else f"{yr}", fontsize=5.4, color=T.SUB,
               ha="center", va="bottom")
    a.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
    a.set_xscale("log"); a.set_xlim(3e-3, 14); a.set_ylim(18, 95)
    a.set_xlabel("developmental time (years of waking input)")
    a.set_ylabel(ylab)
    T.clean(a)
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)
T.save(fig, "fig4_development")
