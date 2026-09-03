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
from scaling_fit import fit, logistic, mc_band, CHANCE

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
WB = pd.read_csv(R / "wordbank_anchors.csv")
KIDS = pd.read_csv(R / "levante_child_by_age.csv")
lev = pd.read_csv(R / "lev_scaling_final.csv")
lev = lev[lev.N <= 1_686_105]
# Measured in this corpus (results/utterance_rate.csv): 1,835,504 utterances over 2,633 h
# of transcribed span, 34 children. The rate is the soft part of this axis, so the curves
# carry a horizontal band for the 95% bootstrap-over-children interval on it.
UTT_PER_HR, UTT_LO, UTT_HI, HR_PER_YEAR = 697, 634, 752, 4000
CDI_INK = "#8a6d1f"
to_yr = lambda n: n / UTT_PER_HR / HR_PER_YEAR
tgrid = np.logspace(np.log10(3e-3), np.log10(14), 260)
ngrid = tgrid * UTT_PER_HR * HR_PER_YEAR
OBS_END = to_yr(1_686_105)
rng = np.random.default_rng(0)

ENC = [("dinov3l", "L-OTS", "DINOv3-L off-the-shelf", T.OTHER),
       ("dinov3b", "B-OTS", "DINOv3-B off-the-shelf", T.FREE),
       ("vitb_bv", "B-BV", "ViT-B BabyView-trained", T.INDOM),
       ("vits_bv", "S-BV", "ViT-S BabyView-trained", T.INDOM2)]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6))


def curve_with_fade(a, popt, col, band=None):
    """Solid + banded where we have data; heavily faded past it, where only the seed
    component of the uncertainty is quantified."""
    obs = tgrid <= OBS_END; ext = tgrid >= OBS_END
    if band is not None:
        lo, hi = band
        a.fill_between(tgrid[obs], lo[obs], hi[obs], color=col, alpha=0.16, lw=0, zorder=1)
        a.fill_between(tgrid[ext], lo[ext], hi[ext], color=col, alpha=0.05, lw=0, zorder=1)
    a.plot(tgrid[obs], logistic(ngrid[obs], *popt), color=col, lw=1.1, zorder=2)
    a.plot(tgrid[ext], logistic(ngrid[ext], *popt), color=col, lw=1.1, alpha=0.22, zorder=2)


# ---- A: Konkle -------------------------------------------------------------------
KDY = {"L-OTS": 2.6, "B-OTS": -2.6, "B-BV": 2.2, "S-BV": -2.4}
for enc, key, lab, col in ENC:
    F = fit(enc=enc)
    curve_with_fade(ax, F["popt"], col, band=F["band"](ngrid))
    ax.text(13.5, logistic(ngrid[-1], *F["popt"]) + KDY[key], key, fontsize=5.4, color=col,
            ha="right", va="center")
for form, meas, mk, ls in [("WG", "understands", "o", "-"), ("WS", "produces", "^", (0, (2.5, 1.5)))]:
    d = WB[WB.form == form].sort_values("age")
    ax.plot(d.age / 12, d.pred_4afc, marker=mk, ms=2.6, lw=0.9, ls=ls, color=T.CHILD,
            markerfacecolor="white", markeredgecolor=CDI_INK, markeredgewidth=0.6, zorder=5)
    end = d.iloc[-1]
    if form == "WS":
        ax.text(end.age / 12 * 1.06, end.pred_4afc - 4.5, meas, fontsize=5.2, color=CDI_INK,
                va="top", ha="left")
    else:
        ax.text(end.age / 12 * 0.90, end.pred_4afc + 2.2, meas, fontsize=5.2, color=CDI_INK,
                va="bottom", ha="right")
d0 = WB.sort_values("age").iloc[0]
nwg = int(WB[WB.form == "WG"].n_children.sum()); nws = int(WB[WB.form == "WS"].n_children.sum())
ax.text(d0.age / 12 * 0.92, d0.pred_4afc + 1.5,
        f"children\n(Wordbank CDI;\n{nwg:,} / {nws:,} administrations)", fontsize=5.0,
        color=CDI_INK, ha="right", va="center", linespacing=1.4)

# ---- B: LEVANTE ------------------------------------------------------------------
lev["fair"] = np.where(lev.playable, lev.correct, 0.25)
LDY = {"L-OTS": 2.6, "B-OTS": -2.6, "B-BV": 2.6, "S-BV": -2.6}
LX = 3.0   # children now occupy the right of this panel
for enc, key, lab, col in ENC:
    per = lev[lev.encoder == key].groupby(["N", "seed"]).fair.mean().mul(100).reset_index()
    S = per.groupby("N").agg(m=("fair", "mean"), sd=("fair", "std"),
                             k=("fair", "size")).reset_index()
    x, y = S.N.values.astype(float), S.m.values
    e = np.maximum(S.sd.values, 0.5)
    sem = np.maximum(S.sd.values / np.sqrt(S.k.values), 0.15)
    popt, _ = curve_fit(logistic, x, y, p0=(6, 0.8, 45), sigma=e,
                        bounds=([3, 0.1, 25.5], [10, 3, 100]), maxfev=60000)
    curve_with_fade(bx, popt, col,
                    band=mc_band(x, y, sem, popt, ngrid, rng,
                                 ([3, 0.1, 25.5], [10, 3, 100]), sigma=e))
    bx.text(LX, logistic(LX * UTT_PER_HR * HR_PER_YEAR, *popt) + LDY[key], key,
            fontsize=5.4, color=col, ha="right", va="center")
    print(f"  NOTE fig3 LEVANTE {key}: fair asymptote {popt[2]:.1f}")
bx.errorbar(KIDS.age_yr, 100 * KIDS.acc_macro,
            yerr=[100 * (KIDS.acc_macro - KIDS.lo), 100 * (KIDS.hi - KIDS.acc_macro)],
            fmt="-o", color=T.CHILD, ms=2.8, lw=0.9, elinewidth=0.6, capsize=1.4,
            markeredgecolor=CDI_INK, markeredgewidth=0.4, zorder=5)
nk = int(KIDS.n_children.sum())
bx.text(0.55, 80, f"children, same items\n(LEVANTE, IRT full-scale;\nEnglish, n={nk})",
        fontsize=5.0, color=CDI_INK, ha="center", va="center", linespacing=1.4)

def rate_bar(a, x0=0.028, yb=21.8):
    """One bar for the horizontal uncertainty: the utterances-per-hour conversion is a
    multiplicative factor, so a single log-width bar applies to every curve at every x."""
    lo, hi = x0 * UTT_PER_HR / UTT_HI, x0 * UTT_PER_HR / UTT_LO
    a.plot([lo, hi], [yb, yb], color=T.SUB, lw=0.8, solid_capstyle="butt", zorder=4)
    for xx in (lo, hi):
        a.plot([xx, xx], [yb - 0.9, yb + 0.9], color=T.SUB, lw=0.8, zorder=4)
    a.text(hi * 1.25, yb, f"input rate ({UTT_LO}\u2013{UTT_HI} utt/hr)", fontsize=5.0,
           color=T.SUB, va="center")


for a, ylab in [(ax, "Konkle 4AFC (%)"), (bx, "LEVANTE vocabulary 4AFC (%)")]:
    for yr in (1, 2, 3):
        a.axvline(yr, color=T.GRID, lw=0.7, zorder=0)
        a.text(yr, 19.3, f"{yr} yr" if yr == 1 else f"{yr}", fontsize=5.4, color=T.SUB,
               ha="center", va="bottom")
    a.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
    a.set_xscale("log"); a.set_xlim(3e-3, 14); a.set_ylim(18, 95)
    a.set_xlabel("developmental time (years of waking input)")
    a.set_ylabel(ylab)
    rate_bar(a)
    T.clean(a)
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)
T.save(fig, "fig3_development")
