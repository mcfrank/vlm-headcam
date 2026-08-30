"""Display item 9 — the scaling fits carried into developmental time, beside children.

A: the Konkle fit (shared with fig2 via scaling_fit), faded beyond the observed range, with
   Wordbank CDI trajectories (predicted 4AFC over the same 60 words: know-it-or-guess;
   comprehension from WG, production from WS as a lower bound) plotted at child age.
B: the same construction for LEVANTE: the B-OTS fair score (all 159 items, chance credited
   out-of-vocab) fit with the same free-asymptote logistic — a guess, since the fair ceiling
   depends on vocabulary growth beyond this corpus — against children's MEASURED accuracy by
   age on the same items (levante-bench trials; macro over items, bootstrap 10-90% band).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scaling_fit import fit, logistic, CHANCE

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
WB = pd.read_csv(R / "wordbank_anchors.csv")
KIDS = pd.read_csv(R / "levante_child_by_age.csv")
lev = pd.read_csv(R / "lev_scaling.csv")
UTT_PER_HR, HR_PER_YEAR = 820, 4000
CDI_INK = "#8a6d1f"
to_yr = lambda n: n / UTT_PER_HR / HR_PER_YEAR
tgrid = np.logspace(np.log10(3e-3), np.log10(14), 260)
ngrid = tgrid * UTT_PER_HR * HR_PER_YEAR
rng = np.random.default_rng(0)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6))


def curve_with_fade(a, popt, band, obs_max_n, col):
    lo, hi = band
    obs = tgrid <= to_yr(obs_max_n); ext = tgrid >= to_yr(obs_max_n)
    a.fill_between(tgrid[obs], lo[obs], hi[obs], color=col, alpha=0.14, lw=0, zorder=1)
    a.fill_between(tgrid[ext], lo[ext], hi[ext], color=col, alpha=0.06, lw=0, zorder=1)
    a.plot(tgrid[obs], logistic(ngrid[obs], *popt), color=col, lw=1.1, zorder=2)
    a.plot(tgrid[ext], logistic(ngrid[ext], *popt), color=col, lw=1.1, alpha=0.45, zorder=2)


# ---- A: Konkle vs Wordbank CDI ---------------------------------------------------
F = fit()
curve_with_fade(ax, F["popt"], F["band"](ngrid), F["x"].max(), T.FREE)
ax.text(0.02, 62, "BabyView model\n(unfiltered)", fontsize=5.8, color=T.FREE, ha="center")
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

# ---- B: LEVANTE vs measured children --------------------------------------------
lev["fair"] = np.where(lev.playable, lev.correct, 0.25)
per = (lev[lev.encoder == "B-OTS"].groupby(["N", "seed"]).fair.mean().mul(100)
          .reset_index())
S = per.groupby("N").agg(m=("fair", "mean"), sd=("fair", "std")).reset_index()
x, y, e = S.N.values.astype(float), S.m.values, np.maximum(S.sd.values, 0.5)
popt, pcov = curve_fit(logistic, x, y, p0=(6, 0.8, 60), sigma=e,
                       bounds=([3, 0.1, 26], [10, 3, 100]), maxfev=60000)
draws = []
for _ in range(400):
    try:
        p_, _ = curve_fit(logistic, x, y + rng.normal(0, e), p0=popt,
                          bounds=([3, 0.1, 26], [10, 3, 100]), maxfev=60000)
        draws.append(logistic(ngrid, *p_))
    except Exception:
        pass
band = np.percentile(np.array(draws), [10, 90], axis=0)
print(f"  NOTE fig4 LEVANTE fair-score asymptote {popt[2]:.1f} ± {np.sqrt(pcov[2,2]):.1f} "
      "— a guess; the fair ceiling depends on vocabulary growth beyond this corpus")
curve_with_fade(bx, popt, band, x.max(), T.FREE)
bx.text(0.02, 45, "BabyView model\n(all 159 items;\nchance if out-of-vocab)", fontsize=5.4,
        color=T.FREE, ha="center", linespacing=1.4)
bx.errorbar(KIDS.age_yr, 100 * KIDS.acc_macro,
            yerr=[100 * (KIDS.acc_macro - KIDS.lo), 100 * (KIDS.hi - KIDS.acc_macro)],
            fmt="-o", color=T.CHILD, ms=2.8, lw=0.9, elinewidth=0.6, capsize=1.4,
            markeredgecolor=CDI_INK, markeredgewidth=0.4, zorder=5)
bx.text(KIDS.age_yr.min() * 0.9, 100 * KIDS.acc_macro.iloc[0] - 3.5,
        "children, same items\n(LEVANTE trials)", fontsize=5.2, color=CDI_INK, ha="right",
        va="top", linespacing=1.4)

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
