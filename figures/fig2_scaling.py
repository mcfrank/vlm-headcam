"""Display item 2 — how does word learning scale with the input a child actually receives?

Deliberately shows ONLY the unfiltered stream: the question here is what *all* input buys, so the
oracle-aligned arm (which is not something a learner could select) belongs with the ladder, not on
this axis.

A: 4AFC vs training pairs, with a seed band and a saturating fit; CVCL (Vong et al. 2024) as a
   published reference point.
B: the same fit re-expressed in developmental time — hours of waking input — and extrapolated
   toward child timescales, with children's LEVANTE-bench vocabulary accuracy as calibration.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

UTT_PER_HR, HR_PER_YEAR = 820, 4000          # caregiver utterances/hr; waking hrs/yr
CEIL = D.claim("ladder_vision")["value"]      # clean-label ceiling for the saturating fit
rng = np.random.default_rng(0)

POINTS = [("scale_rand_10k", 1e4), ("scale_rand_30k", 3e4), ("scale_rand_100k", 1e5),
          ("scale_rand_300k", 3e5), ("scale_rand_911k", 9.11e5), ("scale_full", 1.145e6)]
x, y, e, prov = [], [], [], []
for cid, n in POINTS:
    c = D.claim(cid)
    x.append(n); y.append(c["value"]); e.append(c["sd"] or 1.5); prov.append(c["provisional"])
x, y, e, prov = map(np.array, (x, y, e, prov))


def sat(N, k, b):
    return CEIL - k * np.power(N, -b)


popt, _ = curve_fit(sat, x, y, p0=(300, 0.25), maxfev=20000)
# Monte-Carlo the seed noise through the fit for a band
grid = np.logspace(3.6, 7.6, 240)
draws = []
for _ in range(500):
    try:
        p, _ = curve_fit(sat, x, y + rng.normal(0, np.maximum(e, 0.5)), p0=popt, maxfev=20000)
        draws.append(np.clip(sat(grid, *p), 25, CEIL))
    except Exception:
        pass
draws = np.array(draws)
lo, hi = np.percentile(draws, [10, 90], axis=0)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6))

# ---- A: pairs axis --------------------------------------------------------------
ax.fill_between(grid, lo, hi, color=T.FREE, alpha=0.16, lw=0, zorder=1)
ax.plot(grid, sat(grid, *popt), color=T.FREE, lw=1.0, zorder=2)
ax.errorbar(x, y, yerr=e, fmt="o", color=T.FREE, ms=3.4, lw=0, elinewidth=0.7,
            capsize=1.6, ecolor=T.FREE, zorder=4)
if prov.any():
    ax.scatter(x[prov], y[prov], s=46, facecolors="none", edgecolors=T.PROV, lw=0.9, zorder=5)
# published reference: CVCL (Vong et al. 2024) — single child, ~600k frames
ax.scatter([6e5], [34.7], marker="D", s=18, color=T.LIT, zorder=5)
ax.annotate("CVCL (Vong et al. 2024)\nout-of-distribution 34.7", (6e5, 34.7), (4.2e5, 29.5),
            fontsize=5.6, color=T.LIT, ha="right", va="center",
            arrowprops=dict(arrowstyle="-", color=T.LIT, lw=0.5, shrinkA=1, shrinkB=2))
ax.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(2.8e6, 25.9, "chance", fontsize=5.6, color=T.SUB, ha="right")
ax.axhline(CEIL, color=T.ORACLE, lw=0.7, ls=(0, (2, 2)))
ax.text(4e3, CEIL + 1, "clean-label ceiling", fontsize=5.6, color=T.ORACLE)
ax.text(2.9e6, 68, "BabyView,\nunfiltered", fontsize=6, color=T.FREE, ha="right", va="bottom")
ax.set_xscale("log"); ax.set_xlim(4e3, 3e6); ax.set_ylim(20, 90)
ax.set_xlabel("training pairs"); ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)

# ---- B: developmental time ------------------------------------------------------
to_yr = lambda n: n / UTT_PER_HR / HR_PER_YEAR
gyr = to_yr(grid)
bx.fill_between(gyr, lo, hi, color=T.FREE, alpha=0.16, lw=0, zorder=1)
bx.plot(gyr, sat(grid, *popt), color=T.FREE, lw=1.0, zorder=2)
bx.errorbar(to_yr(x), y, yerr=e, fmt="o", color=T.FREE, ms=3.4, lw=0, elinewidth=0.7,
            capsize=1.6, zorder=4)
# children's LEVANTE vocabulary accuracy, plotted at their ages — CALIBRATION, not the same task
bx.axhspan(72, 82, color=T.CHILD, alpha=0.28, lw=0, zorder=1)
bx.text(6.0, 77, "children 5–12 yr\n(LEVANTE vocabulary)", fontsize=5.6, color="#8a6d1f",
        ha="right", va="center")
for yr in (1, 3, 5):
    bx.axvline(yr, color=T.GRID, lw=0.6, zorder=0)
    bx.text(yr, 21.5, f"{yr} yr", fontsize=5.4, color=T.SUB, ha="center")
bx.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
bx.axhline(CEIL, color=T.ORACLE, lw=0.7, ls=(0, (2, 2)))
bx.set_xscale("log"); bx.set_xlim(3e-3, 8); bx.set_ylim(20, 90)
bx.set_xlabel("developmental time (years of waking input)")
bx.text(0.03, 86, "observed", fontsize=5.8, color=T.SUB, style="italic")
bx.text(1.6, 86, "extrapolated", fontsize=5.8, color=T.SUB, style="italic")
bx.axvline(to_yr(x.max()), color=T.SUB, lw=0.6, ls=(0, (1, 2)))
T.clean(bx)
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)

note = D.provisional_note([c for c, _ in POINTS])
cap = ("Fit is saturating (ceiling = clean-label topline); band = 10–90% over seed noise. "
       "Children's band is LEVANTE vocabulary 4AFC — a different item set from Konkle, shown as "
       "calibration of scale, not a like-for-like comparison.")
if note:
    cap += "  ○ " + note
fig.text(0.5, -0.13, cap, ha="center", fontsize=5.5, color=T.SUB, wrap=True)
T.save(fig, "fig2_scaling")
