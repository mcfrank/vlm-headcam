"""Display item 9 — the scaling fit carried into developmental time.

The fitted B26 curve alone (no points), re-expressed as years of waking input and extended to
child timescales, faded beyond the observed range. Anchor lines at a child's first three years;
Wordbank CDI trajectories (predicted 4AFC over the same 60 words: know the word -> correct,
else guess) plotted at child age — comprehension (WG) and production (WS, a lower bound on
comprehension).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt
from scaling_fit import fit, logistic, CHANCE

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
WB = pd.read_csv(R / "wordbank_anchors.csv")
UTT_PER_HR, HR_PER_YEAR = 820, 4000          # caregiver utterances/hr; waking hrs/yr

F = fit()
to_yr = lambda n: n / UTT_PER_HR / HR_PER_YEAR
tgrid = np.logspace(np.log10(3e-3), np.log10(12), 260)
ngrid = tgrid * UTT_PER_HR * HR_PER_YEAR
tlo, thi = F["band"](ngrid)

fig, bx = plt.subplots(figsize=(T.W15, 2.6))
obs = tgrid <= to_yr(F["x"].max())
ext = tgrid >= to_yr(F["x"].max())
bx.fill_between(tgrid[obs], tlo[obs], thi[obs], color=T.FREE, alpha=0.14, lw=0, zorder=1)
bx.fill_between(tgrid[ext], tlo[ext], thi[ext], color=T.FREE, alpha=0.06, lw=0, zorder=1)
bx.plot(tgrid[obs], logistic(ngrid[obs], *F["popt"]), color=T.FREE, lw=1.1, zorder=2)
bx.plot(tgrid[ext], logistic(ngrid[ext], *F["popt"]), color=T.FREE, lw=1.1, alpha=0.45, zorder=2)
bx.text(0.02, 62, "BabyView model\n(unfiltered)", fontsize=5.8, color=T.FREE, ha="center")

for yr in (1, 2, 3):
    bx.axvline(yr, color=T.GRID, lw=0.7, zorder=0)
    bx.text(yr, 19.3, f"{yr} yr" if yr == 1 else f"{yr}", fontsize=5.4, color=T.SUB,
            ha="center", va="bottom")

CDI_INK = "#8a6d1f"
for form, meas, mk in [("WG", "understands", "o"), ("WS", "produces", "^")]:
    d = WB[WB.form == form].sort_values("age")
    bx.plot(d.age / 12, d.pred_4afc, marker=mk, ms=2.6, lw=0.9, color=T.CHILD,
            markeredgecolor=CDI_INK, markeredgewidth=0.4, zorder=5)
    end = d.iloc[-1]
    if form == "WS":
        bx.text(end.age / 12 * 1.06, end.pred_4afc - 4.5, meas, fontsize=5.2, color=CDI_INK,
                va="top", ha="left")
    else:
        bx.text(end.age / 12 * 0.97, end.pred_4afc + 3.0, meas, fontsize=5.2, color=CDI_INK,
                va="bottom", ha="center")
d0 = WB.sort_values("age").iloc[0]
bx.text(d0.age / 12 * 0.92, d0.pred_4afc + 1.5, "children\n(Wordbank CDI)", fontsize=5.2,
        color=CDI_INK, ha="right", va="center", linespacing=1.4)
bx.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
bx.set_xscale("log"); bx.set_xlim(3e-3, 12); bx.set_ylim(18, 95)
bx.set_xlabel("developmental time (years of waking input)")
bx.set_ylabel("Konkle 4AFC (%)")
T.clean(bx)
T.save(fig, "fig9_development")
