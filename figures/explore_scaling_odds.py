"""EXPLORATORY (not in the Makefile): fig2A / fig4 replotted as scaling laws.

Vlad's suggestion: a power law needs two ratio-scale quantities, so plot log odds against
log data and read the exponent off the slope. For a 4AFC with chance 25 the ratio-scale
quantity is the odds of KNOWING under the know-it-or-guess model, k = (p - 25) / 75,
odds = k / (1 - k): zero at chance, unbounded above. Same log base (10) on both axes so
the slope is the exponent. Points whose mean is within 2 pts of chance are drawn open at
the bottom and left out of the fits (their log odds is undefined within noise).

A: Konkle-60 vs training pairs, six encoders (the fig2A data), OLS line per encoder over the
   fitted points with its slope.
B: Konkle-40 in developmental time with the Wordbank CDI children (the fig4A data).
C: LEVANTE fair score in developmental time with the LEVANTE children (the fig4B data).
"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
UTT_PER_HR, HR_PER_YEAR = 697, 4000
to_yr = lambda n: n / UTT_PER_HR / HR_PER_YEAR
FLOOR = 2.0                                  # pts above chance below which log odds is noise


def lodds(p):
    k = np.clip((np.asarray(p, float) - 25) / 75, 1e-6, 1 - 1e-6)
    return np.log10(k / (1 - k))


def series(a, x, y, e, col, mk, key, xlab_at, fit_lo_x=None):
    """points +- sd in log-odds space, OLS over the points clear of chance, slope label"""
    x, y, e = (np.asarray(v, float) for v in (x, y, e))
    ok = y - 25 > FLOOR
    lo, hi = lodds(np.maximum(y - e, 25 + 0.3)), lodds(y + e)
    a.errorbar(x[ok], lodds(y[ok]), yerr=[lodds(y[ok]) - lo[ok], hi[ok] - lodds(y[ok])],
               fmt=mk, color=col, ms=2.8, lw=0, elinewidth=0.6, capsize=1.2, zorder=4)
    a.scatter(x[~ok], np.full((~ok).sum(), YMIN + 0.08), marker=mk, s=12, facecolors="white",
              edgecolors=col, lw=0.8, zorder=4, clip_on=False)
    m = ok if fit_lo_x is None else ok & (x >= fit_lo_x)
    slope, icpt = np.polyfit(np.log10(x[m]), lodds(y[m]), 1)
    xx = np.logspace(np.log10(x[m].min()) - 0.2, np.log10(x[m].max()) + 0.2, 20)
    a.plot(xx, slope * np.log10(xx) + icpt, color=col, lw=0.9, zorder=2)
    resid = lodds(y[m]) - (slope * np.log10(x[m]) + icpt)
    ENDS.append((xlab_at, slope * np.log10(xlab_at) + icpt, f"{key}  α={slope:.2f}", col))
    return slope, np.sqrt(np.mean(resid ** 2))


def flush(a):
    T.end_labels(a, [t[0] for t in ENDS], [t[1] for t in ENDS], [t[2] for t in ENDS],
                 [t[3] for t in ENDS], gap=0.17, xl=[t[0] * 1.25 for t in ENDS], fontsize=4.8)
    ENDS.clear()


ENDS = []


fig, axes = plt.subplots(1, 3, figsize=(T.W2, 2.5), gridspec_kw=dict(wspace=0.36))
YMIN, YMAX = -2.2, 1.2

# ---- A: Konkle-60 vs pairs ---------------------------------------------------------
ax = axes[0]
for E in T.ENCODERS:
    enc = E["tag"]
    fams = sorted((int(m.group(1)), f) for f in D.runs.family.unique()
                  if (m := re.fullmatch(rf"F_{enc}_rand_(\d+)", str(f))))
    fams.append((None, f"F_{enc}_base"))
    fam = [D.family(f) for _, f in fams]
    x = [fm["n_pairs"] for fm in fam]; y = [fm["mean"] for fm in fam]; e = [fm["sd"] for fm in fam]
    s, rms = series(ax, x, y, e, E["color"], E["marker"], E["label"], 2.4e6)
    print(f"  A {E['label']}: exponent {s:.2f}, rms resid {rms:.2f} log10 units")
flush(ax)
ax.set_xscale("log"); ax.set_xlim(2e3, 6e7)
ax.set_xlabel("training pairs")
ax.set_ylabel("log$_{10}$ odds of knowing (Konkle-60)")

# ---- B: Konkle-40 in developmental time + Wordbank ----------------------------------
bx = axes[1]
K40 = pd.read_csv(R / "konkle_wg40_per_seed.csv")
for E in T.ENCODERS:
    S = K40[K40.encoder == E["tag"]].groupby("N").acc.agg(["mean", "std"]).reset_index()
    s, _ = series(bx, to_yr(S.N), S["mean"], S["std"], E["color"], E["marker"], E["label"], 1.3)
    print(f"  B {E['label']}: exponent {s:.2f}")
flush(bx)
WB = pd.read_csv(R / "wordbank_anchors_40.csv")
CDI = "#8a6d1f"
for form, mk, ls in [("WG", "o", "-"), ("WS", "^", (0, (2.5, 1.5)))]:
    d = WB[WB.form == form].sort_values("age")
    xs, ys = d.age / 12, lodds(d.pred_4afc)
    bx.plot(xs, ys, marker=mk, ms=2.6, lw=0.9, ls=ls, color=T.CHILD, markerfacecolor="white",
            markeredgecolor=CDI, markeredgewidth=0.6, zorder=5)
    sl, ic = np.polyfit(np.log10(xs), ys, 1)
    print(f"  B children {form}: exponent {sl:.2f}")
    bx.text(xs.iloc[-1] * 1.25, -1.05 - (0.18 if form == "WG" else 0.0),
            f"CDI {'comp.' if form == 'WG' else 'prod.'}  α={sl:.1f}", fontsize=4.8, color=CDI,
            va="center", ha="left")
bx.set_xscale("log"); bx.set_xlim(3e-3, 200)
bx.set_xlabel("developmental time (years)")
bx.set_ylabel("log$_{10}$ odds of knowing (Konkle-40)")

# ---- C: LEVANTE in developmental time + children ------------------------------------
cx = axes[2]
lev = pd.read_csv(R / "lev_scaling_final.csv"); lev = lev[lev.N <= 1_686_105]
lev["fair"] = np.where(lev.playable, lev.correct, 0.25)
for E in T.ENCODERS:
    per = lev[lev.encoder == E["legacy"]].groupby(["N", "seed"]).fair.mean().mul(100).reset_index()
    S = per.groupby("N").fair.agg(["mean", "std"]).reset_index()
    s, _ = series(cx, to_yr(S.N), S["mean"], S["std"], E["color"], E["marker"], E["label"], 1.3)
    print(f"  C {E['label']}: exponent {s:.2f}")
flush(cx)
KIDS = pd.read_csv(R / "levante_child_by_age.csv")
xs, ys = KIDS.age_yr, lodds(100 * KIDS.acc_macro)
cx.errorbar(xs, ys, yerr=[ys - lodds(100 * KIDS.lo), lodds(100 * KIDS.hi) - ys], fmt="-o",
            color=T.CHILD, ms=2.8, lw=0.9, elinewidth=0.6, capsize=1.2, markeredgecolor=CDI,
            markeredgewidth=0.4, zorder=5)
sl, ic = np.polyfit(np.log10(xs), ys, 1)
print(f"  C children: exponent {sl:.2f}")
cx.text(xs.iloc[-1] * 1.3, ys[-1], f"children  α={sl:.1f}", fontsize=4.8, color=CDI, va="center")
cx.set_xscale("log"); cx.set_xlim(3e-3, 200)
cx.set_xlabel("developmental time (years)")
cx.set_ylabel("log$_{10}$ odds of knowing (LEVANTE)")

for a in axes:
    a.set_ylim(YMIN, YMAX)
    a.axhline(0, color=T.GRID, lw=0.7, zorder=0)
    a.text(0.02, 0.98, "open = within 2 pts of chance", transform=a.transAxes, fontsize=4.6,
           color=T.SUB, va="top")
    T.clean(a)
for a in [axes[2]]:
    ax2 = a.secondary_yaxis("right", functions=(lambda l: 25 + 75 * (10 ** l / (1 + 10 ** l)),
                                                 lambda p: lodds(p)))
    ax2.set_yticks([27, 30, 40, 50, 60, 70, 80, 90]); ax2.tick_params(labelsize=5, colors=T.SUB)
    for sp in ax2.spines.values():
        sp.set_visible(False)
axes[2].text(1.13, 0.5, "4AFC %", transform=axes[2].transAxes, fontsize=5, color=T.SUB,
             rotation=90, va="center", ha="left")
for a, l in zip(axes, "ABC"):
    T.panel(a, l, dx=-0.30)
T.save(fig, "explore_scaling_odds")
