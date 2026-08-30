"""Display item 3 — what closes the gap? Oracle information, referential selection, and
data diversity, each against raw experience. B-OTS throughout; the encoder breakdown joins
when the C8 ladder / diversity families land (detected below).

A: the ladder at three scales — matched subsamples trained with increasing oracle
   information; the wedge between free and clean-label is the referential headroom.
B: the aligned arm — keeping only the ~10% referential moments (the first rung) as its own
   scaling curve beside the unfiltered one: same fitted ceiling, reached ~10x sooner, and
   the aligned supply ends where the corpus runs out.
C: diversity at three budgets — whose data it is matters only once there is enough of it.
"""
import sys, re, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scaling_fit import fit, logistic, CHANCE

if any(re.match(r"C8_.*(filtnat|div)", str(f)) for f in D.runs.family.unique()):
    print("  NOTE fig3: C8 ladder/diversity families exist — add the encoder breakdown")
rng = np.random.default_rng(0)

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(T.W2, 2.5),
                                 gridspec_kw=dict(width_ratios=[1.3, 1, 1], wspace=0.34))

# ---- A: the ladder at three scales ----------------------------------------------
RUNGS = [("base", "unfiltered", None), ("filtnat", "+ align filter", "which moments"),
         ("t15", "+ word select", "which word"), ("t2", "+ vision bind", "which object")]
scales = sorted({m.group(1) for f in D.runs.family.unique()
                 if (m := re.fullmatch(r"B26_lad(\d*)_?(?:base|filtnat|t15|t2)", str(f)))},
                key=lambda s: int(s or 10**9))
fam = lambda sc, r: D.family(f"B26_lad{sc}_{r}" if sc else f"B26_lad_{r}")
X = np.array([fam(sc, "base")["n_pairs"] for sc in scales])
L = {r: dict(y=np.array([fam(sc, r)["mean"] for sc in scales]),
             e=np.array([fam(sc, r)["sd"] for sc in scales])) for r, _, _ in RUNGS}
ax.fill_between(X, L["base"]["y"], L["t2"]["y"], color=T.ORACLE, alpha=0.07, lw=0, zorder=1)
STYLE = {"base": dict(color=T.FREE, ls="-", marker="o", lw=1.2),
         "filtnat": dict(color=T.ORACLE, ls="-", marker="s", lw=0.9, alpha=0.75),
         "t15": dict(color=T.ORACLE, ls=(0, (2, 1.5)), marker="^", lw=0.9, alpha=0.75),
         "t2": dict(color=T.ORACLE, ls="-", marker="D", lw=1.3)}
for r, lab, q in RUNGS:
    ax.errorbar(X, L[r]["y"], yerr=L[r]["e"], ms=2.6, elinewidth=0.6, capsize=1.3, zorder=3,
                **STYLE[r])
for i in range(len(scales)):
    d = L["t2"]["y"][i] - L["base"]["y"][i]
    ax.text(X[i] / 1.06, L["t2"]["y"][i] + L["t2"]["e"][i] + 1.1, f"+{d:.1f}",
            fontsize=5.2, color=T.SUB, ha="right", va="bottom")
ends = {r: L[r]["y"][-1] for r, _, _ in RUNGS}
ANCHOR = {"t2": 88.6, "filtnat": 84.4, "t15": 80.2, "base": 76.0}
for r, lab, q in RUNGS:
    col = T.FREE if r == "base" else T.ORACLE
    a_ = STYLE[r].get("alpha", 1.0)
    ax.plot([X[-1] * 1.06, X[-1] * 1.22], [ends[r], ANCHOR[r]], color=col, lw=0.5,
            alpha=0.6 * a_, zorder=2)
    ax.text(X[-1] * 1.28, ANCHOR[r], lab, fontsize=5.2, color=col, va="center", alpha=a_)
ax.set_xscale("log"); ax.set_xlim(6.5e4, 1.5e7)
ax.set_xticks(X)
ax.set_xticklabels([f"{x/1e3:.0f}k" if x < 1e6 else "1.82M" for x in X], fontsize=6)
ax.minorticks_off()
ax.set_ylim(42, 92)
ax.set_xlabel("raw experience (pairs)")
ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)

# ---- B: the aligned arm as a scaling curve --------------------------------------
F = fit(rng)
grid = np.logspace(3.3, 6.5, 200)
lo, hi = F["band"](grid)
bx.fill_between(grid, lo, hi, color=T.FREE, alpha=0.13, lw=0, zorder=1)
bx.plot(grid, logistic(grid, *F["popt"]), color=T.FREE, lw=1.0, zorder=2)
bx.errorbar(F["x"], F["y"], yerr=F["e"], fmt="o", color=T.FREE, ms=2.6, lw=0,
            elinewidth=0.6, capsize=1.3, zorder=4)
afams = sorted(((f, D.family(f)) for f in D.runs.family.unique()
                if re.fullmatch(r"B26_lad\d*_?filtnat", str(f))), key=lambda t: t[1]["n_pairs"])
ax_ = np.array([f["n_pairs"] for _, f in afams])
ay = np.array([f["mean"] for _, f in afams]); ae = np.array([f["sd"] for _, f in afams])
agrid = np.logspace(3.6, np.log10(ax_.max()), 120)
apopt, _ = curve_fit(lambda N, m, s_: logistic(N, m, s_, F["popt"][2]), ax_, ay,
                     p0=(4.3, 0.5), sigma=ae, maxfev=40000)
adraws = []
for _ in range(300):
    try:
        pa, _ = curve_fit(lambda N, m, s_: logistic(N, m, s_, F["popt"][2]), ax_,
                          ay + rng.normal(0, np.maximum(ae, 0.5)), p0=apopt, maxfev=40000)
        adraws.append(logistic(agrid, *pa, F["popt"][2]))
    except Exception:
        pass
alo, ahi = np.percentile(np.array(adraws), [10, 90], axis=0)
bx.fill_between(agrid, alo, ahi, color=T.ORACLE, alpha=0.13, lw=0, zorder=1)
bx.plot(agrid, logistic(agrid, *apopt, F["popt"][2]), color=T.ORACLE, lw=1.0, zorder=2)
bx.errorbar(ax_, ay, yerr=ae, fmt="o", color=T.ORACLE, ms=2.6, lw=0, elinewidth=0.6,
            capsize=1.3, zorder=4)
aend = logistic(agrid[-1], *apopt, F["popt"][2])
bx.plot([agrid[-1]] * 2, [aend - 2.5, aend + 2.5], color=T.ORACLE, lw=0.8)
bx.text(agrid[-1], aend - 3.6, "all aligned\npairs", fontsize=4.8, color=T.ORACLE,
        ha="center", va="top", linespacing=1.25)
bx.text(3.4e3, 76, "aligned\n(oracle filter)", fontsize=5.2, color=T.ORACLE, ha="left",
        va="top", linespacing=1.3)
bx.text(2.6e5, 38, "unfiltered", fontsize=5.2, color=T.FREE, ha="left")
bx.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
bx.set_xscale("log"); bx.set_xlim(3e3, 4e6); bx.set_ylim(18, 92)
bx.set_xlabel("training pairs")
T.clean(bx)

# ---- C: diversity at three budgets ----------------------------------------------
dfams = {}
for f in D.runs.family.unique():
    if m := re.fullmatch(r"B26_div(\w*?)_?(\d+)c", str(f)):
        dfams.setdefault(m.group(1) or "30k", []).append((int(m.group(2)), f))
SHADE = {"30k": "#9ec9b8", "100k": "#4d9971", "300k": T.FREE}
for budget in ["30k", "100k", "300k"]:
    pts = sorted(dfams.get(budget, []))
    ks = [k for k, _ in pts]; fm = [D.family(f) for _, f in pts]
    cxcol = SHADE[budget]
    cx.errorbar(ks, [f["mean"] for f in fm], yerr=[f["sd"] for f in fm], fmt="-o",
                color=cxcol, ms=2.6, lw=1.0, elinewidth=0.6, capsize=1.4, zorder=3)
    cx.text(ks[-1] * 1.25, fm[-1]["mean"], f"{budget}\npairs", fontsize=5.2, color=cxcol,
            va="center", linespacing=1.2)
cx.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=2)
cx.text(1, 25.8, "chance", fontsize=5.2, color=T.SUB, ha="left")
cx.set_xscale("log")
cx.set_xticks([1, 3, 10, 25, 50]); cx.set_xticklabels([1, 3, 10, 25, 50]); cx.minorticks_off()
cx.set_xlim(0.8, 160); cx.set_ylim(18, 92)
cx.set_xlabel("children contributing")
T.clean(cx)

for a, l in zip((ax, bx, cx), "ABC"):
    T.panel(a, l, dx=-0.18)
T.save(fig, "fig3_composite")
