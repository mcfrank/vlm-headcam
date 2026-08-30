"""Display item 3 — what closes the gap? Oracle information, referential selection, and
data diversity, each against raw experience. B-OTS throughout; the encoder breakdown joins
when the C8 ladder / diversity families land (detected below).

A: the ladder at three scales — matched subsamples trained unfiltered, with the oracle
   alignment filter, and with everything perfectly labeled (topline); the wedge is the
   referential headroom. (Word-selection rung dropped for clarity; it lives in the SI.)
B: the headroom decomposed — what the filter buys (filter - unfiltered) and what perfect
   labels add on top (topline - filter), per-seed paired differences on matched subsamples.
C: diversity at three budgets — whose data it is matters only once there is enough of it.
"""
import sys, re, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scaling_fit import CHANCE

if any(re.match(r"C8_.*(filtnat|div)", str(f)) for f in D.runs.family.unique()):
    print("  NOTE fig3: C8 ladder/diversity families exist — add the encoder breakdown")
rng = np.random.default_rng(0)

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(T.W2, 2.5),
                                 gridspec_kw=dict(width_ratios=[1.3, 1, 1], wspace=0.34))

# ---- A: the ladder at three scales ----------------------------------------------
RUNGS = [("base", "unfiltered", None), ("filtnat", "+ align filter", "which moments"),
         ("t2", "topline: all\nperfectly labeled", "which object")]
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
         "t2": dict(color=T.ORACLE, ls="-", marker="D", lw=1.3)}
for r, lab, q in RUNGS:
    ax.errorbar(X, L[r]["y"], yerr=L[r]["e"], ms=2.6, elinewidth=0.6, capsize=1.3, zorder=3,
                **STYLE[r])
for i in range(len(scales)):
    d = L["t2"]["y"][i] - L["base"]["y"][i]
    ha = "left" if i == 0 else "right"
    ax.text(X[i] * (1.10 if i == 0 else 1 / 1.06),
            L["t2"]["y"][i] + L["t2"]["e"][i] + (2.6 if i == 0 else 1.1),
            f"+{d:.1f}", fontsize=5.2, color=T.SUB, ha=ha, va="bottom")
ends = {r: L[r]["y"][-1] for r, _, _ in RUNGS}
ANCHOR = {"t2": 89.4, "filtnat": 83.6, "base": 77.0}
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

# ---- B: the headroom decomposed (paired per-seed differences, matched subsamples) ----
def per_seed(sc, rung):
    fname = f"B26_lad{sc}_{rung}" if sc else f"B26_lad_{rung}"
    d = D.runs[D.runs.family == fname]
    return dict(zip(d.seed, d.best_acc))

DELTAS = [("referential headroom\n(topline − unfiltered)", "t2", "base", T.SUB, "-", "o"),
          ("alignment filter\n− unfiltered", "filtnat", "base", T.ORACLE, "-", "s"),
          ("perfect labels\n− filter", "t2", "filtnat", T.ORACLE, (0, (2, 1.5)), "D")]
for lab, hi_r, lo_r, col, ls, mk in DELTAS:
    m_, e_ = [], []
    for sc in scales:
        a, b = per_seed(sc, hi_r), per_seed(sc, lo_r)
        diffs = [a[k] - b[k] for k in a if k in b]
        m_.append(np.mean(diffs)); e_.append(np.std(diffs, ddof=1))
    light = lo_r == "filtnat"
    bx.errorbar(X, m_, yerr=e_, color=col, ls=ls, marker=mk, ms=2.6, lw=1.0,
                elinewidth=0.6, capsize=1.3, zorder=3, alpha=0.75 if light else 1.0)
    anchor = {"referential": 8.2, "alignment": 4.6, "perfect": 0.9}[lab.split()[0]]
    bx.plot([X[-1] * 1.05, X[-1] * 1.25], [m_[-1], anchor], color=col, lw=0.45,
            alpha=0.5 if light else 0.7, zorder=2)
    bx.text(X[-1] * 1.3, anchor, lab, fontsize=5.0, color=col, va="center",
            linespacing=1.25, alpha=0.75 if light else 1.0)
bx.axhline(0, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
bx.set_xscale("log"); bx.set_xlim(6.5e4, 3e7)
bx.set_xticks(X)
bx.set_xticklabels([f"{x/1e3:.0f}k" if x < 1e6 else "1.82M" for x in X], fontsize=6)
bx.minorticks_off()
bx.set_ylim(-2, 15)
bx.set_xlabel("raw experience (pairs)")
bx.set_ylabel("oracle gain (Δ 4AFC points)")
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
