"""Display item 2 — how word learning scales with input: the data.

Bundle-2 (B26) production scaling curve: 2026.1 English-filtered corpus, 50 children, DINOv3-B
region-MIL, dev-117 selection, test-60 reporting. Each seed is an independent subsample, so the
sd across a family is subsample+init variance.

4AFC vs training pairs, both arms: unfiltered (free-asymptote logistic in log N, fit shared
with fig9 via scaling_fit) and aligned — the filtnat rungs of the ladder at each scale, fit
with the same logistic constrained to the unfiltered asymptote, ending at the aligned pairs
that exist. Published single-child SAYCam models as external reference points. The
developmental extrapolation lives in fig9.
"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scaling_fit import fit, logistic, CHANCE

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
LIT = pd.read_csv(R / "literature.csv")
WB = pd.read_csv(R / "wordbank_anchors.csv")

rng = np.random.default_rng(0)

F = fit(rng)
x, y, e, popt = F["x"], F["y"], F["e"], F["popt"]
grid = np.logspace(3.3, 7.8, 260)
lo, hi = F["band"](grid)
A_fit, A_sd = popt[2], F["A_sd"]
print(f"  NOTE figS1: fitted asymptote {A_fit:.1f} ± {A_sd:.1f}; seeds per point {F['n']}")

# the aligned (oracle-filter) arm: filtnat rungs of the ladder at every scale run so far
from scipy.optimize import curve_fit
afams = sorted(((f, D.family(f)) for f in D.runs.family.unique()
                if re.fullmatch(r"B26_lad\d*_?filtnat", str(f))), key=lambda t: t[1]["n_pairs"])
ax_ = np.array([f["n_pairs"] for _, f in afams])
ay = np.array([f["mean"] for _, f in afams]); ae = np.array([f["sd"] for _, f in afams])
agrid = np.logspace(3.6, np.log10(ax_.max()), 140)
apopt, _ = curve_fit(lambda N, m, s_: logistic(N, m, s_, popt[2]), ax_, ay, p0=(4.3, 0.5),
                     sigma=ae, maxfev=40000)
adraws = []
for _ in range(400):
    try:
        pa, _ = curve_fit(lambda N, m, s_: logistic(N, m, s_, popt[2]), ax_,
                          ay + rng.normal(0, np.maximum(ae, 0.5)), p0=apopt, maxfev=40000)
        adraws.append(logistic(agrid, *pa, popt[2]))
    except Exception:
        pass
alo, ahi = np.percentile(np.array(adraws), [10, 90], axis=0)

fig, ax = plt.subplots(figsize=(T.W15, 2.6))

# ---- A: the data -----------------------------------------------------------------
ax.fill_between(grid, lo, hi, color=T.FREE, alpha=0.14, lw=0, zorder=1)
ax.plot(grid, logistic(grid, *popt), color=T.FREE, lw=1.0, zorder=2)
ax.errorbar(x, y, yerr=e, fmt="o", color=T.FREE, ms=3.2, lw=0, elinewidth=0.7,
            capsize=1.6, zorder=4)
ax.axhline(A_fit, color=T.FREE, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.text(4.3e3, A_fit + 1, f"fitted asymptote {A_fit:.0f} ± {A_sd:.0f}", fontsize=5.6, color=T.FREE)
ax.text(1.35e6, 70.5, "unfiltered", fontsize=6, color=T.FREE, ha="left", va="center")
# aligned arm: what the same corpus buys if an oracle keeps only the referential moments
ax.fill_between(agrid, alo, ahi, color=T.ORACLE, alpha=0.14, lw=0, zorder=1)
ax.plot(agrid, logistic(agrid, *apopt, popt[2]), color=T.ORACLE, lw=1.0, zorder=2)
ax.errorbar(ax_, ay, yerr=ae, fmt="o", color=T.ORACLE, ms=3.2, lw=0, elinewidth=0.7,
            capsize=1.6, zorder=4)
ax.plot([agrid[-1]] * 2, [logistic(agrid[-1], *apopt, popt[2]) - 2.5,
                          logistic(agrid[-1], *apopt, popt[2]) + 2.5], color=T.ORACLE, lw=0.8)
ax.text(3.6e3, 73, "aligned\n(oracle filter)", fontsize=5.6, color=T.ORACLE, ha="left",
        va="top", linespacing=1.4)
ax.text(agrid[-1], 79.8, "all aligned\npairs", fontsize=5.0, color=T.ORACLE, ha="center",
        va="top", linespacing=1.3)
# external references: single-child SAYCam models, same 60 Konkle categories
lit = LIT.groupby(["source", "split"], sort=False).agg(n=("n_utterances", "first"),
                                                        acc=("konkle_acc", "mean")).reset_index()
lit = lit.sort_values("acc").reset_index(drop=True); dy = np.zeros(len(lit))
for i in range(1, len(lit)):
    if lit.acc[i] - lit.acc[i - 1] < 1.5 and abs(np.log10(lit.n[i] / lit.n[i - 1])) < 0.1:
        dy[i - 1] -= 0.8; dy[i] += 0.8
LIT24, LIT26 = "#D8A6CD", T.LIT                       # light = 2024, dark = 2026
lit["col"] = [LIT24 if "2024" in src else LIT26 for src in lit.source]
ax.scatter(lit.n, lit.acc, marker="D", s=14, c=lit.col, zorder=5)
for r, d in zip(lit.itertuples(), dy):
    if "2026" in r.source:
        ax.text(r.n * 1.18, r.acc + d, r.split.split("-")[0], fontsize=5.4, color=LIT26, va="center")
    else:
        ax.text(r.n * 1.15, r.acc - 1.0, "CVCL 2024", fontsize=5.4, color=LIT24, va="top", ha="left")
ax.text(1.7e5, 30.5, "single-child models\n(SAYCam; Vong et al.)", fontsize=5.4, color=LIT26,
        ha="left", va="center")
ax.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(3.5e6, 22.2, "chance", fontsize=5.6, color=T.SUB, ha="right")
ax.set_xscale("log"); ax.set_xlim(3e3, 4e6); ax.set_ylim(18, 95)
ax.set_xlabel("training pairs"); ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
T.save(fig, "figS1_scaling_saycam")
