"""Display item 2 — how does word learning scale with the input a child actually receives?

Bundle-2 (B26) production scaling curve: 2026.1 English-filtered corpus, 50 children, DINOv3-B
region-MIL, dev-117 selection, test-60 reporting. Each seed is an independent subsample, so the
sd across a family is subsample+init variance.

A: the data. 4AFC vs training pairs with a free-asymptote logistic in log N (chance floor; the
   asymptote is now identified by the curve itself). Published single-child SAYCam models
   (CVCL 2024; Vong & Lake 2026 — same 60 Konkle categories) as external reference points.
B: the extrapolation. The fitted curve alone, carried out over child developmental time, with
   1/2/3-yr anchor lines, Wordbank CDI child anchors (predicted 4AFC over the same 60 words:
   know the word -> correct, else guess), and children's LEVANTE vocabulary band at 5-12 yr.

When the B26 ladder families land (B26_lad*), the aligned arm belongs on panel A again —
detected below and currently absent.
"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
LIT = pd.read_csv(R / "literature.csv")
WB = pd.read_csv(R / "wordbank_anchors.csv")

UTT_PER_HR, HR_PER_YEAR = 820, 4000          # caregiver utterances/hr; waking hrs/yr
CHANCE = 25.0
rng = np.random.default_rng(0)

fams = sorted((int(m.group(1)), f) for f in D.runs.family.unique()
              if (m := re.fullmatch(r"B26_rand_(\d+)", str(f))))
x = np.array([n for n, _ in fams], float)
if "B26_lad_base" in set(D.runs.family.astype(str)):     # full-corpus run lands as the top point
    fams.append((1_820_000, "B26_lad_base"))              # 2026.1 corpus size; repoint to corpus csv
    x = np.array([n for n, _ in fams], float)
fam = [D.family(f) for _, f in fams]
y = np.array([f["mean"] for f in fam]); e = np.array([f["sd"] for f in fam])
n_seeds = [f["n"] for f in fam]
if any(D.runs.family.astype(str).str.startswith("B26_lad")):
    print("  NOTE fig2: B26_lad* families exist — add the aligned arm to panel A")


def logistic(N, m, s, A):
    return CHANCE + (A - CHANCE) / (1 + np.exp(-(np.log10(N) - m) / s))


popt, pcov = curve_fit(logistic, x, y, p0=(5, 0.8, 85), sigma=e,
                       bounds=([3, 0.1, 50], [9, 3, 100]), maxfev=40000)
grid = np.logspace(3.3, 7.8, 260)
draws = []
for _ in range(500):
    try:
        p, _ = curve_fit(logistic, x, y + rng.normal(0, np.maximum(e, 0.5)), p0=popt,
                         bounds=([3, 0.1, 50], [9, 3, 100]), maxfev=40000)
        draws.append(logistic(grid, *p))
    except Exception:
        pass
lo, hi = np.percentile(np.array(draws), [10, 90], axis=0)
A_fit, A_sd = popt[2], np.sqrt(pcov[2, 2])
print(f"  NOTE fig2: fitted asymptote {A_fit:.1f} ± {A_sd:.1f}; seeds per point {n_seeds}")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6))

# ---- A: the data -----------------------------------------------------------------
ax.fill_between(grid, lo, hi, color=T.FREE, alpha=0.14, lw=0, zorder=1)
ax.plot(grid, logistic(grid, *popt), color=T.FREE, lw=1.0, zorder=2)
ax.errorbar(x, y, yerr=e, fmt="o", color=T.FREE, ms=3.2, lw=0, elinewidth=0.7,
            capsize=1.6, zorder=4)
ax.axhline(A_fit, color=T.FREE, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.text(4.3e3, A_fit + 1, f"fitted asymptote {A_fit:.0f} ± {A_sd:.0f}", fontsize=5.6, color=T.FREE)
ax.text(1.05e6, 70.5, "BabyView\nunfiltered", fontsize=6, color=T.FREE, ha="right", va="top")
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

# ---- B: the extrapolation --------------------------------------------------------
to_yr = lambda n: n / UTT_PER_HR / HR_PER_YEAR
tgrid = np.logspace(np.log10(3e-3), np.log10(12), 260)
ngrid = tgrid * UTT_PER_HR * HR_PER_YEAR
tdraws = []
for _ in range(500):
    try:
        p, _ = curve_fit(logistic, x, y + rng.normal(0, np.maximum(e, 0.5)), p0=popt,
                         bounds=([3, 0.1, 50], [9, 3, 100]), maxfev=40000)
        tdraws.append(logistic(ngrid, *p))
    except Exception:
        pass
tlo, thi = np.percentile(np.array(tdraws), [10, 90], axis=0)
obs = tgrid <= to_yr(x.max())                      # solid where we have data, faded beyond
ext = tgrid >= to_yr(x.max())
bx.fill_between(tgrid[obs], tlo[obs], thi[obs], color=T.FREE, alpha=0.14, lw=0, zorder=1)
bx.fill_between(tgrid[ext], tlo[ext], thi[ext], color=T.FREE, alpha=0.06, lw=0, zorder=1)
bx.plot(tgrid[obs], logistic(ngrid[obs], *popt), color=T.FREE, lw=1.1, zorder=2)
bx.plot(tgrid[ext], logistic(ngrid[ext], *popt), color=T.FREE, lw=1.1, alpha=0.45, zorder=2)
# anchor lines: a child's first three years of waking input
for yr in (1, 2, 3):
    bx.axvline(yr, color=T.GRID, lw=0.7, zorder=0)
    bx.text(yr, 19.3, f"{yr} yr" if yr == 1 else f"{yr}", fontsize=5.4, color=T.SUB,
            ha="center", va="bottom")
# Wordbank CDI trajectories over the same 60 words (children plotted at their age)
CDI_INK = "#8a6d1f"
for form, meas, mk in [("WG", "understands", "o"), ("WS", "produces", "^")]:
    d = WB[WB.form == form].sort_values("age")
    bx.plot(d.age / 12, d.pred_4afc, marker=mk, ms=2.6, lw=0.9, color=T.CHILD,
            markeredgecolor=CDI_INK, markeredgewidth=0.4, zorder=5)
    end = d.iloc[-1]
    if form == "WS":
        bx.text(end.age / 12 * 1.07, end.pred_4afc - 2.0, meas, fontsize=5.2, color=CDI_INK,
                va="top", ha="left")
    else:
        bx.text(end.age / 12 * 1.06, end.pred_4afc - 4.5, meas, fontsize=5.2, color=CDI_INK,
                va="top", ha="left")
bx.text(WB.age.min() / 12 * 0.92, WB.pred_4afc.min() + 1.5, "children\n(Wordbank CDI)",
        fontsize=5.2, color=CDI_INK, ha="right", va="center", linespacing=1.4)
bx.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
bx.set_xscale("log"); bx.set_xlim(3e-3, 12); bx.set_ylim(18, 95)
bx.set_xlabel("developmental time (years of waking input)")
T.clean(bx)

for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)
T.save(fig, "fig2_scaling")
