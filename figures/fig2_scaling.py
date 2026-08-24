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
ax.scatter(lit.n, lit.acc, marker="D", s=14, color=T.LIT, zorder=5)
for r, d in zip(lit.itertuples(), dy):
    if "2026" in r.source:
        ax.text(r.n * 1.18, r.acc + d, r.split.split("-")[0], fontsize=5.4, color=T.LIT, va="center")
    else:
        ax.text(r.n, r.acc - 1.8, "CVCL 2024", fontsize=5.4, color=T.LIT, va="top", ha="center")
ax.text(1.7e5, 30.5, "single-child models\n(SAYCam; Vong et al.)", fontsize=5.4, color=T.LIT,
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
bx.fill_between(tgrid, tlo, thi, color=T.FREE, alpha=0.14, lw=0, zorder=1)
bx.plot(tgrid, logistic(ngrid, *popt), color=T.FREE, lw=1.1, zorder=2)
obs_end = to_yr(x.max())
bx.axvline(obs_end, color=T.SUB, lw=0.6, ls=(0, (1, 2)))
bx.text(obs_end / 1.25, 92.5, "observed", fontsize=5.8, color=T.SUB, style="italic",
        ha="right", va="top")
bx.text(obs_end * 1.25, 92.5, "extrapolated", fontsize=5.8, color=T.SUB, style="italic", va="top")
# anchor lines: a child's first three years of waking input
for yr in (1, 2, 3):
    bx.axvline(yr, color=T.GRID, lw=0.7, zorder=0)
    bx.text(yr, 19.3, f"{yr} yr" if yr == 1 else f"{yr}", fontsize=5.4, color=T.SUB,
            ha="center", va="bottom")
# Wordbank CDI anchors over the same 60 words
wb = {(r.form, r.age): r.pred_4afc for r in WB.itertuples()}
comp = [(12, wb[("WG", 12)]), (18, wb[("WG", 18)])]
prod = [(24, wb[("WS", 24)]), (30, wb[("WS", 30)])]
bx.scatter([a / 12 for a, _ in comp], [v for _, v in comp], marker="o", s=17, color=T.CHILD,
           edgecolors="#8a6d1f", lw=0.5, zorder=5)
bx.scatter([a / 12 for a, _ in prod], [v for _, v in prod], marker="^", s=18, color=T.CHILD,
           edgecolors="#8a6d1f", lw=0.5, zorder=5)
for (a, v), (dx, ha, va, dv) in zip(comp + prod,
        [(1.0, "center", "top", -2.2), (0.93, "right", "center", 0),
         (1.08, "left", "center", 0), (1.0, "center", "bottom", 2.2)]):
    bx.text(a / 12 * dx, v + dv, f"{a} mo", fontsize=4.8, color="#8a6d1f", ha=ha, va=va)
from matplotlib.lines import Line2D
mk = dict(color=T.CHILD, markeredgecolor="#8a6d1f", markeredgewidth=0.5, lw=0)
bx.legend(handles=[Line2D([], [], marker="o", ms=3.6, label="understands", **mk),
                   Line2D([], [], marker="^", ms=3.8, label="produces (lower bound)", **mk)],
          title="children, Wordbank CDI", title_fontsize=5.4, fontsize=5.2,
          loc="center left", bbox_to_anchor=(0.02, 0.42), labelspacing=0.3,
          handletextpad=0.3, borderpad=0.2, alignment="left")
bx.get_legend().get_title().set_color("#8a6d1f")
for t in bx.get_legend().get_texts():
    t.set_color("#8a6d1f")
# children's measured 4AFC at school age
bx.axhspan(72, 82, color=T.CHILD, alpha=0.28, lw=0, zorder=1)
bx.text(11.5, 77, "children 5–12 yr\n(LEVANTE)", fontsize=5.2, color="#8a6d1f", ha="right",
        va="center", linespacing=1.4)
bx.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
bx.set_xscale("log"); bx.set_xlim(3e-3, 12); bx.set_ylim(18, 95)
bx.set_xlabel("developmental time (years of waking input)")
T.clean(bx)

for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)
T.save(fig, "fig2_scaling")
