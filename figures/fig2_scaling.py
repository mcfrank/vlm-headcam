"""Display item 2 — how does word learning scale with the input a child actually receives?

A: Konkle 4AFC against training pairs. Two arms from the same corpus: the unfiltered stream
   (what a learner gets for free) and the oracle-aligned stream (only the ~9% of pairs Gemini
   marks as referential). Published single-child models (CVCL; Vong & Lake 2026, three SAYCam
   children) as reference points. Fit is a logistic in log N: chance at no data, saturating
   toward a ceiling.
B: the same two arms in developmental time. A pair costs one utterance of child time on the
   unfiltered arm but 1/0.09 utterances on the aligned arm, so the arms separate on this axis:
   at matched waking time, the gap between them is what referential selection buys. Children's
   LEVANTE vocabulary accuracy at 5–12 yr is calibration of scale, not a matched task.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
C = pd.read_csv(R / "corpus.csv").set_index("key").value.to_dict()
LIT = pd.read_csv(R / "literature.csv")

UTT_PER_HR, HR_PER_YEAR = 820, 4000          # caregiver utterances/hr; waking hrs/yr
CHANCE = 25.0
CEIL = D.claim("ladder_vision")["value"]      # clean-label ceiling
ALIGNED_FRAC = C["aligned_pairs"] / C["pairs"]   # a learner must hear 1/frac utterances per aligned pair
rng = np.random.default_rng(0)

ARMS = {
    "unfiltered": [("scale_rand_10k", 1e4), ("scale_rand_30k", 3e4), ("scale_rand_100k", 1e5),
                   ("scale_rand_300k", 3e5), ("scale_rand_911k", 9.11e5), ("scale_full", 1.145e6)],
    "aligned":    [("scale_align_10k", 1e4), ("scale_align_30k", 3e4), ("scale_align_85k", 8.5e4)],
}
COL = {"unfiltered": T.FREE, "aligned": T.ORACLE}
TIME_COST = {"unfiltered": 1.0, "aligned": 1.0 / ALIGNED_FRAC}   # utterances heard per pair


def logistic(N, m, s):
    """Chance at N -> 0, CEIL as N -> inf; midpoint 10**m pairs, slope s in decades."""
    return CHANCE + (CEIL - CHANCE) / (1 + np.exp(-(np.log10(N) - m) / s))


GRID = {"unfiltered": np.logspace(3.7, 7.6, 240),
        # the aligned arm cannot extend past the aligned pairs that exist in the corpus
        "aligned": np.logspace(3.7, np.log10(C["aligned_pairs"]), 120)}
fits = {}
for arm, pts in ARMS.items():
    grid = GRID[arm]
    x = np.array([n for _, n in pts]); cl = [D.claim(c) for c, _ in pts]
    y = np.array([c["value"] for c in cl]); e = np.array([c["sd"] or 1.5 for c in cl])
    prov = np.array([c["provisional"] for c in cl])
    popt, _ = curve_fit(logistic, x, y, p0=(5.0, 0.8), maxfev=20000)
    draws = []
    for _ in range(500):
        try:
            p, _ = curve_fit(logistic, x, y + rng.normal(0, np.maximum(e, 0.5)), p0=popt, maxfev=20000)
            draws.append(logistic(grid, *p))
        except Exception:
            pass
    lo, hi = np.percentile(np.array(draws), [10, 90], axis=0)
    fits[arm] = dict(x=x, y=y, e=e, prov=prov, popt=popt, lo=lo, hi=hi)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6))


def draw_arm(a, arm, xscale):
    f = fits[arm]; c = COL[arm]; grid = GRID[arm]
    a.fill_between(grid * xscale, f["lo"], f["hi"], color=c, alpha=0.14, lw=0, zorder=1)
    a.plot(grid * xscale, logistic(grid, *f["popt"]), color=c, lw=1.0, zorder=2)
    a.errorbar(f["x"] * xscale, f["y"], yerr=f["e"], fmt="o", color=c, ms=3.2, lw=0,
               elinewidth=0.7, capsize=1.6, zorder=4)
    if f["prov"].any():
        a.scatter(f["x"][f["prov"]] * xscale, f["y"][f["prov"]], s=44, facecolors="none",
                  edgecolors=T.PROV, lw=0.9, zorder=5)
    if arm == "aligned":                       # end-of-corpus marker
        a.plot([grid[-1] * xscale] * 2, [logistic(grid[-1], *f["popt"]) - 2.5,
                                         logistic(grid[-1], *f["popt"]) + 2.5], color=c, lw=0.8)


def refs(a):
    for s in (CHANCE, CEIL):
        a.axhline(s, color=T.SUB if s == CHANCE else T.ORACLE, lw=0.6,
                  ls=(0, (4, 3)) if s == CHANCE else (0, (2, 2)))
    a.set_ylim(20, 90); a.set_xscale("log")
    T.clean(a)


# ---- A: pairs axis ---------------------------------------------------------------
for arm in ARMS:
    draw_arm(ax, arm, 1.0)
# published single-child models, placed at their utterance counts (one pair per utterance here)
lit = LIT.groupby(["source", "split"], sort=False).agg(n=("n_utterances", "first"),
                                                         acc=("konkle_acc", "mean")).reset_index()
ax.scatter(lit.n, lit.acc, marker="D", s=14, color=T.LIT, zorder=5)
# nudge labels of points that sit within 1.5 pts of each other apart
lit = lit.sort_values("acc").reset_index(drop=True); dy = np.zeros(len(lit))
for i in range(1, len(lit)):
    if lit.acc[i] - lit.acc[i - 1] < 1.5 and abs(np.log10(lit.n[i] / lit.n[i - 1])) < 0.1:
        dy[i - 1] -= 0.8; dy[i] += 0.8
for r, d in zip(lit.itertuples(), dy):
    if "2026" in r.source:
        ax.text(r.n * 1.18, r.acc + d, r.split.split("-")[0], fontsize=5.4, color=T.LIT, va="center")
    else:
        ax.text(r.n, r.acc - 1.8, "CVCL 2024", fontsize=5.4, color=T.LIT, va="top", ha="center")
ax.text(1.7e5, 30.5, "single-child models\n(SAYCam; Vong et al.)", fontsize=5.4, color=T.LIT,
        ha="left", va="center")
ax.text(2.9e6, 27, "chance", fontsize=5.6, color=T.SUB, ha="right")
ax.text(4.3e3, CEIL + 1, "clean-label ceiling", fontsize=5.6, color=T.ORACLE)
ax.text(1.25e5, 73.5, "aligned (oracle)\nall such pairs\nin the corpus", fontsize=5.6,
        color=T.ORACLE, ha="left", va="center")
ax.text(1.3e5, 44, "unfiltered", fontsize=6, color=T.FREE, ha="left")
ax.set_xlim(4e3, 3e6)
ax.set_xlabel("training pairs"); ax.set_ylabel("Konkle 4AFC (%)")
refs(ax)

# ---- B: developmental time -------------------------------------------------------
to_yr = lambda n_utt: n_utt / UTT_PER_HR / HR_PER_YEAR
for arm in ARMS:
    draw_arm(bx, arm, to_yr(TIME_COST[arm]))
bx.axhspan(72, 82, color=T.CHILD, alpha=0.28, lw=0, zorder=1)
bx.text(3.4e-3, 77, "children 5–12 yr", fontsize=5.6, color="#8a6d1f", ha="left", va="center")
bx.text(0.42, 74.5, "aligned", fontsize=5.6, color=T.ORACLE, ha="left", va="center")
bx.text(0.42, 61, "unfiltered", fontsize=5.6, color=T.FREE, ha="left", va="center")
for yr in (1, 3, 5):
    bx.axvline(yr, color=T.GRID, lw=0.6, zorder=0)
    bx.text(yr, 21.5, f"{yr} yr", fontsize=5.4, color=T.SUB, ha="center")
obs_end = max(to_yr(f["x"].max() * TIME_COST[a]) for a, f in fits.items())
bx.axvline(obs_end, color=T.SUB, lw=0.6, ls=(0, (1, 2)))
bx.text(obs_end / 1.25, 86, "observed", fontsize=5.8, color=T.SUB, style="italic", ha="right")
bx.text(obs_end * 1.25, 86, "extrapolated", fontsize=5.8, color=T.SUB, style="italic")
bx.set_xlim(3e-3, 8)
bx.set_xlabel("developmental time (years of waking input)")
refs(bx)

for a, l in zip((ax, bx), "AB"):
    T.panel(a, l)
note = D.provisional_note([c for pts in ARMS.values() for c, _ in pts])
if note:
    print("  NOTE fig2:", note)
T.save(fig, "fig2_scaling")
