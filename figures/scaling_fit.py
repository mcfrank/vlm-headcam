"""Shared B26 scaling fit for fig2 (the data) and fig9 (the extrapolation): free-asymptote
logistic in log N over the B26_rand families, Monte-Carlo band over seed noise."""
import re, numpy as np
import data as D
from scipy.optimize import curve_fit

CHANCE = 25.0


def logistic(N, m, s, A):
    return CHANCE + (A - CHANCE) / (1 + np.exp(-(np.log10(N) - m) / s))


def fit(rng=None):
    rng = rng or np.random.default_rng(0)
    fams = sorted((int(mm.group(1)), f) for f in D.runs.family.unique()
                  if (mm := re.fullmatch(r"B26_rand_(\d+)", str(f))))
    if "B26_lad_base" in set(D.runs.family.astype(str)):
        fams.append((1_820_000, "B26_lad_base"))
    fam = [D.family(f) for _, f in fams]
    x = np.array([n for n, _ in fams], float)
    y = np.array([f["mean"] for f in fam]); e = np.array([f["sd"] for f in fam])
    popt, pcov = curve_fit(logistic, x, y, p0=(5, 0.8, 85), sigma=e,
                           bounds=([3, 0.1, 50], [9, 3, 100]), maxfev=40000)
    def band(grid):
        draws = []
        for _ in range(500):
            try:
                p, _ = curve_fit(logistic, x, y + rng.normal(0, np.maximum(e, 0.5)), p0=popt,
                                 bounds=([3, 0.1, 50], [9, 3, 100]), maxfev=40000)
                draws.append(logistic(grid, *p))
            except Exception:
                pass
        return np.percentile(np.array(draws), [10, 90], axis=0)
    return dict(x=x, y=y, e=e, n=[f["n"] for f in fam], popt=popt,
                A_sd=float(np.sqrt(pcov[2, 2])), band=band)
