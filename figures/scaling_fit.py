"""Shared B26 scaling fit for fig2 (the data) and fig9 (the extrapolation): free-asymptote
logistic in log N over the B26_rand families, Monte-Carlo band over seed noise."""
import re, numpy as np
import data as D
from scipy.optimize import curve_fit

CHANCE = 25.0


def logistic(N, m, s, A):
    return CHANCE + (A - CHANCE) / (1 + np.exp(-(np.log10(N) - m) / s))


def mc_band(x, y, sem, popt, grid, rng, bounds, sigma=None, ndraw=2000, pct=(10, 90)):
    """Monte-Carlo band for a fitted curve.

    Each draw perturbs the point MEANS by their standard errors (sd/sqrt(n_seeds)) and
    refits with the SAME weighting as the central fit, so the band and the curve it
    surrounds come from one estimator. This is sampling uncertainty in the fitted mean
    only: uncertainty in the functional form -- which dominates beyond the data -- is not
    quantified here, which is why figures fade the extrapolated segment.
    """
    draws = []
    for _ in range(ndraw):
        try:
            p, _ = curve_fit(logistic, x, y + rng.normal(0, sem), p0=popt, sigma=sigma,
                             bounds=bounds, maxfev=60000)
            draws.append(logistic(grid, *p))
        except Exception:
            pass
    return np.percentile(np.array(draws), list(pct), axis=0)


def fit(rng=None, enc="dinov3b"):
    """Free-asymptote logistic over the FINAL-corpus scaling families F_<enc>_rand_* + base."""
    rng = rng or np.random.default_rng(0)
    fams = sorted((int(mm.group(1)), f) for f in D.runs.family.unique()
                  if (mm := re.fullmatch(rf"F_{enc}_rand_(\d+)", str(f))))
    fams.append((None, f"F_{enc}_base"))
    fam = [D.family(f) for _, f in fams]
    x = np.array([fm["n_pairs"] if not np.isnan(fm.get("n_pairs", np.nan)) else n
                  for (n, _), fm in zip(fams, fam)], float)
    y = np.array([f["mean"] for f in fam]); e = np.array([f["sd"] for f in fam])
    popt, pcov = curve_fit(logistic, x, y, p0=(5, 0.8, 85), sigma=e,
                           bounds=([3, 0.1, 50], [9, 3, 100]), maxfev=40000)
    nseed = np.array([f["n"] for f in fam])
    sem = np.maximum(e / np.sqrt(nseed), 0.15)
    band = lambda grid: mc_band(x, y, sem, popt, grid, rng,
                                ([3, 0.1, 50], [9, 3, 100]), sigma=e)
    return dict(x=x, y=y, e=e, sem=sem, n=[f["n"] for f in fam], popt=popt,
                A_sd=float(np.sqrt(pcov[2, 2])), band=band)
