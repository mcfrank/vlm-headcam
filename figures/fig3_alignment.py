"""Display item 3 — alignment is the factor: the same accuracy from ~10x less data.

Single panel. The unfiltered scaling curve (thin, grey — the same B-OTS curve as fig2)
against the aligned arm: training only on the pairs the oracle marks as referential (the
filtnat rungs of the matched-subsample ladder). Same fitted ceiling; the curve shifts a
decade left. Horizontal arrows mark the data equivalence at the aligned data points, and
the aligned arm ends where the corpus's referential moments run out — the efficient curve
exists only if something can select those moments.

B-OTS throughout; encoder breakdown joins when C8 filtnat families land (NOTE below).
The ladder/topline decomposition and diversity sweeps live in the SI.
"""
import sys, re, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scaling_fit import fit, logistic, CHANCE

if any(re.match(r"C8_.*filtnat", str(f)) for f in D.runs.family.unique()):
    print("  NOTE fig3: C8 filtnat families exist — add the encoder breakdown")
rng = np.random.default_rng(0)

F = fit(rng)
popt = F["popt"]; A = popt[2]
afams = sorted(((f, D.family(f)) for f in D.runs.family.unique()
                if re.fullmatch(r"B26_lad\d*_?filtnat", str(f))), key=lambda t: t[1]["n_pairs"])
ax_ = np.array([f["n_pairs"] for _, f in afams])
ay = np.array([f["mean"] for _, f in afams]); ae = np.array([f["sd"] for _, f in afams])
apopt, _ = curve_fit(lambda N, m, s_: logistic(N, m, s_, A), ax_, ay, p0=(4.3, 0.5),
                     sigma=ae, maxfev=40000)
agrid = np.logspace(3.6, np.log10(ax_.max()), 140)
adraws = []
for _ in range(400):
    try:
        pa, _ = curve_fit(lambda N, m, s_: logistic(N, m, s_, A), ax_,
                          ay + rng.normal(0, np.maximum(ae, 0.5)), p0=apopt, maxfev=40000)
        adraws.append(logistic(agrid, *pa, A))
    except Exception:
        pass
alo, ahi = np.percentile(np.array(adraws), [10, 90], axis=0)

fig, ax = plt.subplots(figsize=(T.W15, 2.7))

# the unfiltered curve as a known reference (it is fig2's B-OTS curve)
grid = np.logspace(3.3, 6.6, 220)
ax.plot(grid, logistic(grid, *popt), color=T.NEUTRAL, lw=1.0, zorder=2)
ax.errorbar(F["x"], F["y"], yerr=F["e"], fmt="o", color=T.NEUTRAL, ms=2.6, lw=0,
            elinewidth=0.6, capsize=1.3, zorder=3)
ax.text(1.05e6, 68, "unfiltered\n(as in Fig. 2)", fontsize=5.6, color="#8a8a86", ha="left",
        va="top", linespacing=1.35)

# the aligned arm
ax.fill_between(agrid, alo, ahi, color=T.ORACLE, alpha=0.14, lw=0, zorder=1)
ax.plot(agrid, logistic(agrid, *apopt, A), color=T.ORACLE, lw=1.2, zorder=4)
ax.errorbar(ax_, ay, yerr=ae, fmt="o", color=T.ORACLE, ms=3.2, lw=0, elinewidth=0.7,
            capsize=1.6, zorder=5)
aend = logistic(agrid[-1], *apopt, A)
ax.plot([agrid[-1]] * 2, [aend - 2.6, aend + 2.6], color=T.ORACLE, lw=0.9, zorder=5)
ax.text(agrid[-1] * 1.12, aend - 3.2, "all referential pairs\nin the corpus", fontsize=5.2,
        color=T.ORACLE, ha="left", va="top", linespacing=1.35)
ax.text(4.4e3, 62, "aligned only\n(oracle filter)", fontsize=6.0, color=T.ORACLE, ha="left",
        va="top", linespacing=1.35)

# data equivalence: at each aligned point's accuracy, how much unfiltered data matches it?
inv = lambda y, m, s_: 10 ** (m - s_ * np.log((A - CHANCE) / (y - CHANCE) - 1))
for y_lev in (ay[0], ay[1]):
    xa = inv(y_lev, *apopt)
    xu = inv(y_lev, popt[0], popt[1])
    ax.annotate("", xy=(xu, y_lev), xytext=(xa, y_lev),
                arrowprops=dict(arrowstyle="-|>", color=T.INK, lw=0.8, shrinkA=2, shrinkB=1))
    ax.text(np.sqrt(xa * xu), y_lev + 1.0, f"{xu / xa:.0f}×", fontsize=6.2, color=T.INK,
            ha="center", va="bottom", fontweight="bold")
print(f"  NOTE fig3: data-equivalence {inv(ay[0], *apopt):,.0f} vs "
      f"{inv(ay[0], popt[0], popt[1]):,.0f} pairs at {ay[0]:.1f}%; shift "
      f"{popt[0] - apopt[0]:.2f} decades at midpoint")

ax.axhline(CHANCE, color=T.SUB, lw=0.6, ls=(0, (4, 3)))
ax.text(3.6e6, 22.3, "chance", fontsize=5.6, color=T.SUB, ha="right")
ax.set_xscale("log"); ax.set_xlim(3e3, 4e6); ax.set_ylim(18, 92)
ax.set_xlabel("training pairs"); ax.set_ylabel("Konkle 4AFC (%)")
T.clean(ax)
T.save(fig, "fig3_alignment")
