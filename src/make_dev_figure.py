"""Scaling re-expressed as developmental time, refined: natural-text asymptote (74, with the 81
clean-label ceiling shown separately), seed-uncertainty bands (Monte-Carlo through the fit), and
the actual observed points at their developmental-time positions. Illustrative extrapolation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

UTT_PER_HR, HR_PER_YEAR, BV_RATE = 820, 4000, 0.093
NAT_CEIL, LABEL_CEIL = 74.0, 81.0
INK, SUB, GRID = "#2c2c2a", "#6b6a66", "#e1e0d9"
rng = np.random.default_rng(0)


def sat(N, k, b):
    return NAT_CEIL - k * np.power(N, -b)


# per-seed observations (Konkle test-60)
rand = {10000: [27.6, 28.3, 34.0], 30000: [38.5, 39.0, 37.5], 100000: [42.0, 42.0, 36.7],
        300000: [55.4, 54.1, 59.1], 911000: [63.7, 63.5, 60.7]}
align = {10000: [58.9, 69.8, 63.1], 30000: [69.2, 68.1, 66.6], 85000: [73.3, 70.2, 70.8]}


def agg(d):
    x = np.array(sorted(d)); m = np.array([np.mean(d[k]) for k in x]); s = np.array([np.std(d[k], ddof=1) for k in x])
    return x, m, s


rx, rm, rs = agg(rand); ax_, am, as_ = agg(align)
hrs = np.logspace(0.7, 4.7, 300); yrs = hrs / HR_PER_YEAR
obs_yr_r = rx.max() / UTT_PER_HR / HR_PER_YEAR
obs_yr_a = ax_.max() / (UTT_PER_HR * BV_RATE) / HR_PER_YEAR


def band(x, m, s, npairs, p0):
    fits = []
    for _ in range(400):
        try:
            p, _ = curve_fit(sat, x, m + rng.normal(0, np.maximum(s, 0.5)), p0=p0, maxfev=5000)
            fits.append(np.clip(sat(npairs, *p), 25, NAT_CEIL))
        except Exception:
            pass
    f = np.array(fits)
    pm, _ = curve_fit(sat, x, m, p0=p0, maxfev=20000)
    return np.clip(sat(npairs, *pm), 25, NAT_CEIL), np.percentile(f, 16, 0), np.percentile(f, 84, 0), pm


fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)

# random (all speech) — data-backed, with band + points
cr, lo, hi, pr = band(rx, rm, rs, hrs * UTT_PER_HR, [500, 0.2])
ax.fill_between(yrs, lo, hi, color="#b0655a", alpha=0.15, zorder=1)
ax.plot(yrs, cr, color="#b0655a", lw=2.3, label="learns from all speech (no filtering)", zorder=3)
ax.errorbar(rx / UTT_PER_HR / HR_PER_YEAR, rm, yerr=rs, fmt="o", color="#b0655a", ms=5, capsize=2, zorder=4)

# aligned at BabyView ~9% — data-backed, with band + points
ca, alo, ahi, pa = band(ax_, am, as_, hrs * UTT_PER_HR * BV_RATE, [80, 0.2])
ax.fill_between(yrs, alo, ahi, color="#1d9e75", alpha=0.15, zorder=1)
ax.plot(yrs, ca, color="#1d9e75", lw=2.3, label="aligned-only, BabyView (~9%)", zorder=3)
ax.errorbar(ax_ / (UTT_PER_HR * BV_RATE) / HR_PER_YEAR, am, yerr=as_, fmt="o", color="#1d9e75", ms=5, capsize=2, zorder=4)

# hypothetical environments (no data) — thin dashed, from the aligned fit
for rate, col, lab in [(0.03, "#9ecae1", "sparse (3%)"), (0.20, "#08519c", "rich (20%)")]:
    ax.plot(yrs, np.clip(sat(hrs * UTT_PER_HR * rate, *pa), 25, NAT_CEIL), color=col, lw=1.6, ls=(0, (4, 3)),
            label=f"aligned-only, {lab}", zorder=2)

ax.axhline(LABEL_CEIL, color=SUB, lw=1.0, ls=(0, (2, 2))); ax.text(0.0016, LABEL_CEIL - 2.2, "clean-label ceiling (81)", color=SUB, fontsize=8.5)
ax.axhline(NAT_CEIL, color=GRID, lw=1.0); ax.text(0.0016, NAT_CEIL + 0.5, "natural-text ceiling (~74)", color=SUB, fontsize=8.5)
ax.axvspan(0.001, max(obs_yr_r, obs_yr_a), color="#f3f1ea", zorder=0); ax.text(0.0016, 27, "observed", color=SUB, fontsize=8.5)
for y in [1, 2, 3, 5]:
    ax.axvline(y, color=GRID, lw=0.8); ax.text(y, 70.5, f"{y}yr", color=SUB, fontsize=8.5, ha="center")
ax.set_xscale("log"); ax.set_xlim(0.001, 8); ax.set_ylim(24, 83)
ax.set_xlabel("developmental time (years, at 4,000 h/year)", fontsize=11, color=SUB)
ax.set_ylabel("Konkle 4AFC (word–object accuracy)", fontsize=11, color=SUB)
ax.set_title("What scaling implies across developmental time", fontsize=13, color=INK, loc="left", pad=10)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=SUB, length=0)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig("/data2/mcfrank/vlm-headcam/book_figs/fig_dev_time.png", bbox_inches="tight")
print("random fit", pr, "| aligned fit", pa)
print("1yr: unfiltered", round(float(np.clip(sat(4000*UTT_PER_HR,*pr),25,NAT_CEIL)),1),
      "| aligned-9%", round(float(np.clip(sat(4000*UTT_PER_HR*BV_RATE,*pa),25,NAT_CEIL)),1))
print("wrote fig_dev_time.png")
