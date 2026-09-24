"""human check of the Gemini referential-alignment annotation.

1,500 (frame, utterance) pairs from the final corpus, stratified by Gemini score (half are
Gemini zeros), rated blind by two lab members with a binary question pinned to Gemini's
50-point anchor: does the utterance refer to a concrete object visible in the frame?

A: calibration. The share of human ratings saying "yes" per Gemini-score bin (both raters
   pooled, Wilson 95% CI; thin lines = each rater). The two bars at 0 split the Gemini zeros
   by whether the utterance contains a concrete-object noun. The >= 50 rule sits where
   humans go from ~1-in-8 yes to ~2-in-3 yes.
B: the >= 50 rule against human consensus (items where both raters agree, n = 1,315) as the
   threshold is raised: precision keeps climbing, recall falls faster, so F1 peaks at the
   paper's threshold. The share of the corpus kept at each threshold runs along the bottom.
   These are sample metrics; corpus-weighted recall is lower (0.56 at 50) because the
   corpus is 89% Gemini zeros and raters accept ~10% of those.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
cal = pd.read_csv(R / "gemini_human_check_calibration.csv").set_index("stratum")
thr = pd.read_csv(R / "gemini_human_check_threshold.csv")
thr = thr[thr.who == "consensus"].sort_values("threshold")
hc = pd.read_csv(R / "gemini_human_check.csv").set_index("metric")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.30))

# ---- A: calibration ---------------------------------------------------------------
BINS = [("0_other", "0\nno object\nnoun"), ("0_noun", "0\nobject\nnoun"), ("1-49", "1–49"),
        ("50-60", "50–60"), ("70-75", "70–75"), ("80", "80"), ("90-95", "90–95"), ("100", "100")]
xs = np.arange(len(BINS))
for i, (b, lab) in enumerate(BINS):
    r = cal.loc[b]
    aligned = b not in ("0_other", "0_noun", "1-49")
    col = T.ORACLE if aligned else T.NEUTRAL
    ax.bar(i, r.yes_rate, width=0.64, color=col, zorder=2, alpha=0.9 if b != "1-49" else 0.5)
    ax.errorbar([i], [r.yes_rate], yerr=[[r.yes_rate - r.yes_lo], [r.yes_hi - r.yes_rate]],
                fmt="none", ecolor=T.INK, elinewidth=0.7, capsize=1.8, zorder=4)
    for k, mk in (("yes_rate_R1", "_"), ("yes_rate_R2", "_")):
        ax.plot([i - 0.22, i + 0.22], [r[k], r[k]], color=T.INK, lw=0.5, alpha=0.55, zorder=3)
    ax.text(i, 1.02, f"{int(r.n_ratings)}", fontsize=4.6, color=T.SUB, ha="center", va="bottom")
ax.text(-0.55, 1.075, "ratings per bin", fontsize=4.6, color=T.SUB, ha="left", va="bottom")
ax.axvline(2.5, color=T.SUB, lw=0.6, ls=(0, (1, 2)), zorder=1)
ax.text(2.62, 0.96, "aligned (≥ 50)", fontsize=5.4, color=T.ORACLE, va="top")
ax.text(2.38, 0.96, "not aligned", fontsize=5.4, color=T.SUB, va="top", ha="right")
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in BINS], fontsize=5.4, linespacing=1.15)
ax.set_xlim(-0.6, len(BINS) - 0.4); ax.set_ylim(0, 1.0)
ax.set_xlabel("Gemini alignment score (bin)", labelpad=2)
ax.set_ylabel("share of human ratings: object present")
ax.text(0.02, 0.86, "thin ticks = each rater", transform=ax.transAxes, fontsize=5.0,
        color=T.SUB, va="top")
T.clean(ax)

# ---- B: threshold sweep -----------------------------------------------------------
for k, lab, col, ls in [("precision", "precision", T.ORACLE, "-"),
                        ("recall", "recall", T.FREE, "-"), ("f1", "F1", T.INK, (0, (2.5, 1.5)))]:
    bx.fill_between(thr.threshold, thr[f"{k}_lo"], thr[f"{k}_hi"], color=col, alpha=0.12, lw=0)
    bx.plot(thr.threshold, thr[k], color=col, ls=ls, marker="o", ms=2.6, lw=1.1, zorder=3)
    bx.text(thr.threshold.iloc[-1] + 1.5, thr[k].iloc[-1], lab, fontsize=5.4, color=col,
            va="center")
bx.axvline(50, color=T.SUB, lw=0.6, ls=(0, (1, 2)), zorder=1)
bx.text(51.5, 0.62, "paper's\nthreshold", fontsize=5.0, color=T.SUB, va="top", linespacing=1.3)
for t, s_ in zip(thr.threshold, thr.corpus_share):
    bx.text(t, 0.02, f"{100*s_:.1f}%", fontsize=4.6, color=T.SUB, ha="center", va="bottom",
            zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
bx.text(75, 0.085, "% of corpus kept", fontsize=4.6, color=T.SUB, ha="center", va="bottom")
bx.set_xticks(thr.threshold); bx.set_xlim(44, 110); bx.set_ylim(0, 1.0)
bx.set_xlabel("threshold on Gemini score (≥)")
bx.set_ylabel("vs. human consensus")
T.clean(bx)

k = float(hc.loc["kappa_R1_R2", "value"]) if "kappa_R1_R2" in hc.index else float("nan")
print(f"  NOTE figS_gemini_check: yes-rate by bin " +
      ", ".join(f"{b}:{cal.loc[b].yes_rate:.2f}" for b, _ in BINS) +
      f"; consensus at 50 P/R/F1 {thr.precision.iloc[0]:.2f}/{thr.recall.iloc[0]:.2f}/"
      f"{thr.f1.iloc[0]:.2f}, weighted {thr.w_precision.iloc[0]:.2f}/{thr.w_recall.iloc[0]:.2f}/"
      f"{thr.w_f1.iloc[0]:.2f}; F1 peak at {int(thr.threshold[thr.f1.idxmax()])}")
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.16)
T.save(fig, "figS_gemini_check")
