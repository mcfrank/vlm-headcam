"""C6 — the aligned advantage is not exposure to evaluation nouns, and the aligned 10%
carry essentially all of the signal.

Five arms per encoder, 3 seeds each, all at the same corpus. A = the Gemini-aligned set
(alignment >= 50; 171,782 pairs, 10.2% of the corpus). NB this figure trains on ALL of A,
whereas the aligned scaling arms in fig2B and figS_alignment_encoders take the N
highest-rated pairs -- a subset of the same set (align_170000 is 170,000 of these 171,782,
dropping 1,782 of the 33,993 tied at exactly 50).
  base            the whole corpus
  aligned-only    train on A alone
  matched-random  |A| pairs drawn to match A's joint (eval-noun x utterance-length)
                  distribution, so eval-noun exposure is equated (31,047 vs 30,946 pairs;
                  a plain random draw of that size has only 12,906)
  full - aligned  the corpus with A removed
  full - random   the corpus with a random |A| removed (the removal control)
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

ENC = [("dinov3l", "L-OTS", T.OTHER), ("dinov3b", "B-OTS", T.FREE),
       ("vitb_bv", "B-BV", T.INDOM), ("vits_bv", "S-BV", T.INDOM2)]
ARMS = [("base", "whole corpus", T.NEUTRAL),
        ("alignedonly", "aligned (≥ 50)", T.ORACLE),
        ("matchrand", "matched random", T.SUB),
        ("minusaligned", "full − aligned", T.INDOM),
        ("minusrand", "full − random", T.NEUTRAL)]

fig, axes = plt.subplots(1, 4, figsize=(T.W2, 2.7), sharey=True,
                         gridspec_kw=dict(wspace=0.12))
xs = np.arange(len(ARMS))
for a, (enc, key, col) in zip(axes, ENC):
    vals = []
    for i, (arm, lab, acol) in enumerate(ARMS):
        f = D.family(f"F_{enc}_{arm if arm != 'base' else 'base'}")
        a.bar(i, f["mean"] - 25, bottom=25, width=0.66, color=acol, zorder=3,
              yerr=f["sd"], error_kw=dict(elinewidth=0.7, capsize=1.8, ecolor=T.INK))
        a.text(i, f["mean"] + f["sd"] + 1.6, f"{f['mean']:.0f}", ha="center", fontsize=5.4,
               color=T.INK)
        vals.append(f["mean"])
    a.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=2)
    a.axhline(vals[0], color=T.SUB, lw=0.5, ls=(0, (1, 2)), zorder=2)
    a.set_xticks(xs)
    a.set_xticklabels([l for _, l, _ in ARMS], fontsize=5.0, rotation=42, ha="right",
                      rotation_mode="anchor")
    a.set_title(key, fontsize=6.5, color=col, pad=3)
    a.set_ylim(22, 95)
    T.clean(a)
    print(f"  NOTE figS_alignment_controls {key}: " +
          "  ".join(f"{lab}={v:.1f}" for (_, lab, _), v in zip(ARMS, vals)))
axes[0].set_ylabel("Konkle 4AFC (%)")
axes[0].text(0.03, 0.97, "dotted = whole corpus", transform=axes[0].transAxes,
             fontsize=5.0, color=T.SUB, va="top")
T.save(fig, "figS_alignment_controls")
