"""C6 — the aligned advantage is not exposure to evaluation nouns, and the aligned 10%
carry essentially all of the signal.

Five arms per encoder, 3 seeds each, all at the same corpus. A = the Gemini-aligned set
(alignment >= 50; 171,782 pairs, 10.2% of the corpus). NB this figure trains on ALL of A,
whereas the aligned scaling arms in fig2B and figS_alignment_encoders take the N
highest-rated pairs -- a subset of the same set (align_170000 is 170,000 of these 171,782,
dropping 1,782 of the 33,993 tied at exactly 50).
  base            the whole corpus
  aligned-only    train on A alone
  plain random    |A| pairs drawn uniformly -- the naive size control
  matched-random  |A| NON-aligned pairs drawn to match A's joint (eval-noun x
                  utterance-length) distribution, so eval-noun exposure is equated
                  (30,205 vs 30,303 pairs; a plain draw of that size has only 12,328)
  full - aligned  the corpus with A removed
  full - matched  the corpus with the matched-unaligned set removed (the removal control)

The two new arms close the loop: matched-unaligned pairs -- a target noun spoken while its
referent is judged absent -- train WORSE than a plain random draw of the same size, and
removing them from the corpus HELPS, while removing the aligned pairs collapses learning.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T, data as D
import matplotlib.pyplot as plt

ENC = [(E["tag"], E["label"], E["color"]) for E in T.ENCODERS]
# Six arms, ordered so the two contrasts read left to right: what the aligned pairs buy on
# their own (bars 2-4, all 172k) and what removing them costs (bars 5-6, all 1.51M).
ARMS = [("base", "whole corpus", T.NEUTRAL),
        ("alignedonly", "aligned (≥ 50)", T.ORACLE),
        ("rand172k", "plain random", "#9c9a92"),
        ("matchrand", "matched random", T.SUB),
        ("minusaligned", "full − aligned", T.INDOM),
        ("minusmatch", "full − matched", T.NEUTRAL)]

fig, axes = plt.subplots(2, 3, figsize=(T.W2, 4.9), sharey=True,
                         gridspec_kw=dict(wspace=0.10, hspace=0.62))
xs = np.arange(len(ARMS))
for a, (enc, key, col) in zip(axes.ravel(), ENC):
    vals = []
    for i, (arm, lab, acol) in enumerate(ARMS):
        f = D.family(f"F_{enc}_{arm if arm != 'base' else 'base'}")
        a.bar(i, f["mean"] - 25, bottom=25, width=0.66, color=acol, zorder=3,
              yerr=f["sd"], error_kw=dict(elinewidth=0.7, capsize=1.8, ecolor=T.INK))
        a.text(i, f["mean"] + f["sd"] + 1.6, f"{f['mean']:.0f}", ha="center", fontsize=5.0,
               color=T.INK)
        vals.append(f["mean"])
    a.axhline(25, color=T.SUB, lw=0.6, ls=(0, (4, 3)), zorder=2)
    a.axhline(vals[0], color=T.SUB, lw=0.5, ls=(0, (1, 2)), zorder=2)
    a.set_xticks(xs)
    a.set_xticklabels([l for _, l, _ in ARMS], fontsize=4.6, rotation=42, ha="right",
                      rotation_mode="anchor")
    a.set_title(key, fontsize=6.5, color=col, pad=3)
    a.set_ylim(22, 95)
    T.clean(a)
    print(f"  NOTE figS_alignment_controls {key}: " +
          "  ".join(f"{lab}={v:.1f}" for (_, lab, _), v in zip(ARMS, vals)))
for a in axes[:, 0]:
    a.set_ylabel("Konkle 4AFC (%)")
axes[1, 0].text(0.03, 0.97, "dotted = whole corpus", transform=axes[1, 0].transAxes,
                fontsize=5.0, color=T.SUB, va="top")
T.save(fig, "figS_alignment_controls")
