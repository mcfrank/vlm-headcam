"""Speech density in the 2026.1 release — is the utterance rate plausible and stable?

A: utterances per hour per video. An outlier here is usually a transcription failure
   rather than a quiet child.
B: the same rate aggregated per child, against that child's median age, which is the check
   that the rate is not drifting with development in a way the corpus-level constant hides.
The corpus-level rate this figure summarises is what the developmental-time figure (main
Fig. 4) uses to convert training pairs into years (results/utterance_rate.csv). Ported from supplement.qmd
(fig-density); source is the committed release diagnostics.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import theme as T
import pandas as pd
import matplotlib.pyplot as plt

R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
D = __import__("pathlib").Path(__file__).resolve().parent.parent / "diagnostics" / "2026.1"
V = pd.read_parquet(D / "video_level.parquet")
V = V[(V.hours > 0) & V.n_utterances.notna()].copy()
V["utt_per_hr"] = V.n_utterances / V.hours
rate = pd.read_csv(R / "utterance_rate.csv").iloc[0]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(T.W2, 2.4), gridspec_kw=dict(wspace=0.26))
ax.hist(V.utt_per_hr.clip(0, 3000), bins=50, color=T.ORACLE, zorder=3)
ax.axvline(rate.utt_per_hr, color=T.INK, lw=0.8, zorder=4)
ax.text(rate.utt_per_hr * 1.08, ax.get_ylim()[1] * 0.92,
        f"corpus rate {rate.utt_per_hr:.0f}/h\n(used in Fig. 4)", fontsize=5.2, color=T.INK,
        va="top", linespacing=1.4)
ax.set_xlabel("utterances / hour (per video)"); ax.set_ylabel("videos")
T.clean(ax)

m = V.groupby("child").apply(lambda d: d.n_utterances.sum() / d.hours.sum(),
                             include_groups=False)
a_ = V.groupby("child").age_months.median()
bx.scatter(a_, m, s=13, color=T.ORACLE, alpha=0.85, zorder=3)
bx.axhline(rate.utt_per_hr, color=T.INK, lw=0.8, zorder=2)
bx.axhspan(rate.boot_lo, rate.boot_hi, color=T.INK, alpha=0.10, lw=0, zorder=1)
bx.set_xlabel("child's median age (months)"); bx.set_ylabel("utterances / hour (per child)")
T.clean(bx)
print(f"  NOTE figS_speech_density: per-video median {V.utt_per_hr.median():.0f}/h; "
      f"per-child median {m.median():.0f}/h (range {m.min():.0f}-{m.max():.0f}); "
      f"corpus rate {rate.utt_per_hr:.0f} [{rate.boot_lo:.0f}-{rate.boot_hi:.0f}]")
for a, l in zip((ax, bx), "AB"):
    T.panel(a, l, dx=-0.14, dy=1.09)
T.save(fig, "figS_speech_density")
