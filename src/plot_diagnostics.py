"""Render the 2026.1 release-diagnostics figures for the book (static PNGs; the book has
execute disabled). Reads the committed aggregates bundle only — no cluster access, no frames.
usage: python src/plot_diagnostics.py [bundle=diagnostics/2026.1] [out=book/figures]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[1] if len(sys.argv) > 1 else "diagnostics/2026.1")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "book/figures")
V = pd.read_parquet(D / "video_level.parquet")
C = pd.read_parquet(D / "child_level.parquet")
prov = json.loads((D / "provenance.json").read_text())
for df in (V, C):        # older bundles carry the survey as a 0-1 proportion
    if df.survey_pct_english.max() <= 1.5:
        df["survey_pct_english"] *= 100

BLUE, GREEN, YELLOW, RED, PURPLE = "#4477AA", "#228833", "#CCBB44", "#EE6677", "#AA3377"
INK, SUB, GRID = "#2c2c2a", "#6b6a66", "#e1e0d9"


def clean(ax, gy=True):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=SUB, length=0); ax.set_axisbelow(True)
    if gy: ax.grid(axis="y", color=GRID, lw=0.6)


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT/name}")


# 1. layer coverage
LAYERS = [("frames / pairs", "n_pairs"), ("transcripts", "n_utterances"), ("pose", "pose_persons"),
          ("referent (Gemini)", "ref_scored"), ("language", "lang_utterances"), ("embeddings", "emb_frames")]
lab, val = zip(*[(n, 100 * (V[c] > 0).mean()) for n, c in LAYERS if c in V.columns])
fig, ax = plt.subplots(figsize=(8.5, 3.4))
ax.barh(range(len(lab)), val, color=[GREEN if v > 99 else (YELLOW if v > 90 else RED) for v in val])
for i, v in enumerate(val): ax.text(min(v + 1, 100.5), i, f"{v:.1f}%", va="center", fontsize=9, color=SUB)
ax.axvline(100, color=SUB, lw=1, ls=(0, (4, 3)))
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab); ax.set_xlim(0, 110)
ax.set_xlabel("% of release videos covered"); ax.invert_yaxis(); clean(ax, gy=False)
ax.grid(axis="x", color=GRID, lw=0.6)
save(fig, "diag_coverage.png")

# 2. per-child hours
c = C.sort_values("hours", ascending=False)
fig, ax = plt.subplots(figsize=(9.5, 4))
ax.bar(range(len(c)), c.hours, color=BLUE)
ax.set_xticks(range(len(c))); ax.set_xticklabels(c.child, rotation=90, fontsize=5.5)
ax.set_ylabel("hours recorded"); clean(ax)
save(fig, "diag_child_hours.png")

# 3. age: hours histogram + longitudinal spaghetti
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].hist(V.age_months, bins=40, weights=V.hours, color=GREEN)
ax[0].set_xlabel("age at recording (months)"); ax[0].set_ylabel("hours"); clean(ax[0])
for _, g in V.sort_values("age_months").groupby("child"):
    ax[1].plot(g.age_months, g.hours.cumsum(), lw=0.9, alpha=0.75, color=BLUE)
ax[1].set_xlabel("age (months)"); ax[1].set_ylabel("cumulative hours (per child)"); clean(ax[1])
save(fig, "diag_age.png")

# 4. speech density
V2 = V[V.hours >= 2 / 60].copy(); V2["uph"] = V2.n_utterances / V2.hours   # rates: videos >= 2 min
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].hist(V2.uph.clip(0, 3000), bins=50, color=PURPLE)
ax[0].set_xlabel("utterances / hour (per video)"); ax[0].set_ylabel("videos"); clean(ax[0])
m = V2.groupby("child").apply(lambda d: d.n_utterances.sum() / d.hours.sum(), include_groups=False)
ax[1].scatter(V2.groupby("child").age_months.median(), m, s=30, color=PURPLE, alpha=0.85)
ax[1].set_xlabel("child median age (months)"); ax[1].set_ylabel("utterances / hour"); clean(ax[1])
save(fig, "diag_density.png")

# 5. survey vs measured English
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.plot([0, 100], [0, 100], color=SUB, lw=1, ls=(0, (4, 3)))
ax.scatter(C.survey_pct_english, C.measured_pct_english, s=44, color=RED, alpha=0.85)
r = C[["survey_pct_english", "measured_pct_english"]].corr().iloc[0, 1]
for x in C[C.survey_pct_english < 80].itertuples():        # the informative points
    ax.annotate(x.child, (x.survey_pct_english, x.measured_pct_english),
                xytext=(5, -4), textcoords="offset points", fontsize=6.5, color=SUB)
ax.set_xlabel("survey % English (household report)")
ax.set_ylabel("measured % English (labelled utterances)")
ax.set_title(f"r = {r:.2f}", loc="left", fontsize=11, color=INK)
clean(ax)
save(fig, "diag_english.png")

# 6. distributions
def load(n):
    return pd.read_parquet(D / n) if (D / n).exists() else None
ppf = load("dist_persons_per_frame.parquet")
if ppf is not None and 0 not in set(ppf.persons_in_frame) and "pose_frames" in V.columns:
    zero = int(V.pose_frames.sum() - V.pose_frames_with_person.sum())
    ppf = pd.concat([pd.DataFrame({"persons_in_frame": [0], "count": [zero]}), ppf], ignore_index=True)
panels = [(load("dist_alignment.parquet"), "alignment_bin", "Gemini alignment", GREEN),
          (ppf, "persons_in_frame", "persons per frame", BLUE),
          (load("dist_utterance_len.parquet"), "n_words", "words per utterance", YELLOW)]
panels = [p for p in panels if p[0] is not None]
fig, axs = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.2))
for a_, (df, xc, t, col) in zip(np.atleast_1d(axs), panels):
    a_.bar(df[xc], df["count"], color=col, width=(df[xc].diff().median() or 1) * 0.85)
    a_.set_xlabel(t); a_.set_ylabel("count"); clean(a_)
save(fig, "diag_dists.png")
print(f"release {prov['release']}: {prov['n_videos']:,} videos, {prov['n_children']} children")


# 7. video length distribution
qc = pd.read_parquet(D / "video_qc.parquet") if (D / "video_qc.parquet").exists() else None
if qc is not None:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(qc.dur_s / 60, bins=60, color=BLUE)
    ax.axvline(1, color=RED, lw=1, ls=(0, (4, 3)))
    n1 = (qc.dur_s < 60).sum()
    ax.text(1.2, ax.get_ylim()[1] * 0.9, f"<1 min: {n1} videos\n(0.13% of utterances)", fontsize=8.5, color=RED)
    ax.set_xlabel("video length (minutes)"); ax.set_ylabel("videos"); clean(ax)
    save(fig, "diag_video_len.png")

# 8. metrics over age: dense video-level scatter + per-child binned spaghetti
V3 = V[(V.hours >= 2 / 60) & (V.n_utterances > 0)].copy()
V3["uph"] = V3.n_utterances / V3.hours
V3["mlu_w"] = V3.n_words / V3.n_utterances          # mean utterance length (words), per video
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, col, ylab, ylim in [(axs[0], "uph", "utterances / hour", (0, 2200)),
                            (axs[1], "mlu_w", "mean utterance length (words)", (0, 10))]:
    ax.scatter(V3.age_months, V3[col], s=4, color=BLUE, alpha=0.12, lw=0, zorder=2)
    for _, g in V3.groupby("child"):                 # per-child trajectory, smoothed in 3-mo bins
        if len(g) < 8: continue
        b = g.groupby((g.age_months // 3) * 3)[col].median()
        if len(b) >= 3: ax.plot(b.index + 1.5, b.values, lw=1.0, alpha=0.55, color=PURPLE, zorder=3)
    med = V3.groupby((V3.age_months // 3) * 3)[col].median()
    ax.plot(med.index + 1.5, med.values, lw=2.6, color=INK, zorder=4)
    ax.set_xlabel("age at recording (months)"); ax.set_ylabel(ylab); ax.set_ylim(*ylim); clean(ax)
save(fig, "diag_age_trends.png")
