# Provenance: what feeds what, and where the stack disagrees with itself

Audit of 2026-08-21, at the exploration → paper transition. Companion to `notes/PHASES.md`
(what each era means) and `results/README.md` (how the tables are built).

## The chain, as it should be

```
run logs on ccn2 ──scrape_runs.py──▶ results/runs.parquet  ──┐
chain-log stdout ──scrape_evals.py─▶ results/evals.parquet ──┼─▶ figures/*.py ─▶ figures/out/*.pdf
                                     results/published.csv ──┘        (paper display items)
                                              │
                                    check_provenance.py ─▶ results/provenance_report.csv
```

## The chain, as it actually is (book era)

Most book figures **hardcode** numbers that were read off a log by hand and typed into a script.
That is the root cause of every divergence below.

| Figure | Script | Numbers come from | Reads data? |
|---|---|---|---|
| fig_waterfall_ladder / _label / _mechanisms | `make_ladder_figs.py` | hardcoded | ✗ |
| fig_scaling_curves / fig_whose_data | `make_scaling_figs.py` | hardcoded | ✗ |
| fig_dev_time | `make_dev_figure.py` | hardcoded per-seed lists | ✗ |
| fig_encoder_compare / _ladder_slope | `make_encoder_ladder_fig.py` | hardcoded | ✗ |
| fig_architectures | `make_arch_fig.py` | hardcoded | ✗ |
| fig_item_acc | `make_item_plot.py` | run + eval caches | ✓ |
| fig_item_analysis, fig_confusion(+mds), fig_cues | `make_item_analysis.py`, `make_confusion.py`, then `plot_item_confusion_cues.py` | cached parquet/npz in `book/figdata/` | ✓ |
| fig_lev_vocab, fig_levante_vocab.svg | `make_lev_vocab_fig.py` | per-seed parquet | ✓ |
| fig_data_descriptives | `data_stats.py` | corpus + Gemini parquet | ✓ |
| fig_clip_arch.svg, fig_konkle_4afc.svg | hand-authored SVG | n/a (schematics) | n/a |

## DIVERGENCES — fix before the paper

**D1. The ch6 scaling curve mixes two rigs on one axis.** Its 10k–300k points are *old rig*
(`train_region_mil`, post-hoc `eval_model.py`, **final** epoch); its 911k point is *clean rig*
(`train_frame_mil --window 0`, **best** epoch). The rig offset is ~+2–3 points, so the curve's
top end is lifted by a methods change rather than by data. Same for the aligned arm (all old rig)
being compared against a clean-rig endpoint. **This inflates the headline "still rising at 911k".**
→ Fix: re-run the whole curve on the clean rig (cheap: the manifests exist), or plot the old-rig
911k point (62.6) and state the rig.

**D2. Two figures in the same chapter disagree about the same point.** For random-911k,
`make_scaling_figs.py` has **65.3** while `make_dev_figure.py` has per-seed `[63.7, 63.5, 60.7]`
= **62.6**. The 62.6 values are the recovered old-rig truth (`logs/grid_mil.log`); the 65.3 is
clean-rig. The developmental-time figure is therefore built on stale numbers relative to its
neighbour.

**D3. Hand-transcription drift in the scaling figure.** `make_scaling_figs.py` random arm reads
29.3 / 38.8 / 39.7 / 56.3 where both the book table *and* the recovered logs say
**30.0 / 38.3 / 40.2 / 56.2**. Nobody's conclusion changes, but no number in the figure is exactly
the number in the text.

**D4. best-epoch vs final-epoch is not stated anywhere.** The clean rig reports best-epoch; the old
rig reported final-epoch. The one MISMATCH in the audit is this: ch6's full-corpus **65.6** is
final-epoch-like, while the recovered best-epoch mean is **67.1**. Pick one convention, state it in
Methods, and apply it everywhere. (Best-epoch on a test set is itself a mild optimism — worth a
sentence, or switch to a held-out dev split for selection.)

**D5. The same configuration is published twice with different values.** ch4's "+ region MIL"
(**65.3**) and ch9's "hard max" (**66.1**) are both window-0 region-MIL on 911k with frozen DINOv2.
Recovered `max` = **66.4**. These should be one number.

**D6. The ch4 ladder has no surviving provenance.** `ladder_pure / region / filter / word / vision`
(61.3 / 65.3 / 69.9 / 74.0 / 81.5) — the paper's central display item — plus the ch9 captioner
(62.1) come from runs deleted in the July home-dir cleanup; only `DONE` stubs remain. They are
**UNRECOVERABLE** and must be re-run before the paper cites them. Note the DINOv2 rows of the ch8
encoder table are the same lost runs.

**D7. Two scripts own the same output.** `fig_confusion.png` / `fig_item_analysis.png` are written
both by the ccn2 scripts (`make_confusion.py`, `make_item_analysis.py`) and by the local
`plot_item_confusion_cues.py`. Whichever ran last wins. → Split cleanly: ccn2 scripts emit **data**
to `book/figdata/`, local scripts emit **figures**.

**D8. Figure outputs land in two places.** Some scripts write `/data2/mcfrank/vlm-headcam/book_figs/`,
others write `book/figures/` directly; the copy step is manual and undocumented.

## Eval sets in play (say which, always)

- **Konkle test-60** — the headline 4AFC. Vong's 60-category test split, out-of-corpus photos.
- **Konkle dev-117** — the larger dev split; used for the item analysis (176–177 categories = test+dev).
- **CDI-detector 4AFC** — Phase 1–2 only. Noisy, child-specific, ~24 points lower. Never mix with Konkle.
- **LEVANTE-bench vocab** — external benchmark vs children (159 items, 100 in vocab). Clean provenance
  already (per-seed parquets) — the model the rest should follow.
