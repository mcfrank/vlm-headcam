# results/ — the single source of truth for model numbers

Every number the book or paper reports should be traceable to a row here. Built by scrapers, not
by hand. Regenerate on ccn2 (where the logs live), then copy the parquets here.

| File | What | Built by |
|---|---|---|
| `runs.parquet` | one row per training run: best/final 4AFC, best epoch, n_pairs, store | `src/scrape_runs.py` |
| `evals.parquet` | post-hoc `eval_model.py` Konkle passes recovered from chain-log stdout | `src/scrape_evals.py` |
| `published.csv` | **hand-curated registry** of every claim made in the book, with its rig + expected source | edited by hand |
| `provenance_report.csv` | published vs recovered, with MATCH / MISMATCH / UNRECOVERABLE | `src/check_provenance.py` |
| `lev_vocab_seedmean*.parquet` | LEVANTE-bench per-item results (already clean provenance) | `src/agg_lev_vocab.py` |

```bash
# on ccn2
python src/scrape_runs.py && python src/scrape_evals.py
# locally
scp "ccn2-14:/data2/mcfrank/vlm-headcam/results/*.parquet" results/
.venv/bin/python src/check_provenance.py
```

## Two metric conventions exist — always say which

- **old rig** (`train_region_mil.py`, exploration era): the in-training eval was the CDI-detector
  4AFC; the published number came from a *separate* `eval_model.py … eval_frames_konkle.parquet`
  pass at the end of a chain script, reported at the **final** epoch. Those stdout lines are the
  only record — `scrape_evals.py` recovers them from `logs/*chain*.log`, `logs/scaling_seeds.log`.
- **clean rig** (`train_frame_mil.py`, current): evaluates on Konkle test-60 *during* training and
  reports the **best** epoch. Recorded per-epoch in the run's text log.

The rig offset is ~+2–3 points (best-epoch + eval-during-training vs final-epoch). **Never put an
old-rig and a clean-rig number on the same axis without saying so** — the current ch6 scaling
figure does exactly that (see `notes/PROVENANCE.md`).

## Status (2026-08-21 audit)

38 published claims: **30 MATCH, 1 MISMATCH, 7 UNRECOVERABLE.** The unrecoverable set is the whole
ch4 ladder plus the captioner — their run dirs were deleted in the July home-dir cleanup and only
`DONE` stub logs survive. **They must be re-run before the paper cites them.** The manifests and
embedding caches all still exist, so this is a few GPU-hours, not a redo.
