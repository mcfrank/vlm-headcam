# vlm-headcam

Learning word–object mappings from **BabyView** child egocentric video: a frozen visual
encoder, a bag-of-words text tower, and contrastive region-MIL training on 1.69M
utterance–frame pairs from 48 children. The paper asks what limits learning from this data
(scaling, encoder pretraining regime and size, referential alignment) and compares the models
with children on Konkle object images and the LEVANTE vocabulary items.

## Layout (paper code = `src/` + `runners/` + `figures/`; everything else is archive or notes)

| Path | What |
|---|---|
| [`src/`](src/) | the pipeline behind every paper number: release consolidation, pairs, embeddings, manifests, training (`train_frame_mil.py`), evaluations, number/table generators, lexicon analyses. Map from each result to its script: [`notes/PAPER_REPRO.md`](notes/PAPER_REPRO.md) |
| [`src/archive/`](src/archive/) | legacy modules (2025.2 book phases, excluded cue analyses, superseded filters) — not used by the paper; [README](src/archive/README.md) |
| [`runners/`](runners/) | the shell drivers that produced the runs on ccn2-14 (resumable); legacy drivers in `runners/archive/`; [README](runners/README.md) |
| [`results/`](results/) | **single source of truth for numbers**: `runs.parquet` (every run), per-analysis tables, `encoder_grid.csv`, `experiments_table.tex`, `methods_numbers.tex`; [README](results/README.md) lists which files the paper reads |
| [`figures/`](figures/) | paper display items, one script each, reading only `results/` (`make -C figures`); encoders enumerated from `theme.ENCODERS` |
| [`human_check/`](human_check/) | the lab rating app for validating the Gemini alignment annotation (frames never leave the cluster) |
| [`diagnostics/2026.1/`](diagnostics/2026.1/) | committed corpus diagnostics (per-video / per-child tables) behind the SI corpus figures |
| [`release_docs/`](release_docs/) | READMEs for the derived layers we added to the BabyView 2026.1 release tree |
| [`notes/`](notes/) | [experiments.md](notes/experiments.md) (journal) · [CONTROLS_TABLE](notes/CONTROLS_TABLE.md) · [PAPER_REPRO](notes/PAPER_REPRO.md) · [CLEANUP_PLAN](notes/CLEANUP_PLAN.md) · [STORAGE](notes/STORAGE.md) · [DATA_LAYOUT](notes/DATA_LAYOUT.md) · [MIGRATION](notes/MIGRATION.md) · `sessions/` (cross-session handoffs) |
| [`archive/`](archive/) | the deprecated Quarto book (2025.2 corpus), the drafting supplement, cue-era eval scripts; [README](archive/README.md) |

Annotation pipelines (pose, Gemini alignment, language ID) live in the companion repo
`bv-annotations`; the BabyView-trained DINOv3 encoders come from the DINO-retraining fork
(cited by commit hash in the paper).

## Start here

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make -C figures                              # rebuild every display item from results/
.venv/bin/python src/make_encoder_grid.py    # the 3×2 encoder table
.venv/bin/python src/make_experiments_table.py
```

Training and embedding run on the cluster (`runners/`, `notes/STORAGE.md`); results are
scraped into `results/runs.parquet` with `src/scrape_runs.py` and committed.

## Data governance

BabyView is human-subjects data. Frames, audio and transcripts stay on the cluster (and the
Stanford Oak mirror); only aggregate results are committed. Example-frame figures use
face-blurred frames (`src/blur_faces.py`); the rating app serves frames from the cluster only.
