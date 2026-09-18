# EVAL — the two word-learning evaluations, and how to run them the way the paper did

Two out-of-corpus 4AFC evaluations score every model in the paper: **Konkle** object photos
(the headline number) and the **LEVANTE** picture-vocabulary items (the comparison with
measured child performance). This file says what the assets are, what exactly "a trial" is,
how this project runs them, and what you must hold constant to get numbers comparable to ours.

Both evaluations ask the same question of a model: *given a word and four images, which image
gets the highest word–image score?* Anything that can produce `score(word, image)` can be
evaluated. Our models are frozen-encoder two-towers (`RegionMIL`), so the scripts here assume
that format; §5 says what to match if your model is something else.

## 1. Assets — `/ccn2b/dataset/babyview/eval_assets/` (shared; not human-subjects data)

All paths below are relative to that root (its README: `release_docs/README_eval_assets.md`).
Archived on Oak as `babyview-2026.1-mirror/project/eval_assets_shared_20260918.tar`.

| asset | where | what |
|---|---|---|
| Konkle **test-60** images | `konkle/<category>/*.jpg` | 60 categories × 17 photos = 1,020; Vong et al.'s (2024) test split of the Konkle object set |
| Konkle **dev-117** images | `konkle_dev/<category>/*.jpg` | 117 other categories, 1,431 photos (4–16 per category, median 15); used ONLY for epoch selection and item analyses |
| Konkle manifests | `manifests/konkle_manifest.parquet` (test), `manifests/eval_frames_konkle.parquet` (same rows, no `path`), `manifests/eval_frames_konkle_dev.parquet` | columns `video_id, frame_idx, category[, path]`; `video_id`/`frame_idx` are pseudo-keys so eval images share the frame-cache format; `path` is relative to the root |
| LEVANTE item table | `lev_vocab_items.csv` | 159 items: `item_uid, target_word, c0, c1, c2, c3, d`. **`c0` is always the correct image**; `d` is the child IRT item difficulty, **higher = harder** (lowest: lollipop, carrot; highest: turnstile, aesthete) |
| LEVANTE images | `lev_vocab_images/` + `manifests/lev_vocab_manifest.parquet` (629 images = exactly those the items use) | from the tracked item bank of `langcog/levante-bench` (`data/assets/2026-02-22/corpus/vocab/`) |
| child comparison tables | in git: `results/wordbank_anchors_40.csv`, `results/levante_child_by_age.csv` | Wordbank CDI trajectories on the 40 CDI-matched Konkle words; LEVANTE children's IRT-imputed accuracy by age (`figures/make_wordbank_anchors.R`, `figures/make_wordbank_40.py`, `figures/make_levante_ages.py`) |

**Use the manifests, not a directory walk.** The `konkle_dev` folders carry junk from the
original distribution (macOS `._*` files, `Thumbs.db`, nested `TestItems/`) and 22 valid
images with an uppercase `.JPG` extension that the paper never used; the manifests list
exactly the paper's images. They are rebuilt, row for row, by
`src/build_eval_manifests.py --root <root>`; the item table by `src/build_lev_vocab.py`.
The Konkle images are the public Konkle et al. object set; the split is Vong et al.'s.

## 2. Konkle 4AFC — the exact protocol

- **Word** = the category label (e.g. `airplane`), passed through the training tokenizer
  (lowercase, `[a-z]+`; a multi-word label is the mean of its word embeddings).
- **Trial**: one target image drawn at random from the category's images; three foil
  *categories* drawn without replacement from the other categories, one random image from
  each. The model scores the word against the four images; correct iff the target scores
  highest.
- **100 trials per category**, `numpy.random.default_rng(0)`, categories visited in sorted
  order. (Item-level analyses use 200.)
- **Accuracy = macro mean over categories** of per-category accuracy, ×100.
- **Out-of-vocabulary words score chance (25%)**, they are not dropped. Dropping them makes a
  small-vocabulary model look better on an easier subset (we measured corr = −0.94 between
  accuracy and categories scored before fixing this). In our implementation an OOV category is
  also left out of the foil pool, so trial draws are identical across models only when all 60
  labels are in vocabulary — true for every model trained on ≥300k pairs (mean OOV: 49 of 60
  at 3k pairs, 27 at 10k, 10 at 30k, 1 at 100k, 0 from 300k).
- **Selection**: train 20 epochs, evaluate dev-117 and test-60 after every epoch, **select the
  epoch on dev-117, report test-60 at that epoch** (`reported_test_acc` in `metrics.json`).
  Never select on test-60.
- Evaluate with dropout off (`model.eval()`); we once reported numbers ~1.4 points low
  because it was on.
- Code: `eval_4afc_region()` in `src/train_region_mil.py`; called every epoch by
  `src/train_frame_mil.py`; item-level table by `src/eval_items_konkle.py`.

Reference values (test-60, full corpus, mean ± sd over 5 seeds): OTS-304M 81.6 ± 0.6, OTS-86M
78.7 ± 0.9, OTS-22M 67.0 ± 0.7, BV-304M 44.2 ± 1.6, BV-86M 44.1 ± 1.2, BV-22M 39.6 ± 1.5
(`results/encoder_grid.csv`). Chance is 25.

## 3. LEVANTE vocabulary 4AFC — the exact protocol

- 159 fixed items; each is one trial: target word vs the four listed images, `c0` correct.
  No sampling, so there is no seed.
- An item is **playable** for a model if the target word is in its vocabulary (and all four
  images are embedded). We report two numbers and you should too:
  - **fair accuracy**: mean over all 159 items with unplayable items credited 0.25 (this is
    what Fig. 4B plots and what is comparable across models);
  - playable-only accuracy **with the number of playable items**.
- **Child-likeness**: Spearman correlation, over playable items, between item accuracy
  (averaged over seeds) and child difficulty `d`. Negative = the model finds hard what
  children find hard.
- The method is DevBench-style contrastive forced choice. The LEVANTE-bench paper itself
  evaluates generative VLMs by prompting; that harness is not what we use.
- Children: `results/levante_child_by_age.csv` gives accuracy by age imputed over ALL items
  from the fitted IRT model (the task is adaptive, so raw proportion correct is not
  comparable across ages). NB that script uses the fitted model's *intercept* (higher =
  easier); the `d` column in `lev_vocab_items.csv` is a difficulty (higher = harder).
- Code: `src/eval_lev_scaling.py` → `results/lev_scaling_final.csv` (one row per model × item).

Reference values (full corpus, fair accuracy): OTS-304M 47.7, OTS-86M 40.4, OTS-22M 36.6,
BV-304M 31.9, BV-86M 30.7, BV-22M 31.1.

## 4. How this project runs them (RegionMIL-format models)

Our score is `max over regions` of the cosine between a projected region vector and the
mean-of-word-embeddings text vector; eval images are embedded as **17 regions: CLS + a 4×4
grid** of average-pooled patch tokens.

```bash
# on ccn2-14, from the repo root; PY=/ccn2/u/khaiaw/miniconda3/envs/ccwm/bin/python
R=/ccn2b/dataset/babyview/eval_assets
# 1. embed the eval images with YOUR frozen encoder (HF id or local path)
$PY -B src/embed_konkle.py --root $R --manifest $R/manifests/konkle_manifest.parquet        --model <hf_id> --grid 4 --out emb_ch8_eval/<enc>_konkle
$PY -B src/embed_konkle.py --root $R --manifest $R/manifests/eval_frames_konkle_dev.parquet --model <hf_id> --grid 4 --out emb_ch8_eval/<enc>_konkle_dev
$PY -B src/embed_konkle.py --root $R --manifest $R/manifests/lev_vocab_manifest.parquet     --model <hf_id> --grid 4 --out emb_lev_<enc>
# 2. train; Konkle dev/test are scored every epoch and the dev-selected test score is recorded
$PY -B src/train_frame_mil.py --window 0 --manifest manifests/<pairs>.parquet --caches <train cache dirs> \
    --eval-cache emb_ch8_eval/<enc>_konkle --eval-frames manifests/eval_frames_konkle.parquet \
    --dev-cache emb_ch8_eval/<enc>_konkle_dev --dev-frames manifests/eval_frames_konkle_dev.parquet \
    --seed 0 --out runs/F_<enc>_<condition>_s0          # writes model.pt, vocab.json, metrics.json
# 3. LEVANTE and item-level Konkle over saved checkpoints
$PY -B src/eval_lev_scaling.py --out results/lev_scaling_<name>.csv
$PY -B src/eval_items_konkle.py --runs-glob "runs/F_<enc>_*" --out results/item_eval_<name>.csv
```

Caveats of the current scripts, if you reuse them as-is:
- `eval_lev_scaling.py`, `eval_items_konkle.py` and `eval_indomain.py` carry a **hard-coded
  encoder registry** (tag → cache path, and a run-name regex `F_<tag>_<cond>_s<seed>`); a new
  encoder needs an entry in each.
- Non-HF encoders (e.g. the BabyView-trained DINOv3s) are embedded with the DINO fork's
  `embed_native_dino.py`, which writes the same cache format.
- Per-encoder caches are never interchangeable; scoring a model on another encoder's cache
  gives chance.

## 5. Evaluating a model that is not a RegionMIL two-tower

To be comparable with the paper, hold these constant and change only `score(word, image)`:
the asset files of §1; the trial construction, seed, trial counts and macro-averaging of §2;
the item list and fair-accuracy rule of §3; chance credit for words your model cannot
represent; dev-117 (never test-60) for any selection; and, for the developmental comparison,
the 40-word Konkle subset in `results/wordbank_anchors_40.csv`.

Sanity check your implementation by reproducing one of ours first: load any
`runs/F_dinov3l_base_s*/` checkpoint, score it through your trial loop, and confirm you get
its `reported_test_acc` (≈ 81.6) to within trial-sampling noise (exactly, if you reuse
`eval_4afc_region`).

## 6. Gotchas we hit

- Selecting the epoch on test-60 inflates scores by 1–2 points; don't.
- Dropping OOV categories instead of crediting chance inflates small models (see §2).
- **Two LEVANTE items were unscorable in the paper's results.** The original image manifest
  globbed `*.webp`, so the only two `.jpg` targets (`rubber band`, `turnstile`) were never
  embedded and both items count as unplayable (chance) for every model in
  `lev_scaling_final.csv`. The shared manifest includes them (found 2026-09-18). A new model
  evaluated on the shared manifest can score up to 2 more items than ours did; for an exact
  comparison, treat `vocab__rubberband` and `vocab__turnstile` as unplayable.
- Native (non-HF) encoders are embedded with the DINO fork's `embed_native_dino.py`, which
  has no `--root`: run it from `/ccn2b/dataset/babyview/eval_assets` so the relative paths
  resolve.
- `lev_scaling_final.csv` also contains legacy 2025.2-corpus rows labelled
  `L-OTS (2025.2)` / `L-BV (preview, 2025.2)`; filter by encoder label.
- The OTS-86M Konkle cache has 16 regions (grid only), the others 17 (CLS + grid); the
  prototype probe drops CLS for all so encoders are comparable.
- The training tokenizer splits contractions (`don't` → `don`, `t`) and has no lemmatizer:
  `dogs` is not `dog`. Category labels are singular, so plural-only exposure does not count.
