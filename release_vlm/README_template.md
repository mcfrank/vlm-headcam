---
license: cc-by-nc-4.0
tags: [vision-language, word-learning, cross-situational-learning, developmental, babyview, egocentric, dinov3]
---

*Model card drafted with Claude Opus 5.5.*

# {name} — a cross-situational word learner trained on children's egocentric video ({label})

## Access
These weights are gated: access is limited to researchers who have been granted access to
the BabyView dataset, because BabyView contains egocentric video of children in home and
school environments.

1. Request access to BabyView through Databrary: https://www.databrary.org/volume/1882
2. Once approved, click **Request access** on this page, then email mcfrank@stanford.edu with your
   Hugging Face username and a screenshot showing your BabyView access on Databrary.
3. We will approve your request.

More about the dataset: https://babyview-project.github.io/dataset/

## About
A small contrastive vision–language model that learned word–object mappings from the
**BabyView corpus** (release 2026.1): 1.69M utterance–frame pairs (2,441 hours of
head-mounted-camera video from 48 children, English-language recordings). The image side is
a **frozen** {params}-parameter {regime} (`{encoder}`); this repository contains only the
parts that were learned from BabyView: a projection of each image region into a 512-d space,
a 512-d embedding for each vocabulary word, and a temperature.

Two kinds of model are included: models trained on the **full corpus** (the paper's main
result), and models trained only on the ~10% of pairs that a vision–language model judged
**referentially aligned** — the utterance names an object visible in the frame. The aligned-only
models are the paper's best performers despite using a tenth of the data.

From "Scaling cross-situational word learning on children's naturalistic experiences"
(Frank, Tan, Aw, Sparks, Moreno, Keene-Lee, Ryan, Sepuri, Yang, Marchman, Yamins & Long).
Code: https://github.com/mcfrank/vlm-headcam

## Model
- **Image regions**: the encoder's patch tokens, average-pooled to a 4×4 grid (16 regions of
  dimension {emb_dim}); 1 frame per utterance (the frame at the utterance's temporal midpoint).
- **Text**: bag of words. Lowercase, keep runs of a–z, drop out-of-vocabulary words, keep the
  first 16; the utterance vector is the mean of its word embeddings. Vocabulary = every word
  type occurring at least 5 times in the training corpus.
- **Score**: cosine similarity between the utterance vector and each projected region; the
  image's score is the **maximum over regions** (multiple-instance learning).
- **Training**: symmetric InfoNCE, batch 256, AdamW (lr 3e-4, weight decay 0.1), 20 epochs;
  the epoch was selected on 117 held-out Konkle categories (dev-117) and the model is reported
  on the 60-category Konkle test set.

## Files
- `seed0/` … `seed4/`: five independent training runs (different random seeds) on the full
  corpus.
- `aligned/seed0/` … `aligned/seed2/`: three runs trained only on referentially aligned pairs
  (Gemini 2.5 Flash alignment rating ≥ 50 of 100, judged from the frame and the utterance).
  Their vocabulary is built from that subset, so it is smaller.

Each run directory holds `model.safetensors`, `vocab.json` and `config.json`.
`babyview_vlm.py` is a self-contained loader and scorer.

| run | training pairs | vocabulary | Konkle 4AFC (%) | selected epoch |
|---|---|---|---|---|
{seed_table}

## Usage
```python
import babyview_vlm as bv                       # from this repository
m = bv.load("mcxfrank/{name}", seed=0)          # gated: request access first, then `hf auth login`
# m = bv.load("mcxfrank/{name}", seed=0, aligned=True)   # an aligned-only model
{loader}
S = m.score(["dog", "ball", "spoon"], regions)  # [3 words, n images]; higher = better match
```
The image encoder is **not** included and must be obtained separately: {encoder_link}.
Models were trained on the 16 grid regions; the paper's Konkle numbers were computed with
`include_cls={include_cls}` for this model. Words outside the vocabulary return NaN scores (the
paper scores them at chance).

## Results (mean ± SD over runs; 4AFC, chance 25%)
| model | image encoder | Konkle, full corpus | Konkle, aligned-only | in-domain BabyView frames (full) | LEVANTE vocabulary (full) |
|---|---|---|---|---|---|
{table}

Konkle: 60 object categories, photos from Konkle et al. (2010), following Vong et al. (2024).
In-domain: 603 held-out BabyView frames from videos excluded from training. LEVANTE: 159
child vocabulary items; out-of-vocabulary items scored at chance. Full protocols:
`EVAL.md` in the code repository.

## Data statement
Model weights only. BabyView videos, frames and transcripts are restricted human-subjects data
and are NOT distributed here. Note that the vocabulary (`vocab.json`) and word embeddings are
derived from the BabyView transcripts and contain every word type that occurred at least five
times, which can include names; this is why these models are gated under the same terms as
the dataset. Corpus access is governed by the BabyView data-sharing process (Long et al.,
2025, arXiv:2406.10447).

## License
CC BY-NC 4.0 for the weights in this repository. The image encoder carries its own license
and access terms (see its model page).
