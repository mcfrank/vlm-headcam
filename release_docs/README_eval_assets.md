# eval_assets/ — word-learning evaluation images (Konkle 4AFC, LEVANTE vocabulary)

Shared copy of the two out-of-corpus evaluations used in vlm-headcam. **Protocol, reference
numbers and comparability rules: `EVAL.md` in github.com/mcfrank/vlm-headcam.** Nothing here
is human-subjects data. Archived on Oak as
`babyview-2026.1-mirror/project/eval_assets_shared_20260918.tar`.

| path | what |
|---|---|
| `konkle/<category>/*.jpg` | Konkle **test-60** (Vong et al. 2024 test split of the Konkle object set): 60 × 17 = 1,020 photos |
| `konkle_dev/<category>/*.jpg` | Konkle **dev-117**: 1,431 photos used (4–16 per category). For epoch selection / item analyses only |
| `lev_vocab_items.csv` | 159 LEVANTE vocabulary items: `item_uid, target_word, c0..c3, d`; **c0 is the correct image**; `d` = child IRT difficulty, higher = harder |
| `lev_vocab_images/` | stimulus images from `langcog/levante-bench` (item bank 2026-02-22) |
| `manifests/*.parquet` | image lists; `path` is relative to this directory |

The `konkle_dev` folders also contain junk from the original distribution (macOS `._*` files,
`Thumbs.db`, `.DS_Store`, nested `TestItems/`) and 22 images with an uppercase `.JPG`
extension. The manifests use exactly the case-sensitive `*.jpg` files, which is what every
model in the paper was evaluated on; use the manifests, not a directory walk.

Rebuild: `python src/build_lev_vocab.py <dir>` (item table, from a levante-bench checkout) and
`python src/build_eval_manifests.py --root <this dir>` (manifests; reproduces the paper's
row for row). Embed with `python src/embed_konkle.py --manifest <this dir>/manifests/<m>.parquet
--root <this dir> --model <hf id> --grid 4 --out <cache dir>`.
