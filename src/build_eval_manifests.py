"""Build the evaluation image manifests from an asset root (see EVAL.md):

  <root>/konkle/<category>/*.jpg        -> manifests/konkle_manifest.parquet (+ eval_frames_konkle)
  <root>/konkle_dev/<category>/*.jpg    -> manifests/eval_frames_konkle_dev.parquet
  <root>/lev_vocab_items.csv + lev_vocab_images/  -> manifests/lev_vocab_manifest.parquet

Eval images reuse the frame-cache key format: video_id = category (Konkle) or image stem
(LEVANTE), frame_idx = index of the file in sorted order within its category (Konkle) or 0.
`path` is RELATIVE to <root>; pass --root to src/embed_konkle.py when embedding.
The LEVANTE manifest lists exactly the images the item table needs, whatever their extension
(the original manifest globbed *.webp and so missed the two .jpg targets, rubberBand and
turnstile, leaving those items unplayable).

usage: python src/build_eval_manifests.py --root /ccn2b/dataset/babyview/eval_assets
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
a = ap.parse_args()
root = Path(a.root); out = root / "manifests"; out.mkdir(exist_ok=True)


def konkle(sub):
    # Image set = files matching the CASE-SENSITIVE glob `*.jpg` directly in each category
    # folder, in sorted order. This reproduces the manifests every existing cache was built
    # from (verified row-for-row, 2026-09-18). The folders also hold macOS `._*` forks,
    # Thumbs.db, .DS_Store and nested `TestItems/` dirs, all correctly excluded; the one real
    # consequence is that 22 valid dev-117 images with an uppercase `.JPG` extension were never
    # used (dev only; test-60 has none). Kept as-is so keys stay stable.
    rows = []
    for cat in sorted(p.name for p in (root / sub).iterdir() if p.is_dir() and not p.name.startswith(".")):
        for i, f in enumerate(sorted(p.name for p in (root / sub / cat).glob("*.jpg")
                                     if p.is_file() and not p.name.startswith("."))):
            rows.append(dict(video_id=cat, frame_idx=i, category=cat, path=f"{sub}/{cat}/{f}"))
    return pd.DataFrame(rows)


test = konkle("konkle"); dev = konkle("konkle_dev")
test.to_parquet(out / "konkle_manifest.parquet", index=False)
test.drop(columns="path").to_parquet(out / "eval_frames_konkle.parquet", index=False)
dev.to_parquet(out / "eval_frames_konkle_dev.parquet", index=False)

items = pd.read_csv(root / "lev_vocab_items.csv")
files = sorted(set(items[["c0", "c1", "c2", "c3"]].values.ravel()))
missing = [f for f in files if not (root / "lev_vocab_images" / f).exists()]
assert not missing, f"LEVANTE images missing on disk: {missing}"
lev = pd.DataFrame(dict(video_id=[f.rsplit(".", 1)[0] for f in files], frame_idx=0,
                        path=[f"lev_vocab_images/{f}" for f in files]))
assert lev.video_id.is_unique
lev.to_parquet(out / "lev_vocab_manifest.parquet", index=False)
print(f"konkle test {len(test)} images / {test.category.nunique()} categories | dev {len(dev)} / "
      f"{dev.category.nunique()} | levante {len(lev)} images for {len(items)} items -> {out}")
