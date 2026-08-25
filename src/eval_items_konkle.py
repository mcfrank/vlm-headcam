"""Per-item (per-category) Konkle 4AFC for the full-dataset ladder models, over ALL 177
categories on disk (test-60 + dev-117), for item-level comparison against Wordbank CDI norms.
Not a performance eval — the point is the per-item VARIATION, so the output is one row per
(rung, seed, category) with the category's accuracy and vocab status.

usage: python src/eval_items_konkle.py [--out results/item_eval_b26.csv]
"""
import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
import torch

from train_frame_mil import load_region_cache
from train_region_mil import RegionMIL, eval_4afc_region, encode

ap = argparse.ArgumentParser()
ap.add_argument("--runs-glob", default="runs/B26_lad*,runs/B26_rand_*")
ap.add_argument("--out", default="results/item_eval_b26.csv")
ap.add_argument("--n-trials", type=int, default=200, help="4AFC trials per category (item-level stability)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

SETS = [("test60", "emb_enc_grid_eval/dinov3b_ots_konkle", "manifests/eval_frames_konkle.parquet"),
        ("dev117", "emb_dv3_konkle_dev16", "manifests/eval_frames_konkle_dev.parquet")]
caches = {name: (load_region_cache(c), pd.read_parquet(f)) for name, c, f in SETS}

rows = []
dirs = sorted(sum((glob.glob(g) for g in a.runs_glob.split(",")), []))
for rd in dirs:
    if not Path(rd, "model.pt").exists():
        continue
    m = re.search(r"B26_lad(\d*)_(\w+?)_s(\d+)$", rd)
    if m:
        fam, scale, rung, seed = "lad", m.group(1) or "full", m.group(2), int(m.group(3))
    else:
        m = re.search(r"B26_rand_(\d+)_s(\d+)$", rd)
        if not m: continue
        fam, scale, rung, seed = "rand", m.group(1), "base", int(m.group(2))
    vocab = json.load(open(Path(rd) / "vocab.json"))
    model = RegionMIL(len(vocab)).to(dev)
    model.load_state_dict(torch.load(Path(rd) / "model.pt", map_location=dev))
    for setname, ((emb, lut), ev) in caches.items():
        mean, detail = eval_4afc_region(model, emb, lut, ev, vocab, dev,
                                        n_trials=a.n_trials, return_detail=True)
        per_cat = detail.get("per_cat", {})   # tiny-vocab models can leave <4 scoreable
        if not per_cat:
            print(f"  {rd} {setname}: <4 in-vocab categories, skipped", flush=True)
            continue
        for cat, acc in per_cat.items():
            rows.append(dict(family=fam, rung=rung, scale=scale, seed=seed, set=setname, category=cat,
                             acc=round(100 * acc, 2), in_vocab=bool(encode(cat, vocab, 16))))
    print(f"{fam}/{scale}/{rung}_s{seed}: test {100*sum(v for c,v in per_cat.items())/max(len(per_cat),1):.1f} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
n_iv = df.groupby("rung").in_vocab.mean().round(2).to_dict()
print(f"wrote {a.out}: {len(df):,} rows | {df.category.nunique()} categories x "
      f"{df.rung.nunique()} rungs x {df.seed.nunique()} seeds | in-vocab by rung: {n_iv}")
