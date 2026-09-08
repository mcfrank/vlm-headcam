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
ap.add_argument("--runs-glob", default="runs/F_*")
ap.add_argument("--out", default="results/item_eval_final.csv")
ap.add_argument("--n-trials", type=int, default=200, help="4AFC trials per category (item-level stability)")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

EV = pd.read_parquet("manifests/eval_frames_konkle.parquet")
DV = pd.read_parquet("manifests/eval_frames_konkle_dev.parquet")
ENC_SETS = {  # per-encoder (test, dev) caches — item accuracy is meaningless on a foreign cache
    "dinov3b": ("emb_enc_grid_eval/dinov3b_ots_konkle", "emb_dv3_konkle_dev16"),
    "dinov3l": ("emb_ch8_eval/dinov3l_grid4x4_konkle", "emb_ch8_eval/dinov3l_grid4x4_konkle_dev"),
    "vits_bv": ("emb_ch8_eval/vits_bv_konkle", "emb_ch8_eval/vits_bv_konkle_dev"),
    "vitb_bv": ("emb_ch8_eval/vitb_bv_konkle", "emb_ch8_eval/vitb_bv_konkle_dev"),
    "vitl_bv": ("emb_ch8_eval/vitl_bv_konkle", "emb_ch8_eval/vitl_bv_konkle_dev"),
    "dinov3s": ("emb_ch8_eval/dinov3s_konkle", "emb_ch8_eval/dinov3s_konkle_dev"),
}
_cache_memo = {}
def enc_caches(enc):
    if enc not in _cache_memo:
        t, dv = ENC_SETS[enc]
        _cache_memo[enc] = {"test60": (load_region_cache(t), EV), "dev117": (load_region_cache(dv), DV)}
    return _cache_memo[enc]

rows = []
dirs = sorted(sum((glob.glob(g) for g in a.runs_glob.split(",")), []))
for rd in dirs:
    if not Path(rd, "model.pt").exists():
        continue
    m = re.search(r"F_(dinov3l|dinov3b|dinov3s|vits_bv|vitb_bv|vitl_bv)_(.+)_s(\d+)$", rd)
    if not m: continue
    enc, tag, seed = m.group(1), m.group(2), int(m.group(3))
    if tag.startswith("rand_"):
        fam, scale, rung = "rand", tag.split("_")[1], "base"
    elif tag == "base":
        fam, scale, rung = "lad", "full", "base"
    elif tag.startswith("lad_"):
        fam, scale, rung = "lad", "full", tag[4:]
    elif tag.startswith("align_"):
        fam, scale, rung = "align", tag.split("_")[1], "aligned"
    else:
        continue                        # lad@scale/diversity: skip for the item analysis
    vocab = json.load(open(Path(rd) / "vocab.json"))
    caches = enc_caches(enc)
    sd = torch.load(Path(rd) / "model.pt", map_location=dev)
    model = RegionMIL(len(vocab), emb_dim=sd["vproj.2.weight"].shape[1]).to(dev)
    model.load_state_dict(sd)
    for setname, ((emb, lut), ev) in caches.items():
        mean, detail = eval_4afc_region(model, emb, lut, ev, vocab, dev,
                                        n_trials=a.n_trials, return_detail=True)
        per_cat = detail.get("per_cat", {})   # tiny-vocab models can leave <4 scoreable
        if not per_cat:
            print(f"  {rd} {setname}: <4 in-vocab categories, skipped", flush=True)
            continue
        for cat, acc in per_cat.items():
            rows.append(dict(encoder=enc, family=fam, rung=rung, scale=scale, seed=seed, set=setname, category=cat,
                             acc=round(100 * acc, 2), in_vocab=bool(encode(cat, vocab, 16))))
    print(f"{enc}/{fam}/{scale}/{rung}_s{seed}: test {100*sum(v for c,v in per_cat.items())/max(len(per_cat),1):.1f} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
n_iv = df.groupby("rung").in_vocab.mean().round(2).to_dict()
print(f"wrote {a.out}: {len(df):,} rows | {df.category.nunique()} categories x "
      f"{df.rung.nunique()} rungs x {df.seed.nunique()} seeds | in-vocab by rung: {n_iv}")
