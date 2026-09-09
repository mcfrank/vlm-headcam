"""LEVANTE-vocab 4AFC as a SECOND scaling-measured outcome (the Konkle pattern): evaluate
every saved scaling-run model on the LEVANTE picture-vocab items, per item, and join the
child IRT difficulty d from the levante-bench data. Output supports (a) a model scaling
curve on LEVANTE and (b) Spearman(model item accuracy, child item difficulty) BY SCALE —
does the model's item profile become more child-like as data grow?

Encoders map to their own lev image caches: B26_* -> emb_lev_dinov3b,
C8_dinov3l_* -> emb_lev_dinov3l, C8_dinov3l_bv_* -> emb_lev_dinov3l_bv.

usage: python src/eval_lev_scaling.py [--out results/lev_scaling.csv]
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_frame_mil import load_region_cache
from train_region_mil import RegionMIL, encode
from common import frame_key

ap = argparse.ArgumentParser()
ap.add_argument("--items", default="lev_vocab_items.csv")
ap.add_argument("--out", default="results/lev_scaling_final.csv")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

items = pd.read_csv(a.items)
CACHES = {k: load_region_cache(f"emb_lev_{k}") for k in
          ["dinov3b", "dinov3l", "dinov3l_bv", "vits_bv", "vitb_bv", "vitl_bv", "dinov3s"]}

RUNS = []
LBL = {"dinov3l": "L-OTS", "dinov3b": "B-OTS", "dinov3s": "S-OTS", "vits_bv": "S-BV", "vitb_bv": "B-BV", "vitl_bv": "L-BV"}
for rd in sorted(glob.glob("runs/F_*")):
    m = re.search(r"F_(dinov3l|dinov3b|dinov3s|vits_bv|vitb_bv|vitl_bv)_(rand_(\d+)|base)_s(\d+)$", rd)
    if not m or not Path(rd, "model.pt").exists():
        continue
    N = int(m.group(3)) if m.group(3) else 1686105
    RUNS.append((rd, LBL[m.group(1)], m.group(1), N, int(m.group(4))))
for rd in sorted(glob.glob("runs/__none__")):
    m = re.search(r"B26_rand_(\d+)_s(\d+)$", rd) or re.search(r"B26_lad_base_s(\d+)$", rd)
    if not m or not Path(rd, "model.pt").exists():
        continue
    N, s = (int(m.group(1)), int(m.group(2))) if m.re.pattern.startswith("B26_rand") else (1820000, int(m.group(1)))
    RUNS.append((rd, "B-OTS", "dinov3b", N, s))
for rd in sorted(glob.glob("runs/C8_dinov3l*")):
    m = re.search(r"C8_(dinov3l(?:_bv)?)_grid4x4_(rand_(\d+)|base)_s(\d+)$", rd)
    if not m or not Path(rd, "model.pt").exists():
        continue
    # legacy 2025.2-corpus runs: label them DISTINCTLY. Plain "L-OTS"/"L-BV" collided with the
    # final F_ runs at every shared (N, seed) and with the new vitl_bv encoder ("L-BV").
    enc = "L-BV (preview, 2025.2)" if "_bv" in m.group(1) else "L-OTS (2025.2)"
    N = int(m.group(3)) if m.group(3) else 1820000
    RUNS.append((rd, enc, m.group(1).replace("dinov3l", "dinov3l") if "_bv" in m.group(1) else "dinov3l", N, int(m.group(4))))
    RUNS[-1] = (rd, enc, "dinov3l_bv" if "_bv" in m.group(1) else "dinov3l", N, int(m.group(4)))
print(f"{len(RUNS)} models to evaluate")

rows = []
for rd, encname, ckey, N, seed in RUNS:
    emb, lut = CACHES[ckey]
    vocab = json.load(open(Path(rd) / "vocab.json"))
    sd = torch.load(Path(rd) / "model.pt", map_location=dev)
    emb_dim = sd["vproj.2.weight"].shape[1]          # 768 (ViT-B) or 1024 (ViT-L)
    model = RegionMIL(len(vocab), emb_dim=emb_dim).to(dev)
    model.load_state_dict(sd); model.eval()

    def regions(fname):
        key = frame_key(str(fname).rsplit(".", 1)[0], 0)
        return lut.get(key)

    with torch.no_grad():
        for it in items.itertuples():
            toks = encode(str(it.target_word), vocab, 16)
            cand = [regions(c) for c in (it.c0, it.c1, it.c2, it.c3)]
            if not toks or any(c is None for c in cand):
                rows.append(dict(encoder=encname, N=N, seed=seed, item=it.item_uid,
                                 word=it.target_word, d=it.d, playable=False, correct=np.nan))
                continue
            V = torch.from_numpy(np.asarray(emb[cand], dtype=np.float32)).to(dev)
            R = model.enc_regions(V)
            t = torch.zeros(1, 16, dtype=torch.long, device=dev)
            t[0, :len(toks)] = torch.tensor(toks, device=dev)
            tv = model.enc_text(t, torch.tensor([len(toks)], device=dev))
            sc = torch.einsum("brd,md->brm", R, tv).max(1).values.squeeze(-1)
            rows.append(dict(encoder=encname, N=N, seed=seed, item=it.item_uid,
                             word=it.target_word, d=it.d, playable=True,
                             correct=int(sc.argmax().item() == 0)))
    pl = [r for r in rows if r["encoder"] == encname and r["N"] == N and r["seed"] == seed and r["playable"]]
    acc = 100 * np.mean([r["correct"] for r in pl]) if pl else float("nan")
    print(f"  {encname} N={N:>9,} s{seed}: {acc:5.1f} on {len(pl)}/{len(items)} playable", flush=True)

df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(df):,} rows")
