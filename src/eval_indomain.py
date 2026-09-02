"""IN-DOMAIN evaluations (SI): on the held-out BabyView eval set (language-excluded videos),
(a) word-learning 4AFC for every final scaling model, per encoder — the domain-transfer
control; (b) head-free prototype 4AFC per encoder on BOTH the in-domain set and Konkle —
the encoder-quality-by-domain scatter. Frames come from the existing region caches.
usage: python src/eval_indomain.py"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "src")
from common import frame_key
from train_frame_mil import load_region_cache
from train_region_mil import RegionMIL, eval_4afc_region, encode

dev = "cuda" if torch.cuda.is_available() else "cpu"
EMB = "/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings"
ENC = {
    "dinov3b": ([f"{EMB}/dinov3b_grid4x4"], "emb_enc_grid_eval/dinov3b_ots_konkle"),
    "dinov3l": (sorted(glob.glob(f"{EMB}/dinov3l_grid4x4/shard_*")), "emb_ch8_eval/dinov3l_grid4x4_konkle"),
    "vits_bv": (sorted(glob.glob(f"{EMB}/vits_bv_grid4x4/shard_*")), "emb_ch8_eval/vits_bv_konkle"),
    "vitb_bv": (sorted(glob.glob(f"{EMB}/vitb_bv_grid4x4/shard_*")), "emb_ch8_eval/vitb_bv_konkle"),
}
ev = pd.read_parquet("manifests/eval_frames_indomain.parquet")
kv = pd.read_parquet("manifests/konkle_manifest.parquet")


def frame_cache(dirs):
    embs, lut = [], {}
    for ci, d in enumerate(dirs):
        e = np.load(f"{d}/emb.f16.npy", mmap_mode="r")
        ix = pd.read_parquet(f"{d}/index.parquet")
        base = sum(x.shape[0] for x in embs)
        for v, f, r in zip(ix.video_id, ix.frame_idx, range(len(ix))):
            lut[frame_key(v, int(f))] = base + r
        embs.append(e)
    if len(embs) == 1:
        return embs[0], lut
    # unify via concatenated view: materialize only the eval rows we need later
    return embs, lut


def rows_for(emb, lut, df):
    """Materialize [n, R, D] float32 for the eval frames only (handles shard lists)."""
    keys = [frame_key(v, int(f)) for v, f in zip(df.video_id, df.frame_idx)]
    if isinstance(emb, list):
        sizes = np.cumsum([0] + [e.shape[0] for e in emb])
        out = np.zeros((len(keys), emb[0].shape[1], emb[0].shape[2]), dtype=np.float32)
        ok = np.ones(len(keys), bool)
        for i, k in enumerate(keys):
            r = lut.get(k)
            if r is None: ok[i] = False; continue
            si = np.searchsorted(sizes, r, "right") - 1
            out[i] = emb[si][r - sizes[si]].astype(np.float32)
        return out, ok
    idxs = [lut.get(k) for k in keys]
    ok = np.array([i is not None for i in idxs])
    out = np.stack([emb[i].astype(np.float32) if i is not None else
                    np.zeros(emb.shape[1:], np.float32) for i in idxs])
    return out, ok


def proto4afc(X, cats, n_trials=200, seed=0):
    x = torch.from_numpy(X.mean(1)); x = F.normalize(x, dim=-1)
    rng = np.random.default_rng(seed)
    bycat = {}
    for i, c in enumerate(cats): bycat.setdefault(c, []).append(i)
    bycat = {c: v for c, v in bycat.items() if len(v) >= 2}
    ok = tot = 0
    for c, idx in bycat.items():
        others = [k for k in bycat if k != c]
        if len(others) < 3: continue
        for i in idx:
            proto_c = F.normalize(x[[j for j in idx if j != i]].mean(0), dim=-1)
            foils = rng.choice(others, 3, replace=False)
            pf = torch.stack([proto_c] + [F.normalize(x[bycat[f]].mean(0), dim=-1) for f in foils])
            ok += int((x[i] @ pf.T).argmax().item() == 0); tot += 1
    return 100 * ok / max(tot, 1)


probe_rows, wl_rows = [], []
for enc, (dirs, kcache) in ENC.items():
    emb, lut = frame_cache(dirs)
    X, okm = rows_for(emb, lut, ev)
    evx = ev[okm].reset_index(drop=True); Xx = X[okm]
    print(f"{enc}: in-domain frames found {okm.sum()}/{len(ev)}", flush=True)
    probe_rows.append(dict(encoder=enc, domain="indomain", proto=round(proto4afc(Xx, list(evx.category)), 1)))
    ke, klut = load_region_cache(kcache)
    KX, kok = rows_for(ke, klut, kv)
    probe_rows.append(dict(encoder=enc, domain="konkle", proto=round(proto4afc(KX[kok], list(kv[kok].category)), 1)))
    print(f"  probes done: {probe_rows[-2]} {probe_rows[-1]}", flush=True)

    # word-learning 4AFC over the scaling models
    ilut = {frame_key(v, int(f)): i for i, (v, f) in enumerate(zip(evx.video_id, evx.frame_idx))}
    for rd in sorted(glob.glob(f"runs/F_{enc}_rand_*") + glob.glob(f"runs/F_{enc}_base_s*")):
        m = re.search(rf"F_{enc}_(rand_(\d+)|base)_s(\d+)$", rd)
        if not m or not Path(rd, "model.pt").exists(): continue
        N = int(m.group(2)) if m.group(2) else 1686105
        vocab = json.load(open(Path(rd) / "vocab.json"))
        sd = torch.load(Path(rd) / "model.pt", map_location=dev)
        model = RegionMIL(len(vocab), emb_dim=sd["vproj.2.weight"].shape[1]).to(dev)
        model.load_state_dict(sd)
        mean, detail = eval_4afc_region(model, Xx, ilut, evx, vocab, dev,
                                        n_trials=100, return_detail=True)
        wl_rows.append(dict(encoder=enc, N=N, seed=int(m.group(3)), acc=round(100 * mean, 2),
                            n_cats=detail["n_total"], n_oov=detail["n_oov"]))
    print(f"  word-learning evals done for {enc}", flush=True)

pd.DataFrame(probe_rows).to_csv("results/encoder_probe_domains.csv", index=False)
pd.DataFrame(wl_rows).to_csv("results/indomain_eval.csv", index=False)
print("wrote results/encoder_probe_domains.csv + results/indomain_eval.csv")
