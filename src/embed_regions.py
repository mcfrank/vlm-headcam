"""Embed a coarse REGION GRID per frame from frozen DINOv2 patch tokens (no detector).

For each frame we keep the CLS token plus a GxG grid of region embeddings (patch tokens
adaptive-avg-pooled to GxG). Output cache: emb.f16.npy [N, 1+G*G, 768] (row 0 = CLS),
index.parquet (video_id, frame_idx, row). This lets an utterance match its best region
(MIL) instead of the whole cluttered scene.
"""
import argparse
import math
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

from common import DINO_MODEL, EMB_DIM, frame_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=4, help="GxG region grid")
    ap.add_argument("--model", default=None, help="HF model id (default: common.DINO_MODEL = DINOv2)")
    ap.add_argument("--batch", type=int, default=192)
    # 2025.2's DINOv3 training cache is Khai's grid4x4: 16 cells, NO CLS row. Emit the same
    # convention here so both releases share one readout and one set of eval caches — mismatched
    # readouts are how the 2026-08-22 numbers went wrong.
    ap.add_argument("--drop-cls", action="store_true", help="write R=G*G (no CLS row)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index (parallel embedding)")
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    G = args.grid
    MODEL_ID = args.model or DINO_MODEL
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    want = pd.read_parquet(args.frames)[["video_id", "frame_idx"]].drop_duplicates()
    want["frame_idx"] = want["frame_idx"].astype(int)
    if args.nshards > 1:                      # deterministic split across parallel workers
        want = want.iloc[args.shard::args.nshards].reset_index(drop=True)
        print(f"shard {args.shard}/{args.nshards}: {len(want)} frames", flush=True)
    idx_path = out / "index.parquet"; emb_path = out / "emb.f16.npy"
    if idx_path.exists():
        have = pd.read_parquet(idx_path)
        hk = set(zip(have.video_id, have.frame_idx))
        want = want[~want.apply(lambda r: (r.video_id, r.frame_idx) in hk, axis=1)]
        prev = np.load(emb_path); base = len(prev)
        print(f"resume: {base} cached, {len(want)} new", flush=True)
    else:
        have = None; prev = None; base = 0
        print(f"fresh: {len(want)} frames", flush=True)
    if len(want) == 0:
        print("nothing to do"); return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(dev)
    NREG = getattr(model.config, 'num_register_tokens', 0) or 0   # DINOv3: 4, DINOv2: 0
    DIM = getattr(model.config, 'hidden_size', EMB_DIM)
    print(f'model {MODEL_ID} | dim {DIM} | register tokens {NREG}', flush=True)

    rows = want.reset_index(drop=True)
    R = (G * G) if args.drop_cls else (1 + G * G)
    embs = np.zeros((len(rows), R, DIM), dtype=np.float16)
    ok = np.zeros(len(rows), dtype=bool)
    buf, pos = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            inp = proc(images=buf, return_tensors="pt").to(dev)
            h = model(**inp).last_hidden_state          # [B, 1+P, 768]
        cls = h[:, 0]                                    # [B, 768]
        patch = h[:, 1 + NREG:]                                 # [B, P, 768]
        P = patch.shape[1]; s = int(round(math.sqrt(P)))
        assert s * s == P, f"patch count {P} not a perfect square (NREG={NREG} wrong?)"
        grid = patch[:, :s * s].transpose(1, 2).reshape(-1, DIM, s, s)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (G, G))  # [B,768,G,G]
        pooled = pooled.flatten(2).transpose(1, 2)       # [B, G*G, 768]
        feat = pooled if args.drop_cls else torch.cat([cls.unsqueeze(1), pooled], 1)  # [B, 1+G*G, 768]
        f = feat.to(torch.float16).cpu().numpy()
        for j, p in enumerate(pos):
            embs[p] = f[j]; ok[p] = True

    for i, r in enumerate(rows.itertuples(index=False)):
        try:
            img = Image.open(frame_path(r.video_id, r.frame_idx)).convert("RGB")
        except Exception:
            continue
        buf.append(img); pos.append(i)
        if len(buf) >= args.batch:
            flush(); buf, pos = [], []
        if i % 5000 == 0 and i:
            print(f"  {i}/{len(rows)}", flush=True)
    flush()

    rows = rows[ok].reset_index(drop=True); embs = embs[ok]
    rows["row"] = base + np.arange(len(rows))
    if prev is not None:
        embs = np.concatenate([prev, embs], 0); rows = pd.concat([have, rows], ignore_index=True)
    np.save(emb_path, embs); rows.to_parquet(idx_path)
    print(f"done: {len(embs)} frames, shape {embs.shape}", flush=True)


if __name__ == "__main__":
    main()
