"""Embed frames with DINOv3-B/16 at MULTIPLE transformer layers (supplement experiment S1).

Motivation: for ViTs trained with self-distillation, intermediate layers often linear-probe better
than the final layer — the last block specializes toward the pretraining objective. Khai's cached
DINOv3 readouts are final-layer only (CLS + meanpatch), so testing this needs its own extraction.

Cheap by design: ONE forward pass per image yields every layer (output_hidden_states=True), and we
store only the mean-pooled patch vector per layer for the sweep (100k frames x 12 layers x 768 x
fp16 = 1.8 GB). Once a layer wins, re-extract the 4x4 grid at that layer alone for the real runs.

usage (on ccn2, a GPU free):
  python src/embed_dinov3_layers.py --frames manifests/sweep_frames_100k.parquet \
      --out emb_dv3_layers --layers 2,4,6,8,10,12 [--grid 4]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import frame_path

MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="parquet with video_id, frame_idx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="2,4,6,8,10,12", help="1-based block outputs; 12 = final")
    ap.add_argument("--grid", type=int, default=0, help="also store a GxG grid (0 = meanpatch only)")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval().half()

    NREG = getattr(model.config, "num_register_tokens", 0) or 0   # DINOv3: 4
    print(f"register tokens: {NREG}", flush=True)
    layers = [int(x) for x in a.layers.split(",")]
    fr = pd.read_parquet(a.frames)[["video_id", "frame_idx"]].drop_duplicates().reset_index(drop=True)
    R = 1 + (a.grid * a.grid if a.grid else 0)
    D = model.config.hidden_size
    print(f"{len(fr)} frames x {len(layers)} layers x {R} regions x {D}d", flush=True)

    store = {L: np.zeros((len(fr), R, D), np.float16) for L in layers}
    with torch.no_grad():
        for s in range(0, len(fr), a.batch):
            chunk = fr.iloc[s:s + a.batch]
            ims = []
            for r in chunk.itertuples():
                try:
                    ims.append(Image.open(frame_path(r.video_id, r.frame_idx)).convert("RGB"))
                except Exception as e:
                    # a black image would be written as a real embedding — refuse instead
                    raise SystemExit(f"unreadable frame {r.video_id}/{r.frame_idx}: {e}")
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev).half()
            out = model(pixel_values=px, output_hidden_states=True)
            for L in layers:
                h = out.hidden_states[L]                    # [B, 1+P, D]
                patch = h[:, 1 + NREG:, :]
                vecs = [patch.mean(1, keepdim=True)]        # meanpatch
                if a.grid:
                    B, P, Dd = patch.shape
                    g = int(P ** 0.5)
                    assert g * g == P, f"patch count {P} not square (NREG={NREG}?)"
                    gp = patch[:, :g * g, :].transpose(1, 2).reshape(B, Dd, g, g)
                    gp = torch.nn.functional.adaptive_avg_pool2d(gp, (a.grid, a.grid))
                    vecs.append(gp.flatten(2).transpose(1, 2))
                store[L][s:s + len(chunk)] = torch.cat(vecs, 1).float().cpu().numpy().astype(np.float16)
            if s % (a.batch * 100) == 0:
                print(f"  {s}/{len(fr)}", flush=True)

    idx = fr.assign(row=np.arange(len(fr)))
    for L in layers:
        d = f"{a.out}/L{L:02d}"
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/emb.f16.npy", store[L])
        idx.to_parquet(f"{d}/index.parquet", index=False)
        print(f"  wrote {d}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
