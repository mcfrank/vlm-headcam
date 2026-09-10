"""Embed DINOv2 features for OBJECT CROPS defined by bbox rows in a manifest.

Input manifest must have: video_id, frame_idx, xmin, ymin, xmax, ymax (+ any passthrough
columns). Each row gets a unique key = "<video>|<frame>|<rowidx>". Writes a crop embedding
cache (<out>/emb.f16.npy + index.parquet with a 'key' column) and rewrites the manifest to
<manifest_stem>_keyed.parquet with the matching 'key' column so train.py can join.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

from common import DINO_MODEL, EMB_DIM, frame_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="crop cache dir")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pad", type=float, default=0.10, help="expand bbox by this frac each side")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(args.manifest).reset_index(drop=True)
    man["key"] = [f"{v}|{int(f)}|{i}" for i, (v, f) in
                  enumerate(zip(man.video_id, man.frame_idx))]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL).eval().to(dev)

    embs = np.zeros((len(man), EMB_DIM), dtype=np.float16)
    ok = np.zeros(len(man), dtype=bool)
    buf, pos = [], []

    def flush():
        if not buf:
            return
        with torch.no_grad():
            inp = proc(images=buf, return_tensors="pt").to(dev)
            e = model(**inp).pooler_output.to(torch.float16).cpu().numpy()
        for j, p in enumerate(pos):
            embs[p] = e[j]; ok[p] = True

    for i, r in enumerate(man.itertuples(index=False)):
        try:
            img = Image.open(frame_path(r.video_id, r.frame_idx)).convert("RGB")
            W, H = img.size
            w, h = r.xmax - r.xmin, r.ymax - r.ymin
            x0 = max(0, r.xmin - args.pad * w); y0 = max(0, r.ymin - args.pad * h)
            x1 = min(W, r.xmax + args.pad * w); y1 = min(H, r.ymax + args.pad * h)
            crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
        except Exception:
            continue
        buf.append(crop); pos.append(i)
        if len(buf) >= args.batch:
            flush(); buf, pos = [], []
        if i % 5000 == 0 and i:
            print(f"  {i}/{len(man)}", flush=True)
    flush()

    man = man[ok].reset_index(drop=True)
    embs = embs[ok]
    idx = pd.DataFrame({"key": man.key, "video_id": man.video_id,
                        "frame_idx": man.frame_idx, "row": np.arange(len(man))})
    np.save(out / "emb.f16.npy", embs)
    idx.to_parquet(out / "index.parquet")
    keyed = Path(args.manifest).with_name(Path(args.manifest).stem + "_keyed.parquet")
    man.to_parquet(keyed)
    print(f"crops embedded: {len(man)} -> {out} | keyed manifest {keyed}")


if __name__ == "__main__":
    main()
