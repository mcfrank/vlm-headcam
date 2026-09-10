"""Compute + cache frozen DINOv2 embeddings for a list of (video_id, frame_idx) frames.

Input : a parquet/csv with columns video_id, frame_idx (the union of all frames any
        training manifest or eval set needs).
Output: EMB_DIR/emb.f16.npy  [N, 768] float16   (row-aligned to index.parquet)
        EMB_DIR/index.parquet  columns video_id, frame_idx, row

Resumable: frames already present in an existing index are skipped and appended to.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

from common import EMB_DIR, DINO_MODEL, EMB_DIM, frame_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="parquet/csv with video_id, frame_idx")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=str(EMB_DIR))
    args = ap.parse_args()

    out = pd.io.common.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    want = (pd.read_parquet(args.frames) if args.frames.endswith(".parquet")
            else pd.read_csv(args.frames))
    want = want[["video_id", "frame_idx"]].drop_duplicates()
    want["frame_idx"] = want["frame_idx"].astype(int)

    idx_path = out / "index.parquet"
    emb_path = out / "emb.f16.npy"
    if idx_path.exists():
        have = pd.read_parquet(idx_path)
        have_keys = set(zip(have.video_id, have.frame_idx))
        want = want[~want.apply(lambda r: (r.video_id, r.frame_idx) in have_keys, axis=1)]
        prev_emb = np.load(emb_path)
        base_row = len(prev_emb)
        print(f"resume: {base_row} cached, {len(want)} new to embed", flush=True)
    else:
        have = None
        prev_emb = None
        base_row = 0
        print(f"fresh: {len(want)} frames to embed", flush=True)

    if len(want) == 0:
        print("nothing to do")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(DINO_MODEL, use_fast=True)
    model = AutoModel.from_pretrained(DINO_MODEL).eval().to(dev)

    rows = want.reset_index(drop=True)
    embs = np.zeros((len(rows), EMB_DIM), dtype=np.float16)
    ok = np.zeros(len(rows), dtype=bool)

    buf_imgs, buf_pos = [], []

    def flush(buf_imgs, buf_pos):
        if not buf_imgs:
            return
        with torch.no_grad():
            inp = proc(images=buf_imgs, return_tensors="pt").to(dev)
            out_ = model(**inp).pooler_output
        e = out_.to(torch.float16).cpu().numpy()
        for j, p in enumerate(buf_pos):
            embs[p] = e[j]
            ok[p] = True

    for i, r in enumerate(rows.itertuples(index=False)):
        fp = frame_path(r.video_id, r.frame_idx)
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            continue
        buf_imgs.append(img)
        buf_pos.append(i)
        if len(buf_imgs) >= args.batch:
            flush(buf_imgs, buf_pos)
            buf_imgs, buf_pos = [], []
        if i % 5000 == 0 and i:
            print(f"  {i}/{len(rows)}", flush=True)
    flush(buf_imgs, buf_pos)

    rows = rows[ok].reset_index(drop=True)
    embs = embs[ok]
    rows["row"] = base_row + np.arange(len(rows))

    if prev_emb is not None:
        all_emb = np.concatenate([prev_emb, embs], axis=0)
        all_idx = pd.concat([have, rows], ignore_index=True)
    else:
        all_emb, all_idx = embs, rows

    np.save(emb_path, all_emb)
    all_idx.to_parquet(idx_path)
    print(f"done: cache now {len(all_emb)} frames ({(~ok).sum()} missing skipped)", flush=True)


if __name__ == "__main__":
    main()
