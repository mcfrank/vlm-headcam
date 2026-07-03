"""Embed Konkle object images (explicit paths) into a region-grid cache matching emb_reg's
format (index.parquet video_id/frame_idx/row + emb.f16.npy [N,17,768]), so our trained
region-MIL models can be evaluated on the same Konkle eval Vong uses."""
import math, argparse
import numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
import sys; sys.path.insert(0, "/data2/mcfrank/vlm-headcam/src")
from common import DINO_MODEL, EMB_DIM

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)   # video_id, frame_idx, path
ap.add_argument("--out", required=True)
ap.add_argument("--grid", type=int, default=4); ap.add_argument("--batch", type=int, default=128)
args = ap.parse_args(); G = args.grid
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
man = pd.read_parquet(args.manifest).reset_index(drop=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(DINO_MODEL); model = AutoModel.from_pretrained(DINO_MODEL).eval().to(dev)
R = 1 + G * G; embs = np.zeros((len(man), R, EMB_DIM), np.float16); ok = np.zeros(len(man), bool)
buf, pos = [], []

def flush():
    if not buf: return
    with torch.no_grad():
        h = model(**proc(images=buf, return_tensors="pt").to(dev)).last_hidden_state
    cls = h[:, 0]; patch = h[:, 1:]; P = patch.shape[1]; s = int(round(math.sqrt(P)))
    grid = patch[:, :s*s].transpose(1, 2).reshape(-1, EMB_DIM, s, s)
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (G, G)).flatten(2).transpose(1, 2)
    feat = torch.cat([cls.unsqueeze(1), pooled], 1).to(torch.float16).cpu().numpy()
    for j, p in enumerate(pos): embs[p] = feat[j]; ok[p] = True

for i, r in enumerate(man.itertuples(index=False)):
    try: buf.append(Image.open(r.path).convert("RGB")); pos.append(i)
    except Exception: continue
    if len(buf) >= args.batch: flush(); buf, pos = [], []
flush()
man = man[ok].reset_index(drop=True); embs = embs[ok]; man["row"] = np.arange(len(man))
np.save(out / "emb.f16.npy", embs); man[["video_id", "frame_idx", "row"]].to_parquet(out / "index.parquet")
print(f"embedded {len(man)} konkle images -> {out}")
