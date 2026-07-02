"""Shared paths, tokenization, and embedding-cache helpers for the vlm-headcam prototype."""
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ---- paths on ccn2 ----
BV = Path("/ccn2a/dataset/babyview/2025.2")
FRAMES = BV / "extracted_frames_1fps"
DETS = BV / "outputs/object_detections/cdi"
CLIP_RESULTS = BV / "outputs/full_clip_results.csv"
PARSED = BV / "outputs/merged_transcripts_parsed.csv"

WORK = Path("/data2/mcfrank/vlm-headcam")          # our scratch (node-local NVMe)
EMB_DIR = WORK / "emb"                              # DINOv2 embedding cache
MANIFEST_DIR = WORK / "manifests"

DINO_MODEL = "facebook/dinov2-base"
EMB_DIM = 768

# ---- tokenization ----
_tok_re = re.compile(r"[a-z]+")

def tokenize(text: str):
    return _tok_re.findall(str(text).lower())


def frame_path(video_id: str, frame_idx) -> Path:
    return FRAMES / video_id / f"{int(frame_idx):05d}.jpg"


# ---- embedding cache (single memmap + index) ----
def load_emb_cache(emb_dir: Path = EMB_DIR):
    """Returns (emb float16 array [N,D], lookup dict (video_id, frame_idx)->row)."""
    idx = pd.read_parquet(emb_dir / "index.parquet")
    emb = np.load(emb_dir / "emb.f16.npy", mmap_mode="r")
    lut = {(v, int(f)): int(r) for v, f, r in
           zip(idx.video_id, idx.frame_idx, idx.row)}
    return emb, lut


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
