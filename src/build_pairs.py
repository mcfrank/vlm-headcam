"""Build a frame-utterance training manifest from full_clip_results.csv, with the
alignment filter as the central knob.

Each kept utterance is paired with the middle 1fps frame of its time window (CVCL pairs
an utterance with a frame sampled from its window; middle is a cheap deterministic proxy).

Output: MANIFEST_DIR/<name>.parquet  columns video_id, frame_idx, text, clip_score_max, child_id
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from common import CLIP_RESULTS, MANIFEST_DIR, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="manifest name (no extension)")
    ap.add_argument("--min-clip", type=float, default=0.0, help="keep clip_score_max >= this")
    ap.add_argument("--exclude-videos", default=None, help="text file of held-out video_ids to drop")
    ap.add_argument("--exclude-children", default=None, help="comma-sep child_ids to drop (held out)")
    ap.add_argument("--only-children", default=None, help="comma-sep child_ids to KEEP (within-kid pools)")
    ap.add_argument("--max-pairs", type=int, default=0, help="cap N pairs (0 = no cap)")
    ap.add_argument("--sample", choices=["top", "random"], default="random",
                    help="if capping: take top-scoring or a random sample")
    ap.add_argument("--min-tokens", type=int, default=1, help="require >= this many alpha tokens")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(CLIP_RESULTS, usecols=[
        "child_id", "video_name", "utterance", "utterance_start_time",
        "utterance_end_time", "clip_score_max"])
    df = df.rename(columns={"video_name": "video_id", "utterance": "text"})
    df = df.dropna(subset=["clip_score_max", "utterance_start_time", "utterance_end_time"])

    df = df[df.clip_score_max >= args.min_clip]
    df = df[df.text.map(lambda t: len(tokenize(t)) >= args.min_tokens)]

    if args.exclude_videos:
        drop = {v.strip() for v in open(args.exclude_videos) if v.strip()}
        df = df[~df.video_id.isin(drop)]
    if args.exclude_children:
        drop = {c.strip() for c in args.exclude_children.split(",")}
        df = df[~df.child_id.astype(str).isin(drop)]
    if args.only_children:
        keep = {c.strip() for c in args.only_children.split(",")}
        df = df[df.child_id.astype(str).isin(keep)]

    df["frame_idx"] = ((df.utterance_start_time + df.utterance_end_time) / 2).astype(int)

    if args.max_pairs and len(df) > args.max_pairs:
        if args.sample == "top":
            df = df.nlargest(args.max_pairs, "clip_score_max")
        else:
            df = df.sample(args.max_pairs, random_state=args.seed)

    out = df[["video_id", "frame_idx", "text", "clip_score_max", "child_id"]].reset_index(drop=True)
    Path(MANIFEST_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(MANIFEST_DIR) / f"{args.name}.parquet"
    out.to_parquet(path)
    print(f"{args.name}: {len(out)} pairs | clip_max>={args.min_clip} | "
          f"children={out.child_id.nunique()} videos={out.video_id.nunique()}")
    print(f"  clip_score_max: mean={out.clip_score_max.mean():.3f} "
          f"p50={out.clip_score_max.median():.3f} max={out.clip_score_max.max():.3f}")


if __name__ == "__main__":
    main()
