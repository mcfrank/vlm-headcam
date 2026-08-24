"""Build the 2026.1 (utterance, midpoint-frame) pair manifest from the token-level transcript.

2025.2 came with an utterance-level `full_clip_results.csv`; 2026.1 ships token-level rows, so we
aggregate tokens -> utterances first (text from the `utterance` column, span from min/max token
times), then pair each utterance with the frame at its temporal MIDPOINT — identical convention to
2025.2 (see book/03-pipeline.qmd). No alignment score selects the frame.

Emits
  manifests/bv2026_pairs.parquet   video_id, frame_idx, text, child_id, speaker  (training pairs)
  manifests/bv2026_frames.parquet  the UNIQUE frames those pairs touch (what we embed — far fewer
                                   than the full 1 fps extraction, which we never need in full)

usage: python src/build_pairs_2026.py --frames-root <dir> [--out-prefix manifests/bv2026]
"""
import argparse
import os

import numpy as np
import pandas as pd

TRANSCRIPT = "/ccn2a/dataset/babyview/2026.1/outputs/merged_transcripts_parsed.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default=TRANSCRIPT)
    ap.add_argument("--frames-root", required=True, help="the release's extracted_frames_1fps dir")
    ap.add_argument("--out-prefix", default="manifests/bv2026")
    ap.add_argument("--require-frame", action="store_true",
                    help="drop pairs whose midpoint frame is not on disk (slow; do it once)")
    # The extracted frame set can be a SUPERSET of the release. Membership is authoritative in the
    # Airtable export's `release` column (a comma-separated list per video), so filter on it —
    # otherwise the run silently includes videos that are not part of 2026.1 and is not reproducible.
    ap.add_argument("--videos-csv", default="metadata/videos.csv",
                    help="Airtable export with unique_video_id + release")
    ap.add_argument("--release", default="2026.1", help="keep only videos tagged with this release")
    a = ap.parse_args()

    cols = ["video_id", "utterance_id", "utterance", "token_start_time", "token_end_time", "speaker"]
    df = pd.read_csv(a.transcript, usecols=cols)
    print(f"{len(df):,} token rows | {df.video_id.nunique():,} videos", flush=True)

    g = df.groupby(["video_id", "utterance_id"], sort=False)
    utt = g.agg(text=("utterance", "first"),
                start=("token_start_time", "min"),
                end=("token_end_time", "max"),
                speaker=("speaker", "first")).reset_index()
    utt = utt.dropna(subset=["text", "start", "end"])
    utt["text"] = utt.text.astype(str).str.strip()
    utt = utt[utt.text.str.len() > 0]
    # identical to 2025.2: at 1 fps the frame index IS the midpoint second
    utt["frame_idx"] = ((utt.start + utt.end) / 2).astype(int)
    utt["child_id"] = utt.video_id.str.split("_").str[0]
    print(f"{len(utt):,} utterances | {utt.child_id.nunique()} children (before release filter)", flush=True)

    # ---- release filter (authoritative: Airtable `release`) --------------------------
    if a.release:
        if not os.path.exists(a.videos_csv):
            raise SystemExit(f"release filter needs {a.videos_csv} (Airtable export; human-subjects, "
                             f"kept out of git). Copy it to the node or pass --release ''.")
        v = pd.read_csv(a.videos_csv, low_memory=False)[["unique_video_id", "release"]]
        keep = set(v.loc[v.release.fillna("").str.split(",").apply(
            lambda xs: a.release in [x.strip() for x in xs]), "unique_video_id"])
        # ids look like S00220001_2024-02-05_1_recXXXX with optional trailing decorations
        # (_rotated, _blackout, _processed, and combinations), so pull the rec-token out directly
        # rather than assuming it is the last underscore-separated field.
        rec = utt.video_id.str.extract(r"(rec[A-Za-z0-9]{10,})", expand=False)
        n0, v0 = len(utt), utt.video_id.nunique()
        utt = utt[rec.isin(keep)]
        print(f"release filter {a.release!r}: kept {utt.video_id.nunique():,}/{v0:,} videos, "
              f"{len(utt):,}/{n0:,} utterances", flush=True)
        if utt.video_id.nunique() < 0.5 * v0:
            print("  WARNING: dropped >50% of videos — check the id join before trusting this",
                  flush=True)
    print(f"{len(utt):,} utterances | {utt.child_id.nunique()} children (final)", flush=True)

    if a.require_frame:
        exists = [os.path.exists(f"{a.frames_root}/{v}/{f:05d}.jpg")
                  for v, f in zip(utt.video_id, utt.frame_idx)]
        n0 = len(utt); utt = utt[np.array(exists)]
        print(f"  dropped {n0 - len(utt):,} pairs with no frame on disk", flush=True)

    pairs = utt[["video_id", "utterance_id", "frame_idx", "text", "child_id", "speaker"]].reset_index(drop=True)
    pairs.to_parquet(f"{a.out_prefix}_pairs.parquet", index=False)
    frames = pairs[["video_id", "frame_idx"]].drop_duplicates().reset_index(drop=True)
    frames.to_parquet(f"{a.out_prefix}_frames.parquet", index=False)
    print(f"wrote {len(pairs):,} pairs over {len(frames):,} unique frames -> {a.out_prefix}_*.parquet")
    print(f"  (embedding only these frames, not the full 1 fps extraction)")


if __name__ == "__main__":
    main()
