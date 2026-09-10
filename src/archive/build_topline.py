"""Alignment 'topline': inject correct word-object alignment onto ORDINARY (random,
not alignment-selected) frames by appending the frame's detected object labels to the
utterance. Same-frames contrast isolates alignment from frame quality.

Takes random utterances (min-clip 0) whose midpoint frame lies in a detection-video and
has >=1 confident CDI detection. On that IDENTICAL frame set, emits three manifests that
differ only in text:
  topline_control.parquet   text = utterance only
  topline_label.parquet     text = utterance + detected labels   (injected alignment)
  topline_labelonly.parquet text = detected labels only          (pure ceiling)
Frames are already in emb_full, so no new embedding is needed.
"""
import argparse
import os
import pandas as pd
from pathlib import Path
from common import CLIP_RESULTS, DETS, MANIFEST_DIR, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="S00360001")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-pix", type=int, default=2000)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    det_videos = set(os.listdir(DETS))
    cr = pd.read_csv(CLIP_RESULTS, usecols=[
        "child_id", "video_name", "utterance",
        "utterance_start_time", "utterance_end_time"]).dropna()
    cr = cr[cr.child_id.astype(str) != args.held]
    cr = cr[cr.video_name.isin(det_videos)]
    cr["frame_idx"] = ((cr.utterance_start_time + cr.utterance_end_time) / 2).astype(int)

    rows = []  # (video_id, frame_idx, utterance, labels_str)
    for vid, g in cr.groupby("video_name"):
        csv = DETS / vid / "bounding_box_predictions.csv"
        if not csv.exists():
            continue
        det = pd.read_csv(csv, usecols=["frame_number", "confidence", "class_name",
                                        "masked_pixel_count"]).dropna()
        det = det[(det.confidence >= args.conf) & (det.masked_pixel_count >= args.min_pix)]
        if det.empty:
            continue
        by_frame = {f: sorted(set(gg.class_name.str.lower().str.strip()))
                    for f, gg in det.groupby("frame_number")}
        for r in g.itertuples(index=False):
            labs = by_frame.get(r.frame_idx)
            if not labs:
                continue
            rows.append((vid, int(r.frame_idx), str(r.utterance).strip(), " ".join(labs)))

    df = pd.DataFrame(rows, columns=["video_id", "frame_idx", "utterance", "labels"])
    df = df[df.utterance.map(lambda t: len(tokenize(t)) >= 1)]
    if args.max_pairs and len(df) > args.max_pairs:
        df = df.sample(args.max_pairs, random_state=args.seed)
    print(f"topline-eligible pairs (random utt, detection frame, excl {args.held}): {len(df)}")

    out = Path(MANIFEST_DIR)
    df.assign(text=df.utterance)[["video_id", "frame_idx", "text"]] \
        .to_parquet(out / "topline_control.parquet")
    df.assign(text=df.utterance + " " + df.labels)[["video_id", "frame_idx", "text"]] \
        .to_parquet(out / "topline_label.parquet")
    df.assign(text=df.labels)[["video_id", "frame_idx", "text"]] \
        .to_parquet(out / "topline_labelonly.parquet")
    print("wrote topline_{control,label,labelonly}.parquet")


if __name__ == "__main__":
    main()
