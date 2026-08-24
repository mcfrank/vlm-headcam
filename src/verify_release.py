"""The standing release check: every layer joins to release_index.tsv at its declared key,
with coverage at or above its documented floor. Run after any migration and before any
experiment bundle. Exit 0 = green; every failure prints, exit 1.

usage: python src/verify_release.py [--root /ccn2b/dataset/babyview/2026.1]
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/ccn2b/dataset/babyview/2026.1")
a = ap.parse_args()
R = Path(a.root)
O = R / "outputs"
fails = []


def check(name, cond, msg):
    ok = bool(cond)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {msg}")
    if not ok:
        fails.append(name)


idx = pd.read_csv(O / "release_index.tsv", sep="\t")
ids = set(idx.video_id)
# post-release exclusions (e.g. registry mislabels awaiting upstream fix): their files may
# still exist in the layers; treat them as intentionally-absent, never as coverage
xp = O / "excluded_post_release.tsv"
excl = set(pd.read_csv(xp, sep="\t").video_id) if xp.exists() else set()
if excl: print(f"post-release exclusions: {len(excl)} videos")
ok_ids = ids | excl
print(f"release_index: {len(idx):,} videos, {idx.subject_id.nunique()} children")
check("index-unique", idx.video_id.is_unique and idx.rec_id.is_unique, "video_id + rec_id unique")

# frames: dirs match the index exactly
fd = {d.name for d in (R / "extracted_frames_1fps").iterdir() if d.is_dir()}
check("frames", (ids <= fd) and (fd <= ok_ids), f"dirs {len(fd):,} vs index {len(ids):,} "
      f"(missing {len(ids - fd)}, unexplained extra {len(fd - ok_ids)})")

# transcripts
t = pd.read_csv(O / "merged_transcripts_parsed.csv",
                usecols=["video_id", "utterance_id"]).drop_duplicates()
tv = set(t.video_id)
check("transcripts-subset", tv <= ok_ids, f"{len(tv):,} videos, {len(tv - ids)} outside the release")
check("transcripts-count", len(t) == 1_838_288, f"{len(t):,} utterances (expect 1,838,288)")

# pose parquet (post-rekey)
pp = O / "pose_1fps_bbox_limbs.parquet"
if pp.exists():
    p = pd.read_parquet(pp, columns=["video_id"])
    pv = set(p.video_id.unique())
    check("pose-subset", pv <= ok_ids, f"{len(pv):,} videos, {len(pv - ids)} outside the release")
    check("pose-coverage", len(pv) >= len(ids) - 15,
          f"{len(pv):,}/{len(ids):,} (floor: all but the ~12 sub-second clips)")
else:
    check("pose-parquet", False, f"{pp} missing (pose_rekey.py not run)")

# referent + pairs
rf = O / "annotations/referent/gemini_2026.1.parquet"
if rf.exists():
    r = pd.read_parquet(rf, columns=["video_id", "alignment"])
    check("referent-subset", set(r.video_id.unique()) <= ok_ids, f"{r.video_id.nunique():,} videos")
    cov = r.alignment.notna().mean()
    check("referent-scored", cov > 0.999, f"{100*cov:.3f}% scored")
else:
    check("referent", False, f"{rf} missing")
pf = O / "annotations/referent/pairs_2026.1.parquet"
if pf.exists():
    pr = pd.read_parquet(pf)
    check("pairs-key", {"video_id", "utterance_id", "frame_idx"} <= set(pr.columns),
          f"{len(pr):,} rows, cols include the utterance key")
else:
    check("pairs", False, f"{pf} missing")

# language: joins to transcripts on the utterance key
lf = O / "annotations/language/lang_2026.1.parquet"
if lf.exists():
    L = pd.read_parquet(lf, columns=["video_id", "utterance_id", "is_english"])
    j = t.merge(L, on=["video_id", "utterance_id"], how="left")
    cov = j.is_english.notna().mean() + j.is_english.isna().mean() * 0  # presence of a row
    got = t.merge(L[["video_id", "utterance_id"]], on=["video_id", "utterance_id"]).shape[0]
    check("language-join", got == len(t), f"{got:,}/{len(t):,} utterances join on the key")
else:
    check("language", False, f"{lf} missing")

# embeddings: index covers the pairs' frames exactly
ei = sorted(glob.glob(str(O / "image_embeddings/dinov3b_grid4x4/index.parquet"))) or \
     sorted(glob.glob(str(O / "image_embeddings/dinov3b_grid4x4/shard_*/index.parquet")))
if ei:
    e = pd.concat([pd.read_parquet(f, columns=["video_id", "frame_idx"]) for f in ei])
    if pf.exists():
        need = pr[["video_id", "frame_idx"]].drop_duplicates()
        got = need.merge(e, on=["video_id", "frame_idx"]).shape[0]
        check("embeddings", got == len(need), f"{got:,}/{len(need):,} pair-frames embedded")
else:
    check("embeddings", False, "no index.parquet found")

print(f"\n{'ALL GREEN' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
