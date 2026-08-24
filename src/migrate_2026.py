"""2026.1 consolidation: execute M1/M3/M4 copies, M6 quarantine, M8 release_index.tsv, and
MANIFEST.tsv, per notes/MIGRATION.md. Copy -> md5 verify -> breadcrumb; SOURCES ARE NOT
DELETED here (retirement happens only after the Oak mirror verifies).

Dry-run by default; pass --execute to act. Every action lands in migration_log.tsv.
M2 (mp3) = make_mp3_2026.py; M5 (pose) = pose_rekey.py; M7 (embeddings) = merge_emb_shards.py.

usage: python src/migrate_2026.py [--execute]
"""
import argparse
import hashlib
import re
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

R = Path("/ccn2b/dataset/babyview/2026.1")
Q = Path("/ccn2b/dataset/babyview/_mcfrank_not_in_2026.1")
COPIES = [  # (label, src, dst)
    ("M1 transcripts", "/ccn2a/dataset/babyview/2026.1/outputs/merged_transcripts_parsed.csv",
     R / "outputs/merged_transcripts_parsed.csv"),
    ("M3 language", "/ccn2/dataset/babyview/annotations/language",
     R / "outputs/annotations/language"),
    ("M4 referent", "/data2/mcfrank/vlm-headcam/scored/bv2026_gemini.parquet",
     R / "outputs/annotations/referent/gemini_2026.1.parquet"),
    ("M4 pairs", "/data2/mcfrank/vlm-headcam/manifests/bv2026_pairs.parquet",
     R / "outputs/annotations/referent/pairs_2026.1.parquet"),
    ("M4 frames-manifest", "/data2/mcfrank/vlm-headcam/manifests/bv2026_frames.parquet",
     R / "outputs/annotations/referent/frames_2026.1.parquet"),
]
EXPECT = {"M3 language": ("lang_2026.1.parquet", 1_838_288),   # completeness gates
          "M4 pairs": (None, 1_838_134)}
POSE_EXTRA = ["S00360002_2026-05-19_4_recNStFG30FOUiDQV",       # M6: post-release recordings
              "S00510002_2026-04-30_2_rec7AmHPtur4LP42M",
              "S00510002_2026-05-01_6_recJ3aUz3GKXd1z20"]


def md5(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--execute", action="store_true")
a = ap.parse_args()
log = []


def do_copy(label, src, dst):
    src, dst = Path(src), Path(dst)
    files = sorted(src.rglob("*")) if src.is_dir() else [src]
    files = [f for f in files if f.is_file()]
    print(f"{label}: {len(files)} file(s), {sum(f.stat().st_size for f in files)/1e9:.2f} G -> {dst}")
    if label in EXPECT:                    # refuse to migrate a half-finished layer
        fn, n = EXPECT[label]
        check = (src / fn) if fn else src
        got = len(pd.read_parquet(check))
        assert got == n, f"{label}: {check} has {got:,} rows, expected {n:,} — layer incomplete?"
        print(f"  completeness ok ({got:,} rows)")
    if not a.execute:
        return
    for f in files:
        rel = f.relative_to(src) if src.is_dir() else Path(dst.name)
        out = (dst / rel) if src.is_dir() else dst
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        m1, m2 = md5(f), md5(out)
        assert m1 == m2, f"MD5 MISMATCH {f} -> {out}"
        log.append((label, str(f), str(out), m1))
    crumb = (src if src.is_dir() else src.parent) / "MOVED.txt"
    crumb.write_text(f"{date.today()}: migrated to {dst}\nsee {R}/outputs/migration_log.tsv\n"
                     f"source retained until the Oak mirror is verified; do not add new files here.\n")
    print(f"  copied + verified; breadcrumb at {crumb}")


for label, src, dst in COPIES:
    do_copy(label, src, dst)

# ---- M6: quarantine post-release pose dirs --------------------------------------------------
for n in POSE_EXTRA:
    src = R / "outputs/pose_1fps" / n
    print(f"M6 quarantine: {src} -> {Q/'pose_1fps'/n}  (exists: {src.exists()})")
    if a.execute and src.exists():
        (Q / "pose_1fps").mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(Q / "pose_1fps" / n))
        log.append(("M6", str(src), str(Q / "pose_1fps" / n), ""))

# ---- M8: release_index.tsv — nobody regexes ids again ---------------------------------------
at = pd.read_csv(R / "outputs/videos_airtable_2026-07-24.csv", low_memory=False)
at = at[at.release.astype(str).str.contains("2026.1", na=False)]
ids = (R / "outputs/release_2026_1_ids.txt").read_text().split()
rec = {i: m.group(1) for i in ids if (m := re.search(r"(rec[A-Za-z0-9]{14,})", i))}
assert len(rec) == len(ids), "release id without a rec-id"
xw = at.set_index(at.unique_video_id.astype(str)).subject_id.astype(str).to_dict()
idx = pd.DataFrame({"video_id": ids,
                    "rec_id": [rec[i] for i in ids],
                    "subject_id": [xw.get(rec[i], "") for i in ids],
                    "rotated": [i.endswith("_rotated") for i in ids]})
assert (idx.subject_id != "").all(), "release id whose rec-id is not in the registry"
assert idx.rec_id.is_unique, "duplicate rec-id in the release list"
print(f"M8 release_index: {len(idx):,} rows, {idx.subject_id.nunique()} children")
if a.execute:
    idx.to_csv(R / "outputs/release_index.tsv", sep="\t", index=False)

# ---- MANIFEST + log --------------------------------------------------------------------------
if a.execute:
    pd.DataFrame(log, columns=["step", "src", "dst", "md5"]).to_csv(
        R / "outputs/migration_log.tsv", sep="\t", index=False)
    rows = []
    for f in sorted((R / "outputs").rglob("*")):
        if f.is_file() and f.stat().st_size > 0 and "pose_1fps/" not in str(f):
            rows.append((str(f.relative_to(R)), f.stat().st_size,
                         md5(f) if f.stat().st_size < 5e9 else "large-see-size", str(date.today())))
    pd.DataFrame(rows, columns=["path", "bytes", "md5", "date"]).to_csv(
        R / "outputs/MANIFEST.tsv", sep="\t", index=False)
    print(f"wrote migration_log.tsv ({len(log)} actions) + MANIFEST.tsv ({len(rows)} files)")
else:
    print("\nDRY RUN — pass --execute to act")
