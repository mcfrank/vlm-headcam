"""M2: recreate the mp3 layer for all of 2026.1.

Per release video: copy the 2025.2 mp3 when one exists (byte-identical source videos);
otherwise ffmpeg-extract audio from the gcloud pull. Resumable (skips existing non-empty
targets); failures logged, not fatal. Run AFTER the embedding job (same NFS server).

usage: python src/make_mp3_2026.py [--workers 16] [--limit N]
"""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

R = Path("/ccn2b/dataset/babyview/2026.1")
OLD = Path("/ccn2a/dataset/babyview/2025.2/mp3")
PULL = Path("/ccn2b/dataset/babyview/gcloud/pull")

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--limit", type=int, default=0, help="only N videos (pilot)")
a = ap.parse_args()

idx = pd.read_csv(R / "outputs/release_index.tsv", sep="\t")
jobs = []
for r in idx.itertuples():
    dst = R / "mp3" / r.subject_id / f"{r.video_id}.mp3"
    if dst.exists() and dst.stat().st_size > 0:
        continue
    old = OLD / r.subject_id / f"{r.video_id}.mp3"
    if old.exists():
        jobs.append(("copy", old, dst))
    else:
        vids = list(PULL.glob(f"*/{r.subject_id}/{r.video_id}.[mM][pP]4"))
        jobs.append(("extract", vids[0], dst) if vids else ("MISSING-VIDEO", r.video_id, dst))
if a.limit:
    jobs = jobs[:a.limit]
print(f"todo: {sum(k=='copy' for k,_,_ in jobs):,} copies, "
      f"{sum(k=='extract' for k,_,_ in jobs):,} extracts, "
      f"{sum(k=='MISSING-VIDEO' for k,_,_ in jobs):,} missing videos")

fails = []
def run(job):
    kind, src, dst = job
    if kind == "MISSING-VIDEO":
        fails.append(job); return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp3")
    try:
        if kind == "copy":
            subprocess.run(["cp", str(src), str(tmp)], check=True)
        else:  # match the 2025.2 layer: 44.1k mono-preserving lame VBR
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                            "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(tmp)],
                           check=True, timeout=3600)
        tmp.rename(dst)
    except Exception as e:
        tmp.unlink(missing_ok=True); fails.append((kind, str(src), str(e)[:80]))

with ThreadPoolExecutor(a.workers) as ex:
    for i, _ in enumerate(ex.map(run, jobs)):
        if i % 500 == 0:
            print(f"  {i}/{len(jobs)} ({len(fails)} failed)", flush=True)
pd.DataFrame(fails, columns=["kind", "src", "err"]).to_csv(R / "outputs/mp3_failures.tsv",
                                                           sep="\t", index=False)
print(f"done: {len(jobs) - len(fails):,} ok, {len(fails):,} failed -> outputs/mp3_failures.tsv")
