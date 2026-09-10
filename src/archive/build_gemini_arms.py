"""From one Gemini-scored pool, build matched-count training arms: top-N by Gemini
alignment vs top-N by CLIP score, drawn from the SAME pool (fair ranker comparison).
Also audits diversity (unique videos / children / referents, and Gemini↔CLIP overlap)
to check the selected subsample isn't collapsed onto one register."""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="scored/pool_flash.parquet")
ap.add_argument("--ns", default="15000,22000", help="comma-sep selection counts")
ap.add_argument("--outdir", default="manifests")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

pool = pd.read_parquet(args.pool)
pool = pool[pool.alignment.notna() & pool.clip_score_max.notna()].copy()
pool["key"] = pool.video_id + "|" + pool.frame_idx.astype(str)
# shuffle so ties (integer alignment) break randomly, not by original order
pool = pool.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
print(f"pool: {len(pool)} scored pairs | "
      f"align mean {pool.alignment.mean():.1f}  clip mean {pool.clip_score_max.mean():.3f}")

cols = ["video_id", "frame_idx", "text"]
for N in [int(x) for x in args.ns.split(",")]:
    gtop = pool.sort_values("alignment", ascending=False, kind="stable").head(N)
    ctop = pool.sort_values("clip_score_max", ascending=False, kind="stable").head(N)
    gtop[cols].to_parquet(f"{args.outdir}/gemini_top{N}.parquet", index=False)
    ctop[cols].to_parquet(f"{args.outdir}/clip_top{N}.parquet", index=False)
    overlap = len(set(gtop.key) & set(ctop.key)) / N
    print(f"\n=== N={N} ===")
    print(f"  gemini arm: alignment>= {gtop.alignment.min():.0f} (median {gtop.alignment.median():.0f}) "
          f"| clip range {gtop.clip_score_max.min():.3f}-{gtop.clip_score_max.max():.3f}")
    print(f"  clip arm  : clip>= {ctop.clip_score_max.min():.3f} "
          f"| alignment median {ctop.alignment.median():.0f}")
    print(f"  overlap gemini∩clip: {overlap*100:.1f}%   (low = they pick different pairs)")
    for lab, s in [("gemini", gtop), ("clip", ctop)]:
        cid = s.child_id.nunique() if "child_id" in s else "?"
        ref = s.referent.replace("", pd.NA).nunique() if "referent" in s else "?"
        print(f"    {lab:6s}: {s.video_id.nunique()} videos, {cid} children, {ref} distinct referents")
