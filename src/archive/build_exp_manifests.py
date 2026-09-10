"""Build all manifests for exps A (divergence), B (robustness), C (threshold sweep).
Random arms are size-matched per held-out child to the corresponding aligned arm."""
import subprocess
import pandas as pd
from pathlib import Path
from common import MANIFEST_DIR

PY = "/data2/mcfrank/ladder/condaenv/bin/python"
CHILDREN = ["S00360001", "S00240001", "S00370002"]


def bp(args):
    subprocess.run([PY, "build_pairs.py"] + args, check=True)


def n(name):
    return len(pd.read_parquet(Path(MANIFEST_DIR) / f"{name}.parquet"))


# --- Exp B: aligned + size-matched random, per held-out child ---
for ch in CHILDREN:
    bp(["--name", f"aligned_{ch}", "--min-clip", "0.24", "--exclude-children", ch])
    k = n(f"aligned_{ch}")
    bp(["--name", f"random_{ch}", "--min-clip", "0.0", "--max-pairs", str(k),
        "--sample", "random", "--exclude-children", ch])

# --- Exp C: threshold sweep at S00360001 (0.24 == aligned_S00360001) ---
for thr in ["0.26", "0.28"]:
    bp(["--name", f"thresh_{thr}", "--min-clip", thr, "--exclude-children", "S00360001"])

# --- Exp A: full unfiltered (exclude eval child S00360001) ---
bp(["--name", "unfiltered_S00360001", "--min-clip", "0.0", "--exclude-children", "S00360001"])

print("\n=== manifest sizes ===")
for p in sorted(Path(MANIFEST_DIR).glob("*.parquet")):
    if p.stem.startswith(("aligned_", "random_", "thresh_", "unfiltered_")):
        print(f"  {p.stem}: {n(p.stem)}")
