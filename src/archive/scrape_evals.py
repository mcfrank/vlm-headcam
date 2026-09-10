"""Scrape the POST-HOC Konkle evaluations — the quantity the book actually reports.

Runs were trained with an in-training eval that varied by era; the published numbers come from
separate `eval_model.py <run> eval_frames_konkle.parquet` passes launched at the end of the
chain scripts. Their stdout survives only inside logs/*chain*.log, logs/scaling_seeds.log, etc.
This parses those lines so every published number has a machine-readable source.

usage: python src/scrape_evals.py     # -> results/evals.parquet
"""
import glob, os, re
import pandas as pd

W = "/data2/mcfrank/vlm-headcam"
# "name  test60: 4AFC=63.7  (cats scored 60/60)" | "name: 4AFC=27.6 (cats ...)" | "name 4AFC=65.0"
PAT = re.compile(r"^\s*(\S+?):?\s+(?:(test60|dev117):\s*)?4AFC=([0-9.]+)(?:\s*\(cats scored (\d+)/(\d+)\))?")
rows, section = [], None
for lf in sorted(glob.glob(f"{W}/logs/*.log")):
    for line in open(lf, errors="ignore"):
        if line.startswith("==="):
            section = line.strip().strip("= ")
            continue
        m = PAT.match(line)
        if not m:
            continue
        name, evalset, acc, scored, tot = m.groups()
        name = name.rstrip(":")
        sm = re.search(r"_s(\d+)$", name)
        rows.append(dict(
            run=name, family=re.sub(r"_s\d+$", "", name),
            seed=int(sm.group(1)) if sm else 0,          # unsuffixed launcher lines are seed 0
            eval_set=evalset or ("test60" if "dev" not in (section or "").lower() else "dev117"),
            acc=float(acc), cats_scored=int(scored) if scored else None,
            cats_total=int(tot) if tot else None,
            section=section, source=f"logs/{os.path.basename(lf)}",
        ))
df = pd.DataFrame(rows).drop_duplicates(["run", "eval_set", "acc"]).sort_values(["family", "seed"])
os.makedirs(f"{W}/results", exist_ok=True)
df.to_parquet(f"{W}/results/evals.parquet", index=False)
print(f"{len(df)} post-hoc eval records | {df.family.nunique()} families | eval sets: {dict(df.eval_set.value_counts())}")
t = df[df.eval_set == "test60"]
print("\nch6 scaling family (the published curve):")
for f in sorted(x for x in t.family.unique() if "scale_rand" in x or "scale_align" in x):
    s = t[t.family == f]
    print(f"  {f:22s} n={len(s)} mean={s.acc.mean():5.1f}  seeds={sorted(s.acc.tolist())}")
