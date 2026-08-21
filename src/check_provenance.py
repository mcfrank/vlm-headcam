"""Check every PUBLISHED number against its recovered source.

results/published.csv  = the registry of claims made in the book/paper (hand-curated).
results/runs.parquet   = in-training metrics scraped from run logs.
results/evals.parquet  = post-hoc eval_model.py Konkle test-60 passes (the old-rig quantity).

Emits results/provenance_report.csv + a console summary. A claim is
  MATCH         recovered value within tol of published
  MISMATCH      recovered, but differs  -> investigate before the paper cites it
  UNRECOVERABLE run logs no longer exist -> must be re-run to be citable

usage: .venv/bin/python src/check_provenance.py [tol]
"""
import sys
import pandas as pd

TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
pub = pd.read_csv("results/published.csv")
runs = pd.read_parquet("results/runs.parquet")
ev = pd.read_parquet("results/evals.parquet")
ev60 = ev[ev.eval_set == "test60"]

out = []
for _, c in pub.iterrows():
    fam, metric = c.expected_family, c.metric
    rec, n, src = None, 0, ""
    if str(fam).startswith("LOST"):
        status = "UNRECOVERABLE"
    else:
        if metric == "posthoc_test60":
            s = ev60[ev60.family == fam]
            if len(s): rec, n, src = s.acc.mean(), len(s), s.source.iloc[0]
        else:
            s = runs[runs.family == fam]
            if len(s): rec, n, src = s.best_acc.mean(), len(s), s.store.iloc[0]
        if rec is None:
            status = "UNRECOVERABLE"
        else:
            status = "MATCH" if abs(rec - c.published_value) <= TOL else "MISMATCH"
    out.append(dict(claim_id=c.claim_id, chapter=f"ch{c.chapter:02d}", label=c.label,
                    rig=c.rig, published=c.published_value,
                    recovered=None if rec is None else round(rec, 2),
                    delta=None if rec is None else round(rec - c.published_value, 2),
                    n_seeds=n, status=status, source=src or c.expected_family))
rep = pd.DataFrame(out)
rep.to_csv("results/provenance_report.csv", index=False)
counts = rep.status.value_counts().to_dict()
print(f"{len(rep)} published claims -> " + " | ".join(f"{k} {v}" for k, v in counts.items()))
for st in ["MISMATCH", "UNRECOVERABLE"]:
    s = rep[rep.status == st]
    if len(s):
        print(f"\n--- {st} ({len(s)}) ---")
        print(s[["claim_id", "chapter", "label", "rig", "published", "recovered", "delta"]].to_string(index=False))
