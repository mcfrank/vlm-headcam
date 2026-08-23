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
try:                       # dropout-corrected re-evaluation of saved ch8 checkpoints
    reeval = pd.read_csv("results/reeval_corrected.csv")
except FileNotFoundError:
    reeval = pd.DataFrame(columns=["family", "corrected"])

out = []
for _, c in pub.iterrows():
    fam, metric = c.expected_family, c.metric
    rec, n, src = None, 0, ""
    if True:
        if metric == "reeval":
            s = reeval[reeval.family == fam]
            if len(s): rec, n, src = s.corrected.mean(), len(s), "reeval_corrected.csv"
        elif metric == "posthoc_test60":
            s = ev60[ev60.family == fam]
            if len(s): rec, n, src = s.acc.mean(), len(s), s.source.iloc[0]
        else:
            s = runs[runs.family == fam]
            if len(s): rec, n, src = s.best_acc.mean(), len(s), s.store.iloc[0]
        pass
    book = c.book_value if pd.notna(c.book_value) else None
    status = "LIVE" if rec is not None else "MISSING"
    out.append(dict(claim_id=c.claim_id, chapter=str(c.chapter), label=c.label,
                    rig=c.rig, book=book,
                    current=None if rec is None else round(rec, 2),
                    drift=None if (rec is None or book is None) else round(rec - book, 2),
                    n_seeds=n, status=status, source=src or c.expected_family))
rep = pd.DataFrame(out)
rep.to_csv("results/provenance_report.csv", index=False)
counts = rep.status.value_counts().to_dict()
print(f"{len(rep)} claims -> " + " | ".join(f"{k} {v}" for k, v in counts.items()))
miss = rep[rep.status == "MISSING"]
if len(miss):
    print(f"\n--- NO LIVE SOURCE ({len(miss)}) ---")
    print(miss[["claim_id", "chapter", "label", "source"]].to_string(index=False))
d = rep[rep.drift.notna() & (rep.drift.abs() > TOL)].sort_values("drift")
if len(d):
    print(f"\n--- BOOK DRIFT: current run differs from the published book value ({len(d)}) ---")
    print(d[["claim_id", "chapter", "label", "book", "current", "drift"]].to_string(index=False))
    print("\nThese chapters need rewriting to the current numbers.")
