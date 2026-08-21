"""Load the canonical results tables and resolve published claims -> value + provenance status.

Figures must never hardcode a model number. Use claim("ladder_region") and, if the status is not
MATCH, the figure is expected to MARK the point as provisional so a reader (and we) can see which
parts of the story still rest on numbers whose runs no longer exist.
"""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"

runs = pd.read_parquet(R / "runs.parquet")
evals = pd.read_parquet(R / "evals.parquet")
published = pd.read_csv(R / "published.csv")
try:
    report = pd.read_csv(R / "provenance_report.csv")
except FileNotFoundError:
    report = None


def claim(claim_id):
    """-> dict(value, sd, status, recovered, label, rig). `value` is the number to plot:
    the recovered one when we have it, else the published one (flagged provisional)."""
    p = published[published.claim_id == claim_id]
    if not len(p):
        raise KeyError(f"unknown claim {claim_id!r}")
    p = p.iloc[0]
    st, rec = "UNCHECKED", None
    if report is not None:
        rr = report[report.claim_id == claim_id]
        if len(rr):
            st, rec = rr.iloc[0].status, rr.iloc[0].recovered
    val = rec if (st == "MATCH" and pd.notna(rec)) else p.published_value
    return dict(value=float(val), sd=None if pd.isna(p.published_sd) else float(p.published_sd),
                status=st, recovered=rec, label=p.label, rig=p.rig,
                provisional=st != "MATCH")


def family(name, metric="best_acc"):
    """All seeds of a run family -> dict(mean, sd, n, values)."""
    if metric == "posthoc_test60":
        s = evals[(evals.family == name) & (evals.eval_set == "test60")].acc
    else:
        s = runs[runs.family == name][metric]
    return dict(mean=float(s.mean()), sd=float(s.std()) if len(s) > 1 else 0.0,
                n=int(len(s)), values=sorted(s.tolist()))


def provisional_note(claim_ids):
    """One-line caption fragment naming which plotted points lack surviving provenance."""
    bad = [c for c in claim_ids if claim(c)["provisional"]]
    if not bad:
        return ""
    return (f"{len(bad)} of {len(claim_ids)} values are provisional (source runs deleted; "
            f"see notes/PROVENANCE.md D6): " + ", ".join(bad))
