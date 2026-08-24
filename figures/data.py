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
    # published.csv gained a `book_value` column on 2026-08-23 (what the BOOK prints) while the
    # live number now comes from the runs. Accept either schema so this keeps working.
    if "published_value" not in p.index and "book_value" in p.index:
        p = p.copy()
        p["published_value"] = p["book_value"]
        p["published_sd"] = float("nan")
    st, rec = "UNCHECKED", None
    if report is not None:
        rr = report[report.claim_id == claim_id]
        if len(rr):
            _r = rr.iloc[0]
            st = _r.status
            rec = _r.recovered if "recovered" in rr.columns else _r.get("current")
    val = rec if (st in ("MATCH", "LIVE") and pd.notna(rec)) else p.published_value
    return dict(value=float(val), sd=None if pd.isna(p.published_sd) else float(p.published_sd),
                status=st, recovered=rec, label=p.label, rig=p.rig,
                provisional=st not in ("MATCH", "LIVE"))


def family(name, metric="best_acc"):
    """All seeds of a run family -> dict(mean, sd, n, values)."""
    if metric == "posthoc_test60":
        s = evals[(evals.family == name) & (evals.eval_set == "test60")].acc
    else:
        s = runs[runs.family == name][metric]
    sub = runs[runs.family == name]
    npairs = float(sub.n_pairs.iloc[0]) if len(sub) and "n_pairs" in sub else None
    return dict(mean=float(s.mean()), sd=float(s.std()) if len(s) > 1 else 0.0,
                n=int(len(s)), values=sorted(s.tolist()), n_pairs=npairs)


def ignition_band():
    """(lo, hi): the cue-quality bracket in which a hard gate first overtakes soft weighting.

    Below `lo` selecting on the cue buys nothing over just down-weighting; the band is what a
    learner would need to reach before filtering for referential moments could pay. Derived from
    results/titration.csv so fig3 and the SI titration figure cannot drift apart.
    """
    t = pd.read_csv(R / "titration.csv").pivot(index="rho", columns="mode",
                                               values="acc_4afc").dropna()
    over = t.index[t["gate"] > t["soft"]]
    return float(t.index[t.index < over.min()].max()), float(over.min())


def provisional_note(claim_ids):
    """One-line caption fragment naming which plotted points lack surviving provenance."""
    bad = [c for c in claim_ids if claim(c)["provisional"]]
    if not bad:
        return ""
    return (f"{len(bad)} of {len(claim_ids)} values are provisional (source runs deleted; "
            f"see notes/PROVENANCE.md D6): " + ", ".join(bad))
