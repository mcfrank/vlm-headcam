"""Load the canonical results table for the figures.

Everything paper-visible now comes from results/runs.parquet (scraped per-run metrics.json)
plus per-analysis tables written by src/. The 2025.2-era published.csv / provenance_report.csv
claim machinery and the cues-era titration.csv are no longer read by any figure: the claim(),
provisional_note() and ignition_band() helpers were removed when the ladder, cues and
representation figures retired. If a figure ever needs a published claim again, restore it
from git history rather than re-adding a stale source silently.
"""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"

runs = pd.read_parquet(R / "runs.parquet")


def family(name, metric="best_acc"):
    """All seeds of a run family -> dict(mean, sd, n, n_pairs, values)."""
    d = runs[runs.family == name]
    s = d[metric]
    return dict(mean=float(s.mean()), sd=float(s.std()) if len(s) > 1 else 0.0,
                n=int(len(s)),
                n_pairs=float(d.n_pairs.iloc[0]) if len(d) and d.n_pairs.notna().any()
                else float("nan"),
                values=sorted(s.tolist()))
