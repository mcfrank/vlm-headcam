"""Supplementary experiments table, GENERATED: one row per experiment block — encoders,
training pairs, seeds/draws per cell, total runs, evaluation, and which figure scripts
consume the runs (parsed from figures/*.py, so the figure column cannot drift from the
figures). Emits results/experiments_table.{tex,csv}. Rerun after any new run family.
usage: python src/make_experiments_table.py"""
import glob
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "figures")
import theme as T                      # the encoder registry: tag -> paper label

runs = pd.read_parquet("results/runs.parquet")
f = runs[runs.run.str.startswith("F_")].copy()
ENC = {e["tag"]: e["label"] for e in T.ENCODERS}
ENC_ORDER = [e["label"] for e in T.ENCODERS]
pat = re.compile(r"F_(dinov3l|dinov3b|dinov3s|vitl_bv|vitb_bv|vits_bv)_(.+)_s(\d+)$")
rows = []
for r in f.itertuples():
    m = pat.match(r.run)
    if not m: continue
    rows.append(dict(enc=ENC[m.group(1)], cond=m.group(2), seed=int(m.group(3)),
                     n_pairs=getattr(r, "n_pairs", None), family=r.family))
d = pd.DataFrame(rows)

# experiment blocks: (name, regex on cond, evaluation, notes)
BLOCKS = [
    ("Scaling (random subsamples) + full corpus", r"^(rand_\d+|base)$", "Konkle; LEVANTE; in-domain; lexicon", ""),
    ("Aligned-pair scaling", r"^align_\d+$", "Konkle; lexicon", "top-N by Gemini alignment"),
    # the ladder families (lad_*, lad100000_*, lad300000_*) are in runs.parquet but no
    # figure in the manuscript uses them, so they are left out of the table
    ("Diversity (children at fixed budget)", r"^div(30k|100k|300k)_\d+c$", "Konkle", "random child draw per seed"),
    ("No-MIL (whole-frame) control", r"^wf(30000|300000|full)$", "Konkle", "mean-over-grid R=1; paired to region runs"),
    ("Alignment-selection controls", r"^(alignedonly|rand172k|matchrand|minusaligned|minusmatch|minusrand)$", "Konkle", "exposure+length-matched; full-minus"),
    ("Child-speech (KCHI) control", r"^(nokchi|randmatch)$", "Konkle", "matched-N random"),
    ("Temporal-window ($\\pm$5 s) control", r"^win5_(\d+|full)$", "Konkle", "complete neighbor caches"),
]

# figure -> family patterns (templated {enc}/{e} expanded to any encoder)
figmap = {}
for fp in sorted(glob.glob("figures/fig*.py")):
    src = open(fp).read()
    pats = set(re.findall(r"F_[A-Za-z0-9_{}<>\-]+", src))
    rx = []
    for p in pats:
        # a template placeholder stands for ONE token (an encoder or a condition word); a
        # pattern that is nothing but placeholders after F_ is unattributable by regex
        q = re.sub(r"\{[^}]*\}?", "[A-Za-z0-9]+", p)
        if not re.search(r"[a-z]", re.sub(r"\[A-Za-z0-9\]\+|F_|_", "", q)):
            continue
        rx.append(re.compile("^" + q))
    figmap[Path(fp).stem] = rx
# scripts whose family references are fully templated (read via helpers) — explicit map
MANUAL = {"figS_nomil": ["No-MIL", "Scaling"], "figS_alignment_controls": ["Alignment-selection"],
          "figS_window": ["Temporal-window", "Scaling"]}
# scripts that consume these runs through DERIVED tables (item-level Konkle, LEVANTE,
# in-domain, lexicon extractions) rather than by family name, so the regex scan cannot see
# them. Keyed the same way as MANUAL: figure stem -> experiment-name prefixes.
DERIVED = {"fig3_lexicon": ["Scaling"],
           "fig4_development": ["Scaling"],
           "figS_levante": ["Scaling"],
           "figS_encoder_probe": ["Scaling"],
           "figS_item_difficulty": ["Scaling"],
           "figS_lexicon_tsne": ["Scaling"],
           "figS_lexicon_relatedness": ["Scaling"],
           "figS_lexicon_alignment": ["Scaling", "Aligned-pair"]}

def fmt(n):
    return f"{n/1e6:.2g}M" if n >= 1e6 else (f"{n//1000}k" if n >= 1000 else str(n))

NAMED = {"base": "full (1.69M)", "filtnat": "172k referent-bearing", "t15": "172k referent-bearing",
         "t2": "172k referent-bearing", "alignedonly": "172k (aligned)", "rand172k": "172k (plain random)",
         "matchrand": "172k (matched, unaligned)",
         "minusaligned": "1.51M (full$-$aligned)", "minusmatch": "1.51M (full$-$matched)",
         "minusrand": "1.51M (full$-$random)", "win5_full": "full (1.69M)",
         "nokchi": "1.30M (no child speech)", "randmatch": "1.30M (matched)"}

def nominal(cond):
    """the DESIGNED quantity a condition name encodes (not the post-vocab effective count)"""
    cond = re.sub(r"^lad_", "", cond)
    if cond in NAMED: return NAMED[cond]
    if cond == "wffull": return "full (1.69M)"
    m = re.match(r"(?:rand|align|win5)_(\d+)$|wf(\d+)$", cond)
    if m: return fmt(int(m.group(1) or m.group(2)))
    m = re.match(r"lad(\d+)_(base|filtnat|t15|t2)$", cond)
    if m: return fmt(int(m.group(1))) + " (all rungs from one matched subsample)"
    m = re.match(r"div(\d+)k_\d+c$", cond)
    if m: return f"{m.group(1)}k budget"
    return cond

def figs_for(block_conds):
    hit = []
    fams = [f"F_{e}_{c}" for e in ENC for c in block_conds]
    for fig, rx in figmap.items():
        if any(r.match(fam) for r in rx for fam in fams):
            hit.append(fig)
    return hit

out = []
for name, rx, ev, note in BLOCKS:
    b = d[d.cond.str.match(rx)]
    if b.empty:
        out.append(dict(experiment=name, encoders="(pending)", pairs="", seeds="", runs=0, evaluation=ev, figures="", note=note)); continue
    encs = [e for e in ENC_ORDER if e in set(b.enc)]
    seen, pairs_list = set(), []
    def key(c):
        m = re.findall(r"\d+", re.sub(r"^win5_", "", c)); return int(m[-1]) if m else 10**9   # win5_full -> last
    for c in sorted(set(b.cond), key=key):
        q = nominal(c)
        if q not in seen: seen.add(q); pairs_list.append(q)
    pairs = "; ".join(pairs_list)
    seeds = b.groupby(["enc", "cond"]).seed.nunique()
    srange = f"{seeds.min()}--{seeds.max()}" if seeds.min() != seeds.max() else str(seeds.max())
    figs = figs_for(sorted(set(b.cond)))
    figs += [k for k, v in {**MANUAL, **DERIVED}.items()
             if any(name.startswith(x) for x in v) and k not in figs]
    figs = sorted(figs, key=lambda k: (not k.startswith("fig") or k.startswith("figS"), k))
    out.append(dict(experiment=name, encoders=", ".join(encs), pairs=pairs, seeds=srange,
                    runs=len(b), evaluation=ev, figures=", ".join(figs), note=note))
T = pd.DataFrame(out)
T.to_csv("results/experiments_table.csv", index=False)
def figref(name):   # figure script stem -> \ref{fig:anchor}; anchor = stem minus fig/figS prefix
    return "\\ref{fig:" + re.sub(r"^fig[S\d]*_", "", name) + "}"

with open("results/experiments_table.tex", "w") as fh:
    fh.write("% AUTO-GENERATED by src/make_experiments_table.py from results/runs.parquet + figures/*.py\n")
    fh.write("\\begin{table*}[t]\\centering\\footnotesize\n\\caption{Training runs underlying all figures. "
             "Every run uses a frozen encoder, region-MIL head, InfoNCE, 20 epochs, and dev-117 epoch selection; "
             "quantities are designed training pairs (full corpus = 1,686,105).}\n\\label{tab:experiments}\n")
    fh.write("\\begin{tabular}{p{3.1cm}p{1.5cm}p{4cm}p{0.8cm}p{0.7cm}p{2.3cm}p{2.4cm}}\\toprule\n")
    fh.write("Experiment & Encoders & Training pairs & Seeds & Runs & Evaluation & Figures \\\\\\midrule\n")
    for r in T.itertuples():
        encs = {", ".join(ENC_ORDER): "all six"}.get(str(r.encoders), str(r.encoders))
        figs = ", ".join(figref(x.strip()) for x in str(r.figures).split(",") if x.strip() and x.strip() != "nan")
        fh.write(f"{r.experiment} & {encs} & {r.pairs} & {r.seeds} & {r.runs} & {r.evaluation} & {figs} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table*}\n")
print(T.to_string(index=False))
print(f"\nTOTAL runs: {T.runs.sum()}")
