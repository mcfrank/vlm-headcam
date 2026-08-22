# vlm-headcam

Learning word–object mappings from **BabyView** child egocentric video, and diagnosing the
*referential alignment* problem that makes naive contrastive training on this data fail.

**Headline:** a frozen-encoder two-tower learns real word–object mappings from raw co-occurrence
(~65% on an out-of-corpus 60-way 4AFC, chance 25). But almost every further gain requires oracle
information about *which moments are referential, which word is the referent, and which object it
names* — and no accessible cue, no encoder swap, and no architecture change recovers it. The
bottleneck is the referential signal in the data, not the machine.

## Layout

| Path | What |
|---|---|
| [`book/`](book/) | the Quarto book — 10 chapters, published at [mcfrank.quarto.pub](https://mcfrank.quarto.pub/learning-words-from-a-childs-eye-view) (`./render.sh`, publish with `./publish-public.sh`) |
| [`results/`](results/) | **single source of truth for model numbers** — scraped run tables, the registry of published claims, and the provenance checker |
| [`figures/`](figures/) | paper display items; one script each, all reading from `results/` (`make -C figures`) |
| [`src/`](src/) | pipeline: manifests, embedding, training, evaluation, figure data |
| [`notes/`](notes/) | [PHASES](notes/PHASES.md) (what each era means) · [PROVENANCE](notes/PROVENANCE.md) (what feeds what + known divergences) · [experiments](notes/experiments.md) (the running log) · [STORAGE](notes/STORAGE.md) |
| [`archive/`](archive/) | superseded exploration-era launchers |

Annotation pipelines (pose, alignment) live in a separate repo, `bv-annotations`.

## Start here

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/check_provenance.py     # which published numbers still have live sources
make -C figures                              # rebuild every display item
```

**Read [`notes/PHASES.md`](notes/PHASES.md) before trusting any number.** The project ran through
five phases with two different evaluation sets and two different metric conventions; a number is
only comparable to another from the same phase. Per-number status is in
`results/provenance_report.csv`.

## Data governance

BabyView is human-subjects data. Frames, transcripts and derived features stay on the cluster;
only aggregate results are committed. Example-frame figures are gitignored, and
`book/publish-public.sh` replaces them with a "withheld" card behind a safety gate before
publishing to a public URL.
