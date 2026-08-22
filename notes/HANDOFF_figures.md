# Handoff: paper display items (figures session)

*Written 2026-08-21. You own the paper figures. A parallel session owns the compute pipeline —
see "Boundaries" before touching anything outside `figures/`.*

## What this project is

Learning word–object mappings from BabyView child egocentric video with a frozen-encoder
two-tower. Headline: the learner does acquire real word–object mappings from raw co-occurrence
(~65–73% on an out-of-corpus 60-way 4AFC, chance 25), but nearly every further gain requires
**oracle** information about which moments are referential, which word is the referent, and which
object it names. No accessible cue, encoder swap, or architecture change recovers it. *The
bottleneck is the referential signal in the data, not the machine.*

Target: PNAS. Read `notes/PHASES.md` first — the project ran through five eras with two eval sets
and two metric conventions, and **a number is only comparable to another from the same phase**.

## Start here

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # if .venv is missing
make -C figures            # rebuild every display item -> figures/out/*.pdf (+ .png preview)
make -C figures fig2       # just one
make -C figures check      # which published numbers still have live sources
```

## The one rule

**No figure may hardcode a model number.** Values come from `results/` through
`figures/data.py`:

```python
import data as D
c = D.claim("ladder_region")   # -> {value, sd, status, recovered, label, rig, provisional}
f = D.family("G_framereg")     # -> {mean, sd, n, values}  (all seeds of a run family)
```

`claim()` returns the *recovered* value when the run logs still exist and the *published* value
otherwise, with `provisional=True`. **Figures must visibly mark provisional values** (dashed/red
outline + a caption line via `D.provisional_note([...])`). This is not decoration: 7 of 38
published numbers currently have no surviving run logs, including the entire ch4 ladder, and the
marking is how we avoid shipping them unnoticed. See `notes/PROVENANCE.md` (divergences D1–D8).

## Files you own

| Path | What |
|---|---|
| `figures/theme.py` | PNAS geometry (`W1/W15/W2` = 3.42/4.5/7.0 in), Tol colorblind-safe palette with **semantic roles**, `clean()`, `panel()`, `save()` |
| `figures/data.py` | the results loader — extend this rather than reading parquets in a figure |
| `figures/figN_*.py` | one script per display item |
| `figures/out/` | built PDFs + PNG previews (gitignored? no — currently committed; keep them small) |
| `figures/Makefile`, `figures/README.md` | build harness + conventions |

**Palette semantics — keep these consistent across every figure:** `FREE` green = unaided/free
learning · `ORACLE` indigo = oracle information · `INDOM` rose = BabyView-trained encoders (the
negative result) · `CHILD` sand = human children/external reference · `LIT` purple = published
reference points · `NEUTRAL` grey = baseline when it isn't the point · `PROV` wine = provisional.

## State of each display item + feedback already given

| Fig | State | Outstanding |
|---|---|---|
| **1 pipeline** | matplotlib schematic (corpus → midpoint pairing → Gemini → two-tower → 4AFC) | Mike: *"too flow-chart-y."* Wants real BabyView frames, a photo of the camera (from the BabyView site), the referent-annotation composite (small), and proper neural-net iconography. **See Human subjects below.** |
| **2 scaling** | reworked: unfiltered only, seed band + saturating fit, CVCL reference (A), developmental-time rescaling with children's LEVANTE band (B) | Good direction. Legend marker collides with the chance line; CVCL annotation arrow is awkward. Numbers change after the re-run. |
| **3 ladder** | single panel (decomposition panel dropped as redundant) | Mike is open to folding the **cue** result in beside the ladder — the elimination logic (here's the gap; nothing accessible closes it) is tighter than a standalone cue figure. Try it. |
| **4 cues** | lollipop vs ignition band (A) + titration (B) | On hold — may merge into fig3 or move to SI. Panel B is Phase-2 rig; absolute values are **not** comparable to the Konkle ladder (internal comparison only). |
| **5 representation** | encoder bars (A) + ladder across encoders (B) | Mike: *"looks good for now."* Panel B's DINOv3 rungs are still literals pending a scrape. |
| **6 LEVANTE** | model vs VLMs with children's band (A) + frequency-vs-child-difficulty scatter (B) | "Keep in our pocket" — may become a main figure or SI. Only place real children appear. |

## Numbers are about to change — design for it

A re-run is queued (`run_phase5.sh`) that rebuilds the ladder and the scaling curve on a new
standard rig:
- **encoder → DINOv3-B/16 off-the-shelf** (was DINOv2). Unaided goes ~65 → ~73; the referential
  gap shrinks +20 → +14 but does not close.
- **epoch selection → Konkle dev-117**, reporting test-60 (removes ~1.4 pts of best-on-test
  optimism that ranged 0.4–3.5 across conditions and so distorted comparisons).
- every run now writes `metrics.json`, so `results/` will rebuild by scraping run dirs.

**Do not chase current values.** When the runs land, someone re-runs the scrapers, `results/`
updates, and every figure should just rebuild with `make -C figures`. If a figure needs editing to
absorb new numbers, that figure is doing something wrong.

A second re-run (`run_2026_pipeline.sh`) adds the 2026.1 release — **51 children vs 36** — as a
scale/diversity extension. That mainly affects fig2 and any diversity panel.

## Human subjects — the hard constraint

BabyView frames show children's faces. We **have** approval for scientific sharing, but:
- **Nothing frame-derived is committed or published right now** (Mike's call). `book/figures/frames/`
  is gitignored and `book/publish-public.sh` swaps in a "withheld" card behind a safety gate.
- For fig1's real frames, use the blur tool first:
  `ssh ccn2-14`, then
  `PYTHONPATH=src /data2/mcfrank/mmpose_env/bin/python src/blur_faces.py --manifest <csv> --out <dir>`
  It locates heads from the 133-kpt pose we already have (dense face landmarks → head keypoints →
  bbox-top fallback) and blurs irreversibly, printing a per-frame report that flags frames where
  the head had to be guessed or no person was found. **Always eyeball the output**; pose coverage
  is ~66%, so a missed person means no blur.
- The referent-annotation composite lives at `book/figures/frames/gemini_examples/` (gitignored),
  with 12 positive + 12 negative candidate cards and `pos.csv`/`neg.csv` to recompose.

## Boundaries (two sessions, one repo)

**You touch:** `figures/**`, and `results/published.csv` *only* to add a claim_id a figure needs.
**The pipeline session touches:** `src/**`, `run_*.sh`, `results/*.parquet`, `notes/experiments.md`.
**Shared, coordinate first:** `notes/PROVENANCE.md`, `README.md`, `book/**`.

Pull before you start and commit narrowly (`git add figures/`) — the other session pushes to
`master` too. If `results/*.parquet` changes under you, that is expected: rebuild, don't revert.

## Useful context files

`notes/PHASES.md` (eras) · `notes/PROVENANCE.md` (what feeds what, divergences D1–D8) ·
`results/README.md` (how the tables are built, the two metric conventions) ·
`notes/experiments.md` (the running log; phases 4–5 at the end) · `notes/STORAGE.md` (what lives where).
The book at `book/` is the long-form version of every result and is published at
mcfrank.quarto.pub/learning-words-from-a-childs-eye-view.
