# vlm-headcam — session handoff

**Project.** Learn word–object mappings from BabyView (child egocentric video + ASR
transcripts); understand why naive CLIP-style training fails and what a naive learner could
do about it. Repo pushed to GitHub `mcfrank/vlm-headcam`.

## Where to look
- **`book/`** — the documentation, as a Quarto book (render: `cd book && ./render.sh`, read
  `book/_book/index.html`). Chapters: Background · Pipeline (architecture + worked examples)
  · Alignment experiments (ch.3) · **Bootstrapping (ch.5)** · Where next.
- **`notes/experiments.md`** — the running experiment log (every run, numbers, verdicts).
- **`src/`** — code. `train.py` (two-tower), `train_boot.py` (cosine-EM bootstrap),
  `train_region_mil.py` (region-MIL + all bootstrap E-step variants), `embed_frames.py` /
  `embed_regions.py`, `build_*` data prep. `eval/` — analysis scripts + `p1_findings.md`.
- **`background/`** — the source papers (gitignored; local only).

## On ccn2 (`ssh ccn2-14`)
- Env: `/data2/mcfrank/ladder/condaenv/bin/python` (torch 2.4.1, transformers, PIL, mpl; no
  torchvision → slow DINO processor, fine). Read `~/.claude/skills/train-on-ccn2/SKILL.md`.
- Work dir `/data2/mcfrank/vlm-headcam/`: `emb_full/` (1.27M frame CLS embeddings, 1.94GB),
  `emb_reg/` (342k region grids [N,17,768]), `manifests/`, `runs/` (all logs + DONE markers).
- Raw data `/ccn2a/dataset/babyview/2025.2/`: 1fps frames, parsed transcripts,
  `full_clip_results.csv` (CLIP alignment filter), CDI YOLOE detections.

## Starting point (what's established)
1. **Alignment filtering works** (ch.3): filtering to CLIP-aligned pairs beats random by
   +12 cross-child and beats 8× more unfiltered data; count-vs-quality sweet spot ~22k
   pairs. Region/object grounding adds +3–4. Injected-label topline (frozen-feature
   ceiling) ≈ 72, so natural 49 has headroom — bottleneck is *alignment quality*, not
   vision features.
2. **But the filter can't be bootstrapped unsupervised** (ch.5): across 6 mechanisms
   (cosine-EM, region-MIL, language prior, distinctiveness, cross-situational prototypes,
   curriculum) the self-supervised alignment weight never correlates with true alignment
   (ρ ≤ 0.1); boot 4AFC stuck ~34–38 vs oracle 49.9. Gains came only from better
   *representation* (region-MIL) and *transfer* (curriculum 37.6). Recurring: language >
   vision, within-kid > across-kid. Conclusion: co-occurrence is too sparse to *discover*
   reference at BabyView's floor → reference must be socially cued.

## Candidate next steps (not yet run)
- **Mutual-exclusivity / competitive assignment** E-step (structural prior, not just a
  better score) — the main untried bootstrap mechanism.
- **Add a social-cue channel** — gaze/pointing/joint-attention proxies (the data has
  IMU/head-motion); test whether a *cued* intent prior ignites the filter.
- **Unfreeze the vision tower** on a winning recipe (frozen ceiling ~72 leaves headroom).
- Combine the two winners: crop/region grounding on the 0.26-filtered set.

## Gotchas
- Human-subjects frames stay OUT of git (only aggregate result plots committed; frame
  figures regenerate via `src/make_figures.py` on ccn2, gitignored).
- ccn2 is shared + unscheduled: check GPUs, run in free headroom, don't kill others' jobs.
- Detached launches need absolute script paths (setsid runs from `$HOME`).
