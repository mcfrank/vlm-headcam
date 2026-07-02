# vlm-headcam

Prototype infrastructure for learning word–object mappings from **BabyView**
child egocentric video, and for iterating on the *referential alignment* problem
that makes naive contrastive training on this data fail.

- **Documentation:** a Quarto book in [`book/`](book/) — background, a full
  pipeline walkthrough (architecture, worked examples), and four experiments.
  Render with `cd book && ./render.sh`; read `book/_book/index.html`.
- **Code:** [`src/`](src/) — data prep, frozen-DINOv2 embedding, the two-tower
  contrastive model, the 4AFC evaluation, region grounding, and a per-GPU
  experiment dispatcher. Runs on the ccn2 cluster.

## Findings so far

Two cheap, composable levers move word–object 4AFC accuracy well above chance on
a corpus where naive training sits near chance:

1. **Alignment filtering** — train on the frame–utterance pairs with high CLIP
   similarity. +12 points cross-child (size-matched); a filtered 137k-pair subset
   beats the full unfiltered 1.14M stream; count-vs-quality has a sweet spot (~22k
   pairs).
2. **Region grounding** — pair the utterance with the cropped object rather than
   the whole scene. +3–4 points; best result overall.

## Data governance

BabyView is human-subjects data. Frames, transcripts, and derived features stay
on the cluster; only aggregate results are committed. Example-frame figures in
the book are regenerated locally (`src/make_figures.py`) and are gitignored.
