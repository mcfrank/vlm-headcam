# Bootstrapping experiments — running log

Goal: make the **self-bootstrapped** alignment filter ignite (no external CLIP/detector
in training). Eval = 4AFC on held-out child S00360001 (chance 25). Frozen-DINOv2 probe.
Reference points: CLS-cosine oracle (CLIP-filtered) across = **46.7**, within = 34.9;
injected-label topline (frozen-feature ceiling) ≈ **71.7**; random/unweighted ≈ 28–35.

Yardstick for "did the filter ignite": ρ(endogenous weight, held-out CLIP score) and
precision of selected pairs vs clip>0.24 base rate. P1 gave ρ≈0.03–0.07 (no toehold).

---

## E0 — P1 recap (DONE, negative)
Cosine-EM on CLS embeddings. boot ≈ unweighted at all sizes; ρ(w,clip)=0.02–0.07;
4AFC stuck 26–33 (≪ oracle 46.7). Within-kid slightly stronger toehold (ρ~0.07) but lower
ceiling (oracle 34.9). **Diagnosis:** instantaneous CLS-cosine on one midpoint frame is
too weak a signal at BabyView's ~5–13% alignment floor.

## Plan (overnight)
- **E1 region-MIL:** embed a coarse region grid (DINOv2 patch tokens → 4×4 + CLS) and let
  an utterance match its *best region* (MIL). Tests: (a) does region attention raise the
  oracle ceiling? (b) does the max-region E-step create a toehold the CLS-cosine lacked?
  This is the "which frames AND which regions" double-sparsity idea.
- **E2 language-informed E-step (non-cheating):** weight utterances by the model's OWN
  lexicon confidence (does the utterance contain words it already grounds?) — FGT's
  referential channel, bootstrapped, no injected knowledge. Design after E1.
- **E3 curriculum / co-teaching** if E1–E2 show partial life.

## Results table (best 4AFC; ρ = spearman(weight, clip))
| id | idea | pool | 4AFC | ρ(w,clip) | notes |
|----|------|------|------|-----------|-------|
| E0 | cosine-EM | across140k | 32.2 | 0.03 | no ignition |
| E0 | cosine-EM | within110k | 31.2 | 0.06 | no ignition |

## Status (overnight)
- **E1 region-MIL RUNNING** (on-box orchestrator run_batch1.sh, GPUs 1,6 — node shared
  with another user's 8-GPU job, so running politely in headroom). Region grid = DINOv2
  patch tokens → 4×4 + CLS. 10 jobs: oracle/plain/boot × across/within × sizes.
  Marker: runs/BATCH1_DONE. NOTE: other user occupies all GPUs ~24-31GB; I use only 1,6.
- **E2 lang-prior READY** (train_region_mil.py --lang-prior): folds a bootstrapped
  word-groundedness prior (a word is "nameable" if the model reliably finds a matching
  region when it's said; function words self-exclude — no POS/CLIP). To run after E1,
  informed by whether region-MIL alone created a toehold.
- Robustness: each batch self-contained with DONE markers; if a client waiter misfires,
  resume by checking runs/*_DONE and runs/*/log.json.

## Idea pool (overnight, expanding)
- **E1 region-MIL** — utterance matches best region (DINOv2 patch grid). [Batch 1, running]
- **E2 language prior** — weight utterances containing words the model already grounds
  (bootstrapped nameability; function words self-exclude). [ready, --lang-prior]
- **E8 distinctiveness / base-rate correction** — subtract each region's generic salience
  so ubiquitous regions (wall/floor/hands) stop winning; the neural intent-prior from
  ch.8's diagnosis. [ready, --score-mode distinct]  → Batch 2 with E2.
- **E5 curriculum within→across** — bootstrap in one child's consistent world (easy
  toehold, per E0) then continue on the pooled corpus (higher ceiling). Needs shared vocab
  + checkpoint continuation. [Batch 3 candidate]
- **E6/E9 cross-situational prototype** — maintain an EMA prototype per word from its
  confidently-matched regions; alignment = does the region match the running prototype?
  Cross-situational accumulation / propose-but-verify flavor. [Batch 3 candidate]
- **E4 co-teaching** — two peers select confident pairs for each other (once a toehold
  exists). [later]

Batch 2 = region-MIL boot × {lang(E2), distinct(E8), distinct+lang} × {across-140k,
within-110k, across-60k}. Compares against Batch 1's plain region-MIL boot.
