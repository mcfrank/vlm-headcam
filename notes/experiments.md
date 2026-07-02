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
| E1 | region-MIL ORACLE | across | **49.9** | — | region attention raises ceiling +3.2 vs CLS 46.7 |
| E1 | region-MIL ORACLE | within | 39.4 | — | +4.5 vs CLS 34.9 |
| E1 | region-MIL unweighted | across140k | 34.8 | — | +1.8 vs CLS; no filter |
| E1 | region-MIL boot | across140k | 34.3 | 0.035 | **still no ignition** (E-step ρ unchanged) |
| E1 | region-MIL boot | within110k | 31.4 | 0.062 | still no ignition |

### E1 takeaway
Region-MIL **raises the achievable ceiling** (oracle 46.7→49.9) and helps the unweighted
baseline (+2), confirming region attention is a better representation. BUT the **bootstrap
still doesn't ignite**: the max-region E-step's ρ(weight,clip) stays 0.03–0.07 — same as
CLS cosine-EM. Stronger *representation* ≠ stronger *self-supervision signal*. The gap
between unweighted region-MIL (34.8) and region-MIL oracle (49.9) — ~15 pts — is what a
working bootstrap could recover. → E8 (distinctiveness) + E2 (lang prior) target the
E-step signal directly [Batch 2, launching].

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

### E9 cross-situational prototype (Batch 3) — also no ignition
proto_across140 34.4 (ρ0.04); protolang_within110 33.0 (ρ0.085); protolang_across140 34.0
(ρ0.063). Accumulating word→region prototypes across situations does NOT beat the language
prior. 4AFC stuck ~34 (=unweighted baseline), far below oracle 49.9.

### Verdict after E0/E1/E2/E8/E9: bootstrap does not ignite
Five distinct self-supervised alignment signals (cosine, region-MIL max, distinctiveness,
language prior, cross-situational prototype) all fail to separate aligned pairs at
BabyView's noise floor. Best ρ(weight,clip) ≈ 0.095 (language prior, within-kid); 4AFC
never exceeds ~35 vs oracle 49.9. Two consistent faint signals: **language side** > vision
side, and **within-kid** > across-kid toehold. → E5 curriculum [Batch 4, running]: does
the within-kid toehold, transferred, seed the across-kid bootstrap?

Implication if E5 also fails: BabyView's raw stream is too referentially sparse for
UNSUPERVISED bootstrapping with frozen features — a child-plausible learner likely needs
the extra cues children actually have (gaze/pointing/joint attention, prosody), which this
stream lacks. That is itself the paper's point: alignment must be *given* by social
structure, not discovered from co-occurrence alone.

### E5 curriculum (Batch 4) — the one bright spot: transfer helps, filter still doesn't
| run | 4AFC | warmup | ρ |
|-----|------|--------|---|
| scratch across (matched control) | 35.5 | 33.2 | 0.017 |
| stage1 within | 34.2 | 33.6 | 0.067 |
| **stage2 across (init from within)** | **37.6** | 36.2 | 0.008 |
Curriculum transfer (within→across) gives the **best bootstrap of the night, 37.6** vs
35.5 from-scratch (+2.1). BUT the gain is an **initialization effect** — warmup is already
36.2 before any EM, and ρ stays ~0. The within-kid model learns transferable word meanings
unsupervised; the alignment *filter* still never ignites. Best bootstrap 37.6 vs oracle
49.9 — a ~12pt gap a working filter could still recover.

## OVERNIGHT SUMMARY (E0–E9)
Six mechanisms. Unsupervised alignment *filtering* does not ignite on BabyView (ρ≤0.1
everywhere). Positives: (1) region attention raises the representation ceiling (oracle
46.7→49.9); (2) curriculum transfer within→across is the best bootstrap (37.6). Robust
directional signals: language > vision, within-kid > across-kid. Conclusion: co-occurrence
alone is too sparse to discover alignment here; the gains come from better *representation*
(regions) and *transfer* (curriculum), not from a self-discovered filter — consistent with
children needing social/attentional structure to supply reference.

### E2/E8 results (Batch 2) — modest, no ignition
| variant | pool | 4AFC | ρ(w,clip) |
|---------|------|------|-----------|
| lang (E2) | within110 | 32.6 | **0.095** (best ρ yet) |
| lang (E2) | across140 | 34.9 | 0.059 |
| distinct (E8) | across140 | 34.6 | 0.013 (distinct alone doesn't help ρ) |
| distinct+lang | across140 | **35.1** | 0.048 |
Takeaway: the **language prior helps most** (ρ 0.06→0.095 within); distinctiveness alone
barely moves ρ. Best 4AFC ~35 — just above the unweighted region baseline (34.8), still
far below oracle 49.9. **Still no ignition, but the LANGUAGE side carries more
bootstrappable signal than vision.** → E9 cross-situational prototypes [Batch 3, running].

Batch 3 = cross-situational prototype (E9): accumulate a word→region prototype across
situations; alignment = does the pair's region match the word's accumulated prototype?
Variants: proto, proto+lang, across/within. If this also fails, the strong conclusion is
that no instantaneous/accumulated self-supervised score separates aligned pairs at this
noise floor — motivating curriculum (E5) or minimal external cues (social/prosodic, which
children actually have).

---

# Chapter 5 follow-up: social cues + information titration (2026-07-02)

Two tracks in response to "the alignment floor is too low to bootstrap — can social
information (hands/faces/gaze) + a titration of how-much-info-we-need get it off the ground."

## Track B — TITRATION: how much aligned-ness information does the loop need to ignite?
`train_region_mil.py --titrate-rho R --titrate-cov C --prior-mode {fixed,gate,seed,blend}`
injects a synthetic cue = Gaussian-copula corruption of the held-out CLIP truth with target
Spearman R on a fraction C of pairs. Plain region-MIL boot recipe, across-140k, so the cue
is the ONLY added signal. Validated: --prior-mode gate --titrate-rho 1.0 reconstructs the
CLIP filter exactly (precision_vs_clip = 1.0).

**Soft-weighting (fixed): a graded prior barely helps even at perfect quality.**
| target ρ | 0.0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 1.0 |
| 4AFC (mean 2 seeds) | 33.7 | 33.7 | 35.5 | 34.5 | 35.5 | 36.2 | 37.5 | 37.8 |
A cue with ρ=1.0, used as a per-pair soft weight, reaches only 37.8 — nowhere near oracle
49.9. So *how* the cue is used dominates *how good* it is. Coverage layout (sparse-strong
vs dense-weak, matched info) is irrelevant in soft mode (all 35–36). Seed mode (cue weights
warmup, then endogenous EM) decays back to ~34.8 = baseline: **EM does not amplify a toehold.**

**Hard-gate (gate, keep top 12% as positives — the faithful analog of the ch.3 filter):**
| target ρ | filter precision vs CLIP | 4AFC (mean 2 seeds) |
| 0.1 | 0.155 | 32.5 |  (below baseline — training on a near-random subset hurts)
| 0.2 | 0.195 | 34.1 |
| 0.3 | 0.240 | 34.6 |
| 0.5 | 0.347 | 38.0 |
| 0.7 | 0.490 | 39.5 |
| 1.0 | 1.000 | 41.7 |  (this regime's ceiling; < plain-oracle 49.9 due to floor=0.05 on negatives + boot regime)

**IGNITION THRESHOLD ≈ ρ 0.3–0.5** (filter precision ~0.25–0.35, i.e. 2–3× base-rate
enrichment over the 12% floor). Below ρ≈0.3 a cue is worthless or actively harmful; you need
ρ≳0.4 for a clear lift. This is the quantitative topline: **a bootstrappable cue must reach
ρ≳0.3–0.4 / precision≳0.25–0.35.**

## Track A — CUE AUDIT: does any machine-readable social cue clear that bar?
`cue_audit.py` builds per-pair cues on across-140k and reports ρ(cue, clip_score_max) +
prec@10% (same yardstick as E0–E9). Sources: YOLOE 'cdi' detections (37.6% pair coverage),
head-stability = temporal cosine of adjacent 1fps DINOv2 CLS embeddings (69% coverage).
| cue | ρ vs CLIP | prec@10% (base 0.119) |
| hand present / count / conf / hand-obj contact | ~0.005 | 0.125 |
| person present / area | −0.04 / −0.00 | 0.12 |
| person count / n_obj / n_det | −0.06 / −0.06 / −0.10 | 0.10 |
| head stability | **−0.156** | 0.072 |
| combined logistic (det subset) | **0.136** (AUC 0.539) | 0.124 |
**Hands carry ~zero alignment signal** — hand presence is near-ubiquitous in infant
egocentric video, so it can't discriminate; hand-object contact from coarse boxes at 1fps is
too crude. Only real signal is head-stability, and it's NEGATIVE (stiller world → less
aligned) and weak. Combined real-cue ρ=0.136 does edge above the best endogenous signal
(0.095) — real cues beat self-supervision — but sits WELL BELOW the ρ≈0.3–0.4 ignition
threshold. **Verdict: box/stability cues deliver ~⅓–½ of the information needed to ignite a
filter. A ~2–3× gap remains.**

## Where the missing information likely is: POSE (directional joint attention)
Boxes answer "is a hand present"; the joint-attention literature says what matters is the
*direction* of a point/gaze *toward the named referent* — which boxes can't represent.
Full 133-kpt COCO-WholeBody poses exist (babyview-project/pose-detection):
/ccn2/dataset/babyview/outputs_20250312/pose/4M_frames_old/ (pickles + 4M_with_NA_bbox_limbs.csv,
face+hand+body limb boxes & scores). Gives face landmarks (head-orientation/gaze-cone proxy)
and shoulder-elbow-wrist+hand kpts (pointing vector). **BLOCKER:** pose uses gcp-name hashes
(`00370001_2024-09-27_1_d1010cd9a9`, older frame pull) vs our Airtable `rec` ids; the
child_date_session prefix is NOT unique (rotated/multi-file). Need a rec-id↔gcp-name crosswalk
(babyview metadata, `superseded_gcp_name_feb25`) — ask Mira. Then join on (video, second).
Plan: recompute directional cues (gaze cone hit on a salient region; wrist-vector pointing;
face-toward-object), audit ρ vs CLIP; if any clears ρ≈0.3, plug in as gate prior on the
winning recipe (region-MIL + lang-prior + curriculum).

Code: src/cue_audit.py, src/run_titration.sh, src/run_titration_gate.sh, train_region_mil.py
(--titrate-*/--prior-mode/--gate-frac). Runs: T_q_*/T_seed_*/T_cov_*/G_r*/ on ccn2.

### Track A addendum — PROSODY (128,808 utts, 900 sampled videos; ffmpeg+scipy, no librosa)
| cue | ρ vs CLIP | prec@10% (base 0.119) |
| f0_range (pitch dynamics) | 0.107 | 0.177 |
| rms_range (energy dynamics) | 0.103 | 0.152 |
| f0_std / rms_cv / cent_std | 0.06–0.10 | — |
| dur (utterance length) | 0.168 | 0.221 | ← CONFOUNDED: clip_score_max is a MAX over the
|   |   |   | utterance's frames, so longer utts get a higher max mechanically. Discount.
| combined prosody logistic | **0.147** (AUC **0.607**) | — |
Prosody is the **best cue CHANNEL**: all cues POSITIVE (vs vision ~0/negative), AUC 0.607 >
vision's 0.539. The honest prosodic signal (pitch/energy emphasis, child-directed-speech
proxy) is ρ≈0.10 — an INDEPENDENT, language-side channel, ~ the best endogenous signal
(0.095). Consistent with the recurring language>vision theme. **But still << the ρ≈0.3–0.4
ignition bar.**

### COMBINED VERDICT (Tracks A+B)
No single available cue channel reaches the ignition bar alone: vision boxes ρ≈0.14 (hands 0,
stability −0.16), prosody ρ≈0.15 (emphasis ~0.10). Titration says need ρ≳0.3–0.4 (hard gate).
Open question worth the pose work + a final "stack everything" run: the channels look PARTLY
INDEPENDENT (prosody = language-side AUC 0.607; vision = AUC 0.539), so a multichannel
classifier (prosody emphasis + pose-directional + endogenous language-prior) MIGHT stack
toward 0.3 even though no single channel gets there. That is the concrete next experiment
once the pose crosswalk (see above) is in hand.

---

# Ceiling decomposition + pose cues (2026-07-02, session 2)

## Per-category oracle: why 4AFC is "squished" at ~50
`eval_per_category.py` on the region-MIL oracle (P1_oracle_across). Per-category accuracy is
a BROAD 10-92% distribution, not bimodal: plant 92/car 84/window 84/couch 84 high; sky 10/
present 12/purse 13/cat 14 at-or-below chance. Spearman(acc, times-word-said-in-aligned)=0.22
(necessary, not sufficient): "can" said 1050x -> 28 (modal vs container polysemy), "cat" 92x
-> 14 (visually hard). Only 2/61 categories never said. Words said >=50x still avg only 46.
So the ceiling isn't "half-learn every word"; it's some words learned well + a tail of
visually-hard / lexically-ambiguous ones. oracle_per_category.csv has the table.

## Head-noun ceiling test: text cleaning does NOT raise the ceiling
Is the ~50 oracle capped by weak text (bag-of-words real utterances) vs clean labels (Ch3
topline ~72)? Built aligned oracle set (110k pairs, clip>0.24, excl held child, >=1 noun)
two ways on IDENTICAL pairs: text=full utterance vs text=NOUNS only (corpus noun lexicon,
surface forms predominantly NOUN/PROPN; utterance_id does NOT align across pipelines so used
a lexicon not a join). build_headnoun.py.
| text | 4AFC |
| full utterance | 50.05 |
| nouns only     | 50.15 |
**Identical.** Stripping function words/verbs does nothing — the BoW tower already ignores
them contrastively. So the ~50 ceiling is NOT function-word dilution. Remaining gap to the
72 label-topline is (a) noun-referent correspondence (even CLIP-aligned utterances don't
reliably name the visible object) + (b) visually-hard categories, NOT text verbosity. (Caveat:
72 was a different eval setup; the robust finding is noun-cleaning doesn't move the oracle.)

## Pose directional cue audit: no better than boxes (vs CLIP)
Full 133-kpt COCO-WholeBody poses (feb25 pull) validated & joined (55.4% of pairs; second N =
frame_idx N; pose dense). pose_lib.py (CPU-remap unpickler for CUDA-tensor pickles), reader
numerically validated (kpts in-frame 99.7%, face-in-bbox 85%, shoulder/bbox 0.46), skeleton
overlays in runs/pose_sanity/ for HUMAN review. pose_cue_audit.py, largest person, 31.7k
frames w/ a usable person:
| cue | rho vs CLIP |
| face_score / face_frontal / face_symmetry | -0.01 / -0.02 / 0.02 |
| head_pitch (gaze down) | -0.03 |
| arm_reach (pointing) | -0.02 |
| wrist_above_hip (showing) / hand_centrality | -0.03 / -0.00 |
| person_area / n_person | 0.05 / -0.05 |
| combined pose logistic | rho 0.044, AUC 0.522 |
**Directional pose cues carry ~0 signal vs CLIP — WORSE than presence-boxes (0.136/0.539),
far below the rho~0.3 bar.** CAVEATS before concluding pose is dead: (1) yardstick is CLIP
(whole-frame img-text), not TRUE reference — a caregiver pointing at a small named object is
a great referential moment CLIP may score low; (2) egocentric "largest person" may be the
child's own hands/body, not the caregiver; (3) crude cues — real gaze/point VECTOR-to-referent
(needs the referent location + 68 dense face landmarks fit) not computed. Sharper follow-up
before writing pose off: gaze/point vector to a salient region, evaluated vs a held-out
referent signal rather than CLIP.

Code: eval_per_category.py, build_headnoun.py, pose_lib.py, sanity_pose.py, pose_cue_audit.py.
