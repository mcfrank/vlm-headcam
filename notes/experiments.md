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

## Sharper pose cue: child/adult hand split + CLIP-independent filter test (DECISIVE NULL)
Child's-own vs adult hands ARE separable (characterize_persons.py): orphan hands (no attached
face/torso, bbox on frame bottom, wrist y~0.85) = child; hands on a visible face/torso
(y~0.52) = adult. But split still null vs CLIP (child_hand rho -0.007, adult -0.025; mean clip
identical w/ vs w/o child hand). THREE cue families now orthogonal to CLIP -> tested the
"CLIP is blind" hypothesis directly: use each cue as a TRAINING FILTER, measure downstream
4AFC vs a clip>0.24 positive control (all matched 8k from the pose-covered pool, 3 seeds):
| training filter | 4AFC (mean±sd) |
| CLIP-aligned (control) | 37.9 ± 0.6 |
| random | 28.5 ± 0.3 |
| child-hand present | 27.8 ± 1.0 |
| adult-face present | 27.3 ± 1.2 |
CLIP control +9.4 over random (test has power); child-hand & adult-face = random within noise.
**REJECTS "CLIP is blind": pose cues genuinely don't select better learning moments, not just
invisible to CLIP.** Scope: this is a null for COARSE presence-style cues at 1FPS (utterance-
midpoint frame). Pointing/gaze are sub-second; 1fps sampling misses the gesture. Reference is
socially cued but the cue lives below this data's temporal resolution. => pose RERUN not
justified for this project (would yield more of a null signal). Code: pose_cues_full.py,
build_pose_filter_manifests.py, PF_* runs.

## FTF-2013 rescue: language cues weak-real, vision cues null, pointing undetectable
Motivated by Frank/Tenenbaum/Fernald 2013 (Table 3: pointing precision .78 but recall .10;
child-eyes F .54; discourse F .53) — we had tested the LOW-precision presence cues, not these.
New cues, each per-pair for the filter->4AFC test (CB_* runs; pool = region-cache-restricted,
40.6% aligned, so compressed headroom):
- prosody rms_range rho +0.18 (best single), discourse continuity rho +0.09, frame-center
  (center-vs-periphery DINOv2, the "child eyes = frame center" proxy) rho -0.06 (does NOT
  capture gaze), pose/pointing ~0. COMBINED CV-logistic rho=0.213, AUC 0.619 (beats best
  single -> cues DO combine, but < 0.3 bar).
Filter->4AFC (3 seeds, mean±sd):
| filter | 4AFC | aligned% |
| CLIP-aligned (control) | 37.5 ± 0.8 | 100% |
| discourse | 33.5 ± 0.6 | 51% |
| prosody | 33.5 ± 2.0 | 52% |
| combined | 33.0 ± 0.4 | 55% |
| random | 32.1 ± 0.7 | 41% |
Language cues beat random by ~1-1.5 (discourse steadiest, +1.4 ~2sigma; prosody noisy);
combination does NOT add (FTF redundancy reproduced); all ~1/4 of the CLIP control's +5.4.
Weak-real, but nowhere near ignition.
POINTING: sparsity mapped (extended straight arm ~11%, ~ FTF recall .10) BUT viewing strict
candidates (local pull, authorized) shows the detector catches caregivers LEANING/reaching
toward the child, not distal points; index-finger tell undetectable at arm's length (L3=0).
=> pointing is NOT recoverable from 1fps pose (detection limit, not informativeness).
VERDICT: the rescue lives on the LANGUAGE side (prosody, discourse: weak but real, consistent
with the project-wide language>vision asymmetry); vision-side social cues (pose, pointing,
gaze-as-center) are null or undetectable at 1fps. Code: prosody_for_pairs / discourse_for_pairs
/ pose_pointing / frame_center_for_pairs / build_combined_cues.

## Bootstrap-with-cue (BX): language priors folded into the EM E-step
train_region_mil.py --ext-prior <parquet> --ext-col <col> blends an external per-pair cue
into the boot weight: combined=0.5*(rank01(model_s)+rank01(cue)) -> gmm2. Region-MIL boot,
across-140k:
| E-step | best 4AFC | rho(w,clip) |
| plain (baseline) | 34.3 | 0.045 |
| + discourse (cont_share) | 35.5 | 0.045 |
| + prosody (rms_range) | 35.0 | 0.065 |
| + language prior (ch4) | 35.0 | 0.08 |
Discourse prior best (+1.3 over plain), ~ the lang prior. Small, consistent, but no ignition
(rho<=0.08, «oracle 49.9; still < curriculum 37.6). Confirms: language cues weakly help the
bootstrap; nothing ignites. Refinement TODO (Mike): "looked-at/held -> talked-about" cues are
too crude — child_hand is PRESENCE not hand+object-at-location; frame-center is center-vs-edge
CONTRAST not object-at-center. Build object-at-hand (held) + object-at-center (looked-at) via
the DINOv2 region grid + temporal persistence, rerun filter test.

## Region prior (child gaze = center) + utterance-cue matrix (design from Mike)
Added model.region_prior (biases WHICH region wins the MIL argmax; scored with raw sim, no
logit inflation) + speaker diarization (LENA FEM/MAL/KCHI/OCH via time-overlap join).
CENTER region prior (child-gaze proxy) HURTS the oracle monotonically: delta 0/.05/.1/.2 ->
50.0/47.3/46.7/45.8. Confirmed clean (Option A). The referent is NOT at frame center; the
learned MIL similarity already selects better than a positional bias. => static positional
region priors don't help; deprioritize per-frame hand-location priors (same override mech).
Utterance cues (rho vs clip, on their coverage): prosody rms_range ~0.18, caregiver-speech
(FEM/MAL) 0.08 (caregiver 43% aligned vs child 38%), discourse cont_share 0.09, child_hand ~0,
caregiver-close (adult size upper-frame) -0.01 (NULL). Boot ext-prior (across-140k): caregiver
35.6 (best), discourse 35.5, prosody 35.0, lang 35.0 vs plain 34.3 -- all small, no ignition.
Full independent+combined matrix (BXM_*) running.

### Utterance-cue matrix — SEEDED (the honest verdict)
On the TRUE across-140k pool (12.9% aligned; NOT the enriched region-cache pool where cues
looked ~2-4x stronger), single-cue rho vs clip is only 0.02-0.05. Boot ext-prior, 3 seeds:
| E-step | 4AFC mean±sd | vs plain |
| plain | 34.4 ± 0.6 | — |
| + caregiver speech | 34.5 ± 1.0 | +0.1 (noise; single-seed 35.6 was a lucky draw) |
| + child-hand | 35.2 ± 1.2 | +0.8 (noisy) |
| + combined (unsup mean-of-ranks) | 35.4 ± 0.8 | +1.0 (~1.5 sigma) |
Only the COMBINED is marginally above plain; individual cues within noise. Nothing ignites
(rho(w,clip) ~0.05). CENTER region prior HURTS (50->46). Caregiver-close, child-hand, pointing:
null. VERDICT: at 1fps on the real pool the social/language cues are too weak to matter — best
combined +1 (marginal), «curriculum 37.6 «oracle 49.9. Ch5 negative stands. One principled
mechanism still untested: discourse as a STRUCTURAL prior (temporal runs of high endogenous
weight, non-cheating) — cheap, but unlikely to change the verdict given everything above.

### Discourse-structural prior (Option A) + region-prior — both null/hurt (cue investigation CLOSED)
Discourse structural prior (--discourse-runs: boost E-step weights in temporal runs of high
weight; non-cheating, FTF clumpiness) 3 seeds: 34.41±0.66 = plain 34.40±0.58 (NO effect).
Stacked on combined: 35.29±1.03 = combined 35.43±0.79 (no add). Center REGION prior HURTS
(50->46). Full verdict across utterance-weights AND region-priors, independent+combined, seeded:
best = combined utterance cues +1.0 (~1.5sigma); everything else within noise or hurts. No
social signal reaches the rho~0.3 titration bar on the true pool. Ch5 §sec-which tightened.

### SYSTEM ACCURACY WATERFALL (why the numbers are what they are; for the writeup)
4AFC held-out child, chance 25:
- 100 -> ~72: frozen DINO+BoW + clean labels + alignment GIVEN (ch3 label-topline). Loss to
  representation (frozen features can't separate all cats) + noisy YOLOE eval-gold + hard/
  polysemous CDI words. METHODOLOGICAL (our probe choices; features HAVE the info given clean align).
- 72 -> ~50: real utterances instead of clean labels (region-MIL oracle). Referential slack:
  real speech, even aligned, doesn't cleanly name the on-screen object; BoW. Mostly FUNDAMENTAL
  (+ partly BoW). (caveat: 72/50 not identical eval frame-sets.)
- 50 -> ~36: alignment DISCOVERED not given (best bootstrap). The ch5 result: co-occurrence too
  sparse + social cues too weak at 1fps. FUNDAMENTAL to this stream/tools.
- 36 -> 34/25: recovered / chance.
Not-apples-to-apples vs CVCL on EVERY axis: unfrozen-vs-frozen, single-child-vs-pooled+cross-child
eval, curated-vs-YOLOE gold, all-pairs-vs-discover-alignment. Highest-leverage fixes are the
METHODOLOGICAL leaks: (1) unfreeze vision on a winning recipe (never tried; 72 ceiling has headroom),
(2) cleaner eval, (3) real text encoder > BoW. Alignment leak (14pt) is the proven-hard one.

## WITHIN- vs CROSS-child (new ch6 direction; Vong&Lake 2026 comparison)
Top-3 kids by utts: S00400001(135k/126h), S00510002(119k/99h/108k reg-frames), S00320001(118k/132h).
S00510002 best resourced (95% of utts region-embedded). Within-child pipeline (build_within_child.py):
VIDEO-level TEMPORAL holdout (last 20% of videos; frame-level would leak contiguous frames);
within-child 4AFC eval from CDI detections on held-out videos, matched to the 61 cross-child CDI cats.
S00510002: 396 train / 98 test videos, 89k train pairs (15k aligned).
Results (all on the SAME S00510002 held-out matched-60cat eval):
| model | aligned train | 4AFC |
| within-child all-pairs | (89k all) | 31.2 |
| within-child oracle | 15k | 33.1 |
| pooled oracle HN_full (incl S00510002 other vids) | 110k | 41.2 |
| (HN_full on native S00360001 eval) | | 48.1 |
DIAGNOSTIC REVERSED the hypothesis: within-child is HARDER, not easier. Clean decomposition:
(1) DATA AMOUNT: pooled-110k (41) > single-child-15k (33) on the SAME eval = +8 -> Vong Finding 5
"count is king" reproduced in our pipeline; a single child lacks aligned data. (2) S00510002 eval
~7 harder than S00360001 (41 vs 48, same model). => NO cross-child penalty; POOLING HELPS.
Remaining gap to Vong's within-child 51 (we get 33) is confounded by ENCODER (our frozen generic
dinov2-base vs their unfrozen child-DINO ViT) + EVAL (YOLOE-gold temporal holdout vs curated Labeled-S).
NEXT to isolate single-child-vs-pooled at fixed data: train pooled oracle on matched 15k aligned.
Vong 2026 also rules out: Whisper fine (57 vs 62), better LM doesn't help (CVCL≈+GPT2≈+LM), more data
minimal, CLIP-L only 67% on their eval. => don't chase LM/transcripts; encoder+eval are the levers.

## Pair-count funnel vs Vong (answer: we did NOT lose pairs — we have MORE)
S00510002 funnel: Whisper utts 100.5k -> CLIP-scored 119.4k (finer segmentation) -> region-embedded
113k (95%, only 5% lost) -> aligned(clip>0.24) 18,683 (15.6%). Whole 2025.2: 1.28M utts, 165.8k
aligned (12.9%). Vong baby S: ~122k pairs, 9,320 aligned (7.6%). => our single child has 2x the
aligned pairs AND 2x the alignment rate of Vong's baby S, and we barely lose any in the pipeline.
So the 33-vs-51 within-child gap to Vong is NOT pair count. Combined with Mike's encoder skepticism
(adjusted-DINO got worse; generic dinov2 is strong), the leading suspect for the cross-paper gap is
the EVAL: our YOLOE-gold, temporal-holdout, 60-cat CDI eval (incl hard/ambiguous words) vs Vong's
curated Labeled-S (22-41 clean cats). NB: within OUR setup count still helps (pooled-110k 41 >
single-15k 33); we're just capped at a lower ceiling by the eval. Note (Mike): 2026.1 has 200+h from
several kids -> could rerun pipeline for deeper single-child work at compute cost.

## Within-child across 3 kids + DATA-MATCHED single-vs-pooled (the "whose" answer)
Within-child oracle (each on own matched-CDI-cat eval): S00510002 33.1, S00320001 29.7,
S00400001 36.1; all-pairs 31.2/30.1/34.0. Pattern holds across kids (~30-36), not S00510002-
specific. Oracle barely beats all-pairs within a single child (too few aligned: 10-15k).
DATA-MATCHED (all on S00510002 held-out eval, oracle):
| training | aligned pairs | 4AFC |
| single child (S00510002) | 15k | 33.1 |
| pooled (many kids) | 15k | 38.3 |
| pooled (many kids) | 110k | 41.2 |
=> At MATCHED 15k count, POOLING beats single-child by +5.2 (DIVERSITY), and count adds another
+2.9 (15k->110k, diminishing). So the single-child deficit is BOTH: ~⅔ diversity, ~⅓ count.
"WHOSE" matters more than "how much": a pool of many children's aligned examples per category
generalizes better (even to a single child's OWN held-out videos) than that child's own data.
Single-child consistency helped the bootstrap toehold but HURTS generalizable word learning.
Reconciles with Vong (within-child eval on same child's objects doesn't stress category diversity).

## Curated-eval test + the "no big gap" resolution
Dropped 16/61 categories on PRINCIPLED grounds (places basement/bathroom/kitchen/room/sidewalk/
sandbox; body parts hair/hand; polysemous can/present/picture/orange/mouse/train; generic toy/
person) -> 45 clean concrete-object cats. Re-eval saved models, full vs clean:
| model | full | clean |
| pooled oracle HN_full (S00360001 eval) | 48.1 (61) | 48.6 (45) |
| within-child oracle S00510002 | 33.1 (60) | 31.2 (44) |
Curation does NOT raise accuracy -> REFUTES "hard/ambiguous categories drag the number down": in
4AFC a hard category is also an easy DISTRACTOR, so removing it ~cancels. The eval-difficulty (via
category mix) is NOT the gap.
RESOLUTION: our POOLED oracle (48-50) is already ~ Vong within-child (51) [loose: different evals/
setups, but same ballpark] -> there is no large unexplained pipeline gap. The scary "33 vs 51" was
comparing our DATA-STARVED single child to Vong. The genuinely low numbers are (a) the bootstrap
not igniting (ch5, fundamental) and (b) within-child data-starvation (diversity+count, quantified).
The supervised pooled pipeline performs comparably to the field. Konkle not in wkvong/multimodal-baby
(eval code only); a true apples-to-apples cross-paper eval would need the Konkle stimulus set pulled
to ccn2 (Konkle lab / Databrary) — optional future.

## Temporal frame-sampling (Vong replication) — REFUTED
Hypothesis: Vong's unfiltered ~50 (vs our within-child ~31) comes from sampling a random frame
from each utterance's window every epoch (free within-window alignment via the memorization
effect); our fixed midpoint throws it away. Embedded 133k window frames for S00510002 (append to
emb_reg), trained within-child all-pairs with random per-epoch window sampling vs a midpoint
control on the SAME 94.6k utterances, 3 seeds, within-child matched eval:
| condition | 4AFC mean±sd |
| window sampling | 32.9 ± 0.4 |
| midpoint control | 32.0 ± 0.8 |
Only +0.9 (within noise). NOT the temporal-sampling ingredient. Likely why: our utterances are
SHORT at 1fps (mean 3.6 frames/window, many 1-2), so little temporal spread to exploit — vs
Vong's up to 16 frames from higher-fps SAYCam. The within-child ceiling is ROBUST at ~32-33
across all-pairs, oracle, window-sampling, and midpoint — nothing internal moves it; only POOLING
(diversity) reaches 48. Remaining suspects for the within-child 33-vs-Vong-50 gap: eval (YOLOE-gold
held-out-video vs curated Labeled-S) and/or encoder (generic dinov2 vs child-DINO) and/or genuine
SAYCam-vs-BabyView single-child data consistency — none isolated; encoder is Mike-doubted. Denser
sub-1fps frame extraction could give a bit more spread but unlikely to close 17 points.

## KONKLE EVAL — the eval WAS the gap (+24 points)
Vong supplement S/A.2 + Appendix D: their Konkle eval = 60 categories (CVCL's 64-subset present
in all 3 SAYCam kids' vocab), from Konkle 2010 (200 cats, clean objects on white). Local set in
data/ObjectCategories/{N}-objects/{category}/*.jpg — all 60 Vong words map DIRECTLY to folders
(60/60). Extracted 60 cats x 17 exemplars = 1020 images -> ccn2, embedded region grid (emb_konkle),
built a Konkle 4AFC eval (embed_konkle.py, eval_model.py generalized to any cache).
| model | KONKLE 4AFC | our CDI-YOLOE 4AFC |
| pooled oracle HN_full | 72.4 (58/60 cats) | 48.1 |
| within-child oracle S00510002 | 44.9 (36/60) | 33.1 |
Pooled model jumps +24 on the CLEAN eval -> 72.4, ABOVE Vong's ~50 and CLIP-L's 66.7, and right
on our ch3 clean-label topline (~72). => the model was never underperforming; our CDI-YOLOE eval
(noisy detector gold + held-out-video generalization + ambiguous frames) suppressed the measured
number by ~24. On the SAME eval Vong uses, we're strong. RESOLVES the whole "why are our numbers
low" thread: it was the eval, not the pipeline. (within-child 44.9 is on only 36/60 cats — single-
child vocab is smaller — so less comparable; pooled 58/60 is the clean number.)

## FULL KONKLE RE-EVAL (paired CDI / Konkle) — masked signal + starker oracle gap
Clean full-vocab set (region-MIL, across-140k, 56-60/60 Konkle cats scored):
| model | CDI-YOLOE | Konkle | note |
| boot plain (BX_plain) | 34.3 | 43.4 | baseline |
| + caregiver prior | 35.6 | 43.3 | null (both evals) |
| + prosody prior | 35.0 | 47.5 | +4 on Konkle (masked on CDI) |
| + discourse prior | 35.5 | 48.1 | +5 on Konkle |
| + language prior | 34.9 | 48.6 | +5 on Konkle |
| curriculum scratch ctrl (R4_scratch) | 35.5 | 38.7 | |
| curriculum transfer (R4_stage2) | 37.6 | 43.7 | +5 vs ctrl on Konkle (was +2 on CDI) |
| oracle (P1_oracle_across) | 49.9 | 69.3 | |
| oracle (HN_full) | 48.1 | 72.4 | |
| oracle (HN_noun) | 50.2 | 67.0 | |
FINDINGS: (1) everything rises on the clean eval; the ORACLE rises most (+20-24) so the
oracle-vs-bootstrap gap WIDENS 13->~27 -> "bootstrap doesn't ignite" is CONFIRMED, starker.
(2) MASKED SIGNAL revealed: the LANGUAGE-side priors (lang/discourse/prosody) all +4-5 over
plain on Konkle (3 cues agree) but were ~0 on CDI; curriculum +5 vs +2. The noisy CDI-YOLOE
eval compressed real language-cue signal into noise. Consistent w/ the project-wide language>
vision theme, now clearer. Vision/caregiver/pose cues still null on Konkle too.
CAVEAT (Mike): cross-paper 72 vs Vong 50 is CONFOUNDED (BabyView higher-res + full dinov2 >
their child-DINO) -> within-project comparisons only. Subset-trained models (CB/PF/WC) score
few Konkle cats (22-48/60, small vocab) -> less comparable; excluded from the clean table.

## Gemini (Vertex) referential-alignment scoring — validation
src/gemini_align.py scores (utterance,frame) pairs with Gemini on Vertex (IRB-approved;
service account on ccn2 ~/.secrets, no secrets in repo). Returns alignment 0-100 (ordinal,
anchored) + referent noun. Concurrent, resumable JSONL checkpoint. src/plot_gemini_val.py
does histograms + CLIP correlation.
VALIDATION (400 pairs from unfiltered_S00360001, same sample both models):
| model | mean | frac0 | frac100 | vs CLIP pearson/spearman |
| Flash | 9.6 | 0.88 | 0.04 | 0.26 / 0.22 |
| Flash-Lite | 9.4 | 0.86 | 0.00 | 0.24 / 0.19 |
Flash vs Lite: 0.77 / 0.73.
FINDINGS: (1) 0-100 scale fixed the float->{0,1} collapse; Flash uses full range, Flash-Lite
piles at 70 / never 100 -> USE FLASH. (2) ~88% score 0 on unfiltered = appropriately selective
(most speech not about a visible object), not rubber-stamping. (3) weak-pos corr w/ CLIP
(rho~0.22, attenuated by 88% ties at 0) = lots of INDEPENDENT signal vs CLIP. (4) disagreements
favor Gemini: catches book-reading + toy referents CLIP rates low (Llama Llama->book, Puppy->
puppy, There's a car->car), rejects CLIP keyword false-positives (abstract "high chair" talk,
deictic "get it"). Gemini leans on referent="book" in reading frames (ok for filter, weak label).
NEXT (not yet run): decisive test = score a pool, retrain at matched count on Gemini-top-k vs
CLIP-top-k, compare 4AFC + Konkle. Fig: book_figs/gemini_val.png (ccn2).

## Gemini-filter vs CLIP-filter retrain (matched pool + count)
Pool: 350k random from unfiltered_S00360001, Gemini-2.5-Flash scored (scored/pool_flash.parquet;
9.2% >=50, 6% >=80). build_gemini_arms.py -> top-N by alignment vs top-N by clip, SAME pool.
Diversity audit (N=22k): overlap only 25% (they pick different pairs); gemini arm 2409 distinct
referents / 5028 videos / 36 kids (MORE diverse, not book-collapsed); CLIP arm alignment
median=0 (>half of CLIP's top pairs have NO visible referent per Gemini). Trained whole-frame
train.py on emb_full (NB: must use emb_full not emb, else gemini arm gutted 22k->7.8k since it
rescues low-clip frames). GPU7.
| N | arm | detector 4AFC | Konkle 4AFC |
| 22k | Gemini | 38.1±1.4 | 57.3±2.3 |
| 22k | CLIP   | 41.7±0.6 | 56.5±1.9 |
| 15k | Gemini | 38.2 | 58.2 |
| 15k | CLIP   | 41.6 | 55.3 |
VERDICT: WASH. Matched pool+count, Gemini filter does NOT beat CLIP. Konkle = statistical TIE
(seeds overlap); detector favors CLIP +3.6 but that's partly circular (CLIP cosine & YOLOE both
favor big centered objects = detector-eval style; bias gone on clean Konkle -> tie). Gemini
likely cancelled by storybook confound (rescues referent=book/llama on 2D pages: referentially
right, visually unhelpful for real-object recog). Both arms ~57 < CLIP's existing best (64-67
from full-stream threshold) bc 350k random pool weaker start. Does NOT test quality-at-scale
(needs full-stream). NEXT: pivot to Gemini referent as LABEL (topline: clean labels->72), not
filter. Have referents for 350k pool free.

## Konkle-eval grid: all-data, 80/20 video split, Gemini gold (1 seed, test-60)
Train on 80% of videos (911k pairs), eval Konkle. no-MIL = whole-frame TwoTower; MIL = region-MIL.
Topline-1 = Gemini alignment>=thr filter; Topline-2 = singularized Gemini referent as label.
| condition            | no-MIL test60 | region-MIL test60 | MIL dev117 (cats) |
| baseline (911k)      | 53.1 | 63.7 | 41.6 (116/117) |
| T1 >=50 (83k)        | 62.5 | 67.2 | 58.3 (68) |
| T1 >=70 (67k)        | 59.8 | 68.1 | 58.7 (62) |
| T1 >=80 (54k)        | 60.4 | 68.2 | 57.0 (59) |
| T1 >=90 (33k)        | 61.3 | 68.5 | 64.3 (44) |
| T1 =100 (19k)        | 62.6 | 68.1 | 65.5 (31) |
| T2 labels (84k)      | 74.5 | 84.0 | 61.8 (71) |
FINDINGS: (1) MIL HELPS, doesn't hurt: +10.6 at baseline (53.1->63.7), +9.5 at label topline.
Region max-pool does implicit referent selection. (2) KEY: region-MIL baseline (63.7) already
~= whole-frame FILTER topline (62.5) -> MIL recovers most of the alignment benefit w/o any
filter. So filtering headroom on MIL is only ~+4.5 (63.7->68, flat across thresholds); LABEL
headroom is ~+20 (63.7->84). The prize is LABELS not filtering -> cue work should target
what's-named more than which-frames. (3) Topline-2 MIL=84 is the true clean-label ceiling
(55/60 cats, 5 uncovered -> real ceiling a bit higher). (4) dev-117 harder+broader (baseline
41.6); coverage drops at high thr (fewer training words overlap dev cats) so cross-thr dev
reads need matched-cat care; use test-60 for clean comparisons. CAVEAT: 1 seed (bars pending).
Setup: gemini_full (1.145M all 36 kids), emb_full (whole-frame), emb_reg (1.08M region), 
dev cats emb_konkle_dev (117). Scripts: build_grid_manifests, run_grid_wf, run_grid_mil.

## Grid with error bars (3 seeds, Konkle test-60)
| condition        | no-MIL       | region-MIL   |
| baseline (911k)  | 52.1 +/- 1.9 | 62.6 +/- 1.7 |
| T1 >=50 (aligned)| 61.6 +/- 1.2 | 68.2 +/- 1.1 |
| T2 (Gemini labels)| 75.0 +/- 0.7 | 81.3 +/- 2.6 |
CONFIRMED w/ bars: (1) MIL helps +10.5 (~5sigma), not hurting. (2) filter gain on MIL only
+5.6 (63->68, ~3-4sigma) - MIL's region-select already does implicit alignment. (3) LABEL
headroom +18.7 (63->81) is the prize. NB last turn's single-seed MIL-T2=84 was high seed;
honest = 81.3+/-2.6. Whole-frame dev-117 (s0): baseline 41.5 (116/117, harder+broader);
toplines cover fewer cats (T1=100 -> 71.1 on 31 cats) so cross-cond dev needs matched-cat care;
use test-60. STRATEGIC: cue work should target what's-named (label, +19) not which-frames
(filter, +6). Scripts: run_seeds.sh, grid_agg.py.

## EM utterance-bootstrap 2x2 (3 seeds, Konkle test-60)
| arch \ EM | -EM | +EM | effect |
| -MIL (whole-frame, emb_cls1) | 52.9+/-1.7 | 53.6+/-2.3 | +0.7 |
| +MIL (region, emb_reg)       | 62.6+/-1.7 | 63.8+/-2.3 | +1.2 |
VERDICT: utterance bootstrap (EM reweight pairs by self max-region alignment) is NULL - both
+0.7 and +1.2 within seed noise (sd~2). Firmer "doesn't ignite" than the old detector-eval
result. -MIL cells run as RegionMIL on CLS-only cache (emb_cls1), evaled on emb_konkle_cls1, so
the 2x2 is one code path. Scripts: run same train_region_mil --mode boot/plain.

## Frame-MIL 2x2 (3 seeds, Konkle test-60)
train_frame_mil.py: max over (frame x region) across +-2s window. Unions emb_reg + emb_win_0..3.
| arch \ frames | single | +frame MIL (+-2s) |
| whole-frame (cls) | 52.9+/-1.7 | 55.5+/-1.7 (+2.6) |
| region            | 62.6+/-1.7 | 65.0+/-1.8 (+2.4) |
VERDICT: frame MIL is a REAL modest free gain (+2.4 over region, positive all 3 seeds), roughly
additive w/ region's +9.7. "Which moment" helps a bit on top of "where in frame". Runs
G_framereg_s* (eval emb_konkle), G_framecls_s* (eval emb_konkle_cls1). Window frames: 1.49M new
region embeds (emb_win_0..3). Ladder now: pure 52.9 -> +region 62.6 -> +frame 65.0 (free) ->
oracle filter 68.5/word 73.2/vision 81.3. Waterfalls: make_ladder_figs.py.

## Language cues session (Konkle test-60, region-MIL)
CAREGIVER FILTER (seed 0): baseline(911k)=63.7, drop-child(770k)=63.5, random-drop-ctrl(770k)=
64.3, careg-only(518k)=64.8. VERDICT: dropping child utterances does NOT help (nochild 63.5 <=
random-drop 64.3). Child speech ~ as useful as random; Gemini alignment already handles
referentiality regardless of speaker. (~20% of pairs are child KCHI/OCH via time-overlap join to
token transcript.) filtnat-nochild 67.3 <= filtnat 68.5 too.
NOUN-BIAS word-weighting (weighted bag-of-words, static content-noun prior from spacy_pos):
on referent-bearing set, filtnat uniform 68.5 -> noun-bias best 70.6 / FINAL 68.5. VERDICT:
transient +2 but ~0 at convergence - BoW self-corrects for POS. Static cues won't recover the
+4.7; need CONTEXT-dependent cues (discourse newness, per-word prosody) that pick the referent
AMONG several content nouns. Scripts: build_speaker_manifests, build_word_prior, train_wordweight.
NEXT: per-pair-per-word weights (discourse newness = transcript; prosody = mp3 + token times).

## Word-selection cues on referent-bearing set (recover the +4.7? filtnat 68.5 -> t15 73.2)
Weighted bag-of-words (per-pair per-word pooling weights), region-MIL, Konkle test-60.
Uniform control on the SAME manifest (own text) is the valid baseline. Best-epoch, 3 seeds:
- DISCOURSE-newness (1/(1+recent count), window40): uniform 71.7 vs disc-weighted 70.0 -> HURTS -1.7.
  Newness penalizes repeated referents (caregivers repeat the named object). NULL/negative.
(noun-bias earlier: +2 transient, 0 at convergence.) Language-side word cues not recovering +4.7.
Scripts: build_cue_manifest (discourse), train_perword (per-word weighted pool). PROSODY next.

## PROSODY (per-word RMS) + language-cue session synthesis
Per-word prosodic weight = RMS energy over each word's audio span (mp3 + token times, librosa,
16-way sharded). Normalized loudest-word->1. add_prosody.py, train_perword --weight-col w_pros.
RESULT (best-epoch, 3 seeds, referent-bearing set): uniform 71.6 vs prosody 71.0 -> NULL (-0.6).

SESSION SYNTHESIS - recovering the +4.7 word-selection headroom: ALL cues NULL.
| cue | mechanism | result |
| caregiver filter | drop child utterances | null (drop-child 63.5 <= random-drop 64.3) |
| noun-bias | static POS weighting | null at convergence (+2 transient) |
| discourse-newness | down-weight repeated words | -1.7 (hurts; referents get repeated) |
| prosody | per-word RMS stress | null (-0.6, within noise) |
KEY: uniform bag-of-words at BEST epoch = ~71.6, already within ~1.6 of t15 oracle 73.2. The
+4.7 gap was FINAL-epoch (uniform overfits 71.6->68.5; oracle t15 robust). So the contrastive
objective already extracts the referent word implicitly (referent correlates w/ image across
pairs, noise words wash out); early-stopping recovers most of the +4.7; no accessible cue
recovers the residual. => word-selection is NOT a cue-accessible lever. The remaining real
headroom is the +8.1 VISION-BINDING (unspoken referents) - a vision problem, not language.
Scripts: add_prosody, build_word_prior, build_cue_manifest, train_wordweight, train_perword,
build_speaker_manifests.

## VISION cue: pose region-prior (caregiver hand) - the "obvious" literature cue
Full pose CSV (/ccn2/.../pose_1fps_bbox_limbs.csv, 6.5M rows, per-person body-part bboxes).
build_pose_targets: caregiver(largest body) hand bbox -> target grid cell (3.4M frames, 89% hand).
train_regionprior: per-pair region prior biases MIL cell-selection toward the pose cell
(select-with-prior, score-raw-sim, strength 0.5). Referent-bearing set (70% have hand target).
RESULT (best-epoch, 3 seeds): uniform 70.0 vs caregiver-hand-prior 69.8 -> NULL (-0.2).
Steering region attention to the caregiver's hand doesn't help referent id; region-MIL already
picks the best-matching cell, and on deictic pairs there's no content word to bind anyway.
(Not swept: prior strength; deictic-only subset - but pattern is clearly null.)
FINAL: every cue null - language (caregiver/noun/discourse/prosody) AND vision (pose hand). The
oracle rungs are not accessible-cue-recoverable in this frozen-feature regime. Scripts:
build_pose_targets, train_regionprior. Also: ~5G reclaimed (emb_reg_0/1, shards); /data2 shared
disk at 100% - pm5s frame expansion abandoned (won't fit).

## ch6 SCALING + WHOSE DATA (region-MIL, Konkle test-60, 3 seeds)
Subsamples of 911k train (emb_reg). A: random vs Gemini-aligned scaling. B: diversity + within-child.
RANDOM: 10k 30.0, 30k 38.3, 100k 40.2, 300k 56.2, 911k 62.6 (climbs, NO plateau -> data-limited).
ALIGNED (top-N): 10k 63.9, 30k 68.0, 85k 71.4. KEY: 10k aligned (63.9) ~= 911k random (62.6) ->
alignment ~90x data efficiency; gap ~+30 at every count. (Caveat: low-N coverage confound.)
DIVERSITY @30k: 1c 31.0, 3c 33.0, 10c 36.9, 36c 34.4 (weak/noisy). WITHIN-CHILD: biggest child
110k=35.8 vs pooled-110k=42.8 -> +7 diversity at matched count (real, moderate).
SYNTHESIS: both data- AND signal-limited; fix = concentrate aligned data from many children.
Figs: make_scaling_figs.py -> fig_scaling_curves, fig_whose_data. Scripts: build_scaling_manifests.

## Cue-vs-alignment, extended (Condition 0, full pose CSV/pkls, ch5)
Tested pose on UTTERANCE axis (not just region) + new cues, vs Gemini alignment:
- pose GESTURE (pointing=hand extended from body, showing=hand raised; from CSV bboxes): spearman
  0.028, AUC 0.526, mean-align by quartile 7.4/8.7/8.5/9.2 -> NULL (no better than presence).
- PERSON present (any detection): AUC 0.554, mean-align 8.5(present) vs 5.1(absent) -> BEST social
  cue, weak. FACE present (caregiver face_score>0.3): AUC 0.508 -> null.
- GAZE direction (caregiver head pitch/yaw from raw 133-kpt pkls, 52k pairs w/ valid face): pitch
  AUC 0.505, yaw 0.454, looking-down mean-align 6.3 vs 7.8 -> NULL (slightly negative).
VERDICT: no accessible cue predicts alignment; best=discourse rho 0.14, best-social=person-present
AUC 0.55, all << titration bar rho~0.3/AUC~0.65. Utterance filter would be ~random (AUC 0.53).
Added ch5 "Condition 0" table. Scripts: build_pose_gesture, build_pose_face, build_pose_gaze.

## "What got learned" interpretation (ch7, best organic model G_framereg_s0)
Item plot (make_item_plot): 176 Konkle cats, mean 51.9, 47% >=50%. Learned=distinctive common
nouns (apple/bike/cat/chair 100); failed=rare(bongo/trumpet)/small(bill/glove)/polysemous(frame/
grill)/diffuse(cheese/quilt). cat=100 here vs 14 on old detector eval (eval noise confirmed).
What predicts (make_item_analysis): VISION PROTOTYPE 4AFC (nearest-centroid in frozen DINOv2,
no text) = median 100, MIN 99, 100% of cats >=90% -> frozen features separate EVERY category.
model-acc vs vision-proto rho=0.03 (none); model-acc vs log-frequency rho=0.39. 0 cats vision-
limited; 111/176 learnable-by-vision-but-model-missed. => BOTTLENECK IS THE LEARNING SIGNAL
(frequency/alignment), NOT the vision encoder -> unfreezing would NOT help (refines conclusion).
Also: launched region-MIL 100% refit (held-out 20% embedded to emb_reg_ho_0..7, train_frame_mil
--window 0 on grid_baseline_full 1.14M) = top scaling point + definitive model. Book now 8 ch
(added ch7 What got learned; conclusion->ch8). Scripts: make_item_plot, make_item_analysis.

## Region-MIL 100% refit (definitive model + top scaling point)
Held-out 20% embedded (emb_reg_ho_0..7, 154k frames); train_frame_mil --window 0 on
grid_baseline_full (1.14M pairs). Region-MIL 100% = 65.6+/-2.2 (s0 68.1/s1 65.1/s2 63.7) vs 80%
911k=62.6 -> +3.0, curve still climbing (no plateau, confirms data-limited). Item plot + analysis
re-run on definitive model (G_base_mil_full_s0): vision-proto min 99 (features separate all),
freq rho 0.41 >> vision rho 0.19. ch6 scaling curve + ch7 updated to definitive model.
