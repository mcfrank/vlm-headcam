# Human check of the Gemini referential-alignment annotation

Lab members rate 1,050 (frame, utterance) pairs *blind*, answering the same two questions Gemini
answered, so the SI can report how accurate the paper's `alignment >= 50` rule and its referent
labels are. Design rationale: `notes/HANDOFF_GEMINI_CHECK_APP.md`.

## Pieces

| step | script | runs on | output |
|---|---|---|---|
| 1 sample | `sample.py` | ccn2-14 | `/data2/mcfrank/gemini_check/sample.parquet` (+ `strata.csv`) |
| 2 frames | `pull_frames.py` | ccn2-14 | `/data2/mcfrank/gemini_check/data/{frames/<id>.jpg, items.json}` |
| 3 app | `app/` (FastAPI + one HTML page) | Cloud Run (IAP) **or** ccn2-14 via ssh tunnel | `data/responses/<rater>/<item>.json` |
| 4 numbers | `analyze.py` | ccn2-14 (needs sample.parquet + responses) | `results/gemini_human_check*.csv` |

Steps 1-2 are done (2026-09-09, seed 20260909). The venv is `/data2/mcfrank/gemini_check/venv`.

## Sample

Population = final corpus (`bv26_pairs_en_audio`, 1.69M) joined to the Gemini table on
(video_id, frame_idx, text), scored, 2-20 tokens: 1.31M pairs. Eight strata partition it by
Gemini score, so the weights (`n_pop/n_sample`) turn sample precision/recall into corpus-level
estimates of the >=50 rule:

| stratum | n | pop share |
|---|---|---|
| 0, utterance has a concrete object noun (CDI object cats + Konkle) | 200 | 17% |
| 0, other | 100 | 72% |
| 1-49 | 50 | 0.2% |
| 50-60 | 150 | 2.4% |
| 70-75 | 150 | 2.0% |
| 80 | 150 | 3.0% |
| 90-95 | 100 | 1.8% |
| 100 | 150 | 2.0% |

Constraints: <=30 per child (46 children used), one item per video-minute, one per video per
stratum. Items carry opaque random ids; `items.json` holds only id + utterance text.

## The app

Rater sees one frame (512 px longest edge as stored; Gemini saw the same frame downsized to 512)
and the utterance. Q1 three buttons matching the prompt's anchors (none / present-but-small /
clearly present, plus can't-tell); Q2 free-text noun when Q1 != none. Keys 1/2/3/0, Enter, `b`.

**Resumable / multi-rater.** Nothing is stored in the browser except the rater's name. Every
answer is one JSON file on the server, so a rater can close the tab and continue from any
device; going back lets them fix the previous item. Items are served fewest-ratings-first
(tie-break: per-rater seeded permutation), so however many people rate, coverage stays even and
the target of 3 ratings/item fills uniformly. `/api/progress` shows counts.

**Two deployments, same code.**

*A. Cloud Run behind Identity-Aware Proxy* (shareable; raters need only a Google account
that Mike grants, no cluster access). Frames + responses sit in a private bucket in the
`hs-hs-langcog-gemini` project (the project already used for the Vertex annotation; Stanford-
controlled). The app reads the bucket as a mounted volume; IAP identifies the rater from their
Google login. One-time, from a laptop:

```bash
gcloud auth login
bash human_check/deploy.sh setup          # bucket, APIs, service-account grants
# on ccn2-14:
GOOGLE_APPLICATION_CREDENTIALS=~/.secrets/vlm-headcam-sa.json \
  /data2/mcfrank/gemini_check/venv/bin/python human_check/gcs_sync.py push \
  --data /data2/mcfrank/gemini_check/data --bucket hs-hs-langcog-gemini-gemini-check
# back on the laptop:
bash human_check/deploy.sh deploy         # prints the https URL
bash human_check/deploy.sh grant alice@stanford.edu bob@stanford.edu
```

Pull responses back for analysis with `gcs_sync.py pull`.

Deployed 2026-09-09: service `gemini-check` (us-central1), bucket
`gs://hs-hs-langcog-gemini-gemini-check` (1,051 objects), URL
https://gemini-check-246740721864.us-central1.run.app (IAP active). The deploying account is
Editor + Project IAM Admin on the project, not Owner, so two grants had to be done by hand once:

```bash
gcloud projects add-iam-policy-binding hs-hs-langcog-gemini --member user:mcfrank@stanford.edu --role roles/run.admin
gcloud projects add-iam-policy-binding hs-hs-langcog-gemini --member user:mcfrank@stanford.edu --role roles/iap.admin
gcloud run services add-iam-policy-binding gemini-check --region us-central1 \
  --member serviceAccount:service-246740721864@gcp-sa-iap.iam.gserviceaccount.com --role roles/run.invoker
```

Without the last binding, IAP lets people sign in but Cloud Run answers 403. `deploy.sh` needs
gcloud >= 584 for `--iap` (552 lacked it; `gcloud components update`).

*B. ssh tunnel to ccn2-14* (works today, no GCP step): `bash human_check/run_node.sh` on the
node, then each rater runs `ssh -L 8501:localhost:8501 ccn2-14` and opens http://localhost:8501
(they type a name; use the same name to resume).

## Analysis

`analyze.py` writes `results/gemini_human_check.csv` (long: metric, value, n, bootstrap CI) and
`results/gemini_human_check_calibration.csv` (per stratum). Metrics: Krippendorff's alpha and
mean pairwise kappa (3-level and binary); Spearman Gemini-vs-human; kappa, agreement, and
sample + corpus-weighted precision/recall of the >=50 rule against the human majority;
calibration (human mean by Gemini bin); false-negative rate among Gemini-0 items with/without a
concrete noun; referent match among items both call aligned (exact / lemma / WordNet synonym if
nltk is available); per-child spread of agreement. A rater table with names is written next to
the responses dir, not into the repo.

## Privacy rule

Frames are headcam images of children and families: cluster or Stanford GCP only. Never commit
frames, `sample.parquet`, `items.json`, or response files. Only the scripts and the aggregate
CSVs live in this repo.
