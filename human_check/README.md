# Human check of the Gemini referential-alignment annotation

Lab members rate 1,500 (frame, utterance) pairs *blind*, answering the same two questions Gemini
answered, so the SI can report how accurate the paper's `alignment >= 50` rule and its referent
labels are. Design rationale: `notes/HANDOFF_GEMINI_CHECK_APP.md`.

## Pieces

| step | script | runs on | output |
|---|---|---|---|
| 1 sample | `sample.py` | ccn2-14 | `/data2/mcfrank/gemini_check/sample.parquet` (+ `strata.csv`) |
| 2 frames | `pull_frames.py` | ccn2-14 | `/data2/mcfrank/gemini_check/data/{frames/<id>.jpg, items.json}` |
| 3 app | `app/` (FastAPI + one HTML page) | Cloud Run (IAP) **or** ccn2-14 via ssh tunnel | `data/responses/<rater>/<item>.json` |
| 4 numbers | `analyze.py` | ccn2-14 (needs sample.parquet + responses) | `results/gemini_human_check*.csv` |

Steps 1-2 are done (2026-09-09, seed 20260909; extended 2026-09-10 with `--extend`, which keeps
the existing items and ids and draws only the shortfall). The venv is `/data2/mcfrank/gemini_check/venv`.

## Sample

Population = final corpus (`bv26_pairs_en_audio`, 1.69M) joined to the Gemini table on
(video_id, frame_idx, text), scored, 2-20 tokens: 1.31M pairs. Eight strata partition it by
Gemini score, so the weights (`n_pop/n_sample`) turn sample precision/recall into corpus-level
estimates of the >=50 rule. The pool is half zeros by design: the first draw (300 zeros / 750
aligned) taught raters that "yes" was usually right.

| stratum | n | pop share |
|---|---|---|
| 0, utterance has a concrete object noun (CDI object cats + Konkle) | 450 | 17% |
| 0, other | 300 | 72% |
| 1-49 | 50 | 0.2% |
| 50-60 | 150 | 2.4% |
| 70-75 | 150 | 2.0% |
| 80 | 150 | 3.0% |
| 90-95 | 100 | 1.8% |
| 100 | 150 | 2.0% |

Constraints: <=45 per child (47 children used), one item per video-minute, one per video per
stratum. 1,500 items in all. Items carry opaque random ids; `items.json` holds only id + utterance text.

## The app

Rater sees one frame (512 px longest edge as stored; Gemini saw the same frame downsized to 512)
and the utterance, and answers **yes / no / can't rate** (keys J / F / B; Z or ← goes back).
"Yes" is pinned to Gemini's 50-point anchor: the object is in view even if small, partial, or one
of many, so the human judgment is the binary the paper's `>= 50` rule makes, and the analysis
sweeps the threshold. Before 2026-09-11 the format was three-level (none / present-but-small /
clearly present) plus a free-text referent; those ratings map onto the binary and the referent
analysis uses only them. Six conventions, read off Gemini's own outputs and its "actually
VISIBLE" instruction, are in the instructions and repeated under every item: only this frame
(the camera-wearer and off-screen things are not visible); people count when in view; names and
nicknames refer to what they name; sound words and pronouns count when the thing is in view;
pictures count when the utterance names what is pictured (flipped 2026-09-11 from "write book":
Gemini itself is inconsistent here, and a pictured elephant named "elephant" is the pairing the
paper cares about). The check therefore measures agreement with Gemini's operative definition,
which the SI should say.

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

Pull responses back for analysis with `gcs_sync.py pull`. `responses_v1/` in the bucket (and on the
node) holds Mike's first 126 ratings, made before the conventions were added on 2026-09-10, and
`responses_v2_book/` the 39 ratings made under the old "write book" rule; both are excluded.

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

Without the last binding, IAP lets people sign in but Cloud Run answers 403.

**Bucket-mount performance.** Every file read on the mounted bucket is a GCS request. The app
therefore keeps responses in memory (loaded once, updated on each save, listing rescanned every
2 min for other instances' writes). Before that, `/api/state` reread every response file per
call and took 1-2 minutes once one rater had finished all 1,500 items (2026-09-11), which
new raters experienced as a dead page.

**Only Stanford Google accounts can sign in.** IAP for Cloud Run uses a Google-managed OAuth
client, which admits only identities inside the project's organization (stanford.edu); a granted
gmail.com address gets a Google sign-in error, not the app. Grant people's @stanford.edu
addresses, or open it up (done 2026-09-09 for Bria): a custom OAuth client
(`gcloud iap oauth-brands create` + `gcloud iap oauth-clients create`, attached with
`gcloud iap settings set --resource-type cloud-run --service gemini-check`) plus the consent
screen's audience set to **External** and the app published, which only the Cloud Console can do
(Google Auth Platform > Branding / Audience). Test-mode External apps admit only listed test users. `deploy.sh` needs
gcloud >= 584 for `--iap` (552 lacked it; `gcloud components update`).

*B. ssh tunnel to ccn2-14* (works today, no GCP step): `bash human_check/run_node.sh` on the
node, then each rater runs `ssh -L 8501:localhost:8501 ccn2-14` and opens http://localhost:8501
(they type a name; use the same name to resume).

## Analysis

`analyze.py` writes `results/gemini_human_check.csv` (long: metric, value, n, bootstrap CI),
`results/gemini_human_check_calibration.csv` (per stratum) and
`results/gemini_human_check_threshold.csv` (precision / recall / F1 of Gemini >= t, sample and
corpus-weighted, for t = 50..100). Metrics: Krippendorff's alpha and
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
