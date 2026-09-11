#!/usr/bin/env bash
# Deploy the check app to Cloud Run behind Identity-Aware Proxy, so lab members reach it with
# their Google account and no cluster access. Frames + responses live in a private GCS bucket in
# the same Stanford-controlled GCP project that already holds the Vertex annotation work.
#
# One-time, from a laptop with `gcloud auth login` done (needs Cloud Run Admin, IAP Policy Admin,
# Storage Admin on the project):
#   bash human_check/deploy.sh setup                   # bucket + APIs + SA grant
#   (on ccn2)  gcs_sync.py push ...                    # frames + items.json into the bucket
#   bash human_check/deploy.sh deploy                  # build + deploy the service
#   bash human_check/deploy.sh grant a@stanford.edu b@stanford.edu ...   # who may open it
# Re-run `deploy` after code changes; `grant` any time.
set -euo pipefail
PROJECT=${PROJECT:-hs-hs-langcog-gemini}
REGION=${REGION:-us-central1}
SERVICE=${SERVICE:-gemini-check}
BUCKET=${BUCKET:-${PROJECT}-gemini-check}
NODE_SA=vlm-headcam@hs-hs-langcog-gemini.iam.gserviceaccount.com   # uploads from ccn2
cd "$(dirname "$0")"

case "${1:-}" in
  setup)
    gcloud config set project "$PROJECT"
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com iap.googleapis.com \
        artifactregistry.googleapis.com storage.googleapis.com
    gsutil ls -b "gs://$BUCKET" >/dev/null 2>&1 || gsutil mb -p "$PROJECT" -l "$REGION" -b on "gs://$BUCKET"
    gsutil iam ch "serviceAccount:$NODE_SA:roles/storage.objectAdmin" "gs://$BUCKET"
    PN=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
    gsutil iam ch "serviceAccount:$PN-compute@developer.gserviceaccount.com:roles/storage.objectUser" "gs://$BUCKET"
    echo "bucket gs://$BUCKET ready; now push data from ccn2 with gcs_sync.py"
    ;;
  deploy)
    gcloud run deploy "$SERVICE" --source app --project "$PROJECT" --region "$REGION" \
        --no-allow-unauthenticated --iap --execution-environment gen2 \
        --add-volume "name=data,type=cloud-storage,bucket=$BUCKET" \
        --add-volume-mount "volume=data,mount-path=/data" \
        --set-env-vars DATA_DIR=/data,RATERS_PER_ITEM=3 \
        --memory 512Mi --min-instances 1 --max-instances 2 --concurrency 20
    PN=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
    gcloud run services add-iam-policy-binding "$SERVICE" --project "$PROJECT" --region "$REGION" \
        --member "serviceAccount:service-$PN@gcp-sa-iap.iam.gserviceaccount.com" --role roles/run.invoker
    gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)'
    ;;
  grant)
    shift
    for u in "$@"; do
      gcloud iap web add-iam-policy-binding --project "$PROJECT" --region "$REGION" \
          --resource-type cloud-run --service "$SERVICE" \
          --member "user:$u" --role roles/iap.httpsResourceAccessor
    done
    ;;
  *) echo "usage: deploy.sh setup | deploy | grant EMAIL..."; exit 1 ;;
esac
