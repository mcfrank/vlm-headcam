#!/bin/bash
set -u
W=/data2/mcfrank/vlm-headcam
until [ -f $W/runs/BATCH2_DONE ]; do sleep 30; done
echo "[master2] batch2 done; launching batch 3 (proto)"
bash $W/run_batch3.sh
touch $W/runs/MASTER2_DONE
