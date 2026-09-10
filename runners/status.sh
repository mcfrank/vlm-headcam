#!/bin/bash
# One-command status for every queued job. Read-only; safe any time.
#     ssh ccn2-14 'bash /data2/mcfrank/vlm-headcam/status.sh'
cd /data2/mcfrank/vlm-headcam 2>/dev/null || exit 1
echo "=============== $(date '+%Y-%m-%d %H:%M') on $(hostname -s) ==============="
echo "--- pipelines ---"
for j in run_phase5 run_2026_pipeline run_layer_pilot; do
  pgrep -f "$j.sh$" >/dev/null && s="RUNNING" || s="not running"
  printf "  %-22s %s\n" "$j" "$s"
done
echo "--- last log line ---"
for f in phase5 bv2026 layer_pilot; do
  [ -f /data2/mcfrank/$f.log ] && printf "  %-12s %s\n" "$f" "$(tail -1 /data2/mcfrank/$f.log | cut -c1-95)"
done
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "--- runs completed ---"
for p in P5_ B26_ S1_; do
  n=$(ls -d runs/${p}* 2>/dev/null | wc -l); m=$(ls runs/${p}*/metrics.json 2>/dev/null | wc -l)
  printf "  %-5s %3d run dirs, %3d with metrics.json\n" "$p" "$n" "$m"
done
echo "--- 2026.1 inputs ---"
for d in /ccn2b/dataset/babyview/2026.1/extracted_frames_1fps \
         /ccn2b/dataset/babyview/_mcfrank_2026.1_staging/extracted_frames_1fps; do
  [ -d "$d" ] && printf "  %-66s %s dirs\n" "$(basename $(dirname $d))/$(basename $d)" "$(timeout 25 ls $d 2>/dev/null | wc -l)"
done
[ -f manifests/bv2026_pairs.parquet ] && echo "  pairs manifest: BUILT" || echo "  pairs manifest: not yet"
if [ -f scored/bv2026_gemini.parquet ]; then echo "  gemini 2026.1 : DONE"
else echo "  gemini 2026.1 : $( [ -f scored/bv2026_gemini.jsonl ] && wc -l < scored/bv2026_gemini.jsonl || echo 0 ) pairs scored"; fi
ls -d emb_dv3_2026_* emb_s1 2>/dev/null | sed 's/^/  cache: /'
echo "--- disk ---"; timeout 20 df -h /data2 /ccn2b 2>/dev/null | tail -2 | sed 's/^/  /'
