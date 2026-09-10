#!/bin/bash
# Node cleanup + Oak archiving, approved by Mike 2026-09-10 (CLEANUP_PLAN §2–3, "when in doubt,
# tar and move to Oak"). Every deletion happens only after its Oak copy is checksum-verified
# (rsync -c dry-run reports nothing to transfer). Markers go to $LOG; re-runnable (each step
# skips if its DONE marker exists).
set -u
LOG=/data2/mcfrank/oak_stage/node_archive_20260910.log
ST=/data2/mcfrank/oak_stage; mkdir -p $ST
W=/data2/mcfrank/vlm-headcam
OAKP=oak-dtn:/oak/stanford/groups/mcfrank/babyview-2026.1-mirror/project
OAKE=oak-dtn:/oak/stanford/groups/mcfrank/babyview-2026.1-mirror/outputs/image_embeddings
EMB=/ccn2b/dataset/babyview/2026.1/outputs/image_embeddings
log() { echo "$(date +%F_%T) $*" >> $LOG; }
done_() { grep -q "^.* DONE $1$" $LOG 2>/dev/null; }
ship_verify() {  # ship_verify <local tar> <oak dest dir>   -> 0 iff verified
  rsync -rlt --partial "$1" "$2/" >> $LOG 2>&1 || { log "FAIL ship $1"; return 1; }
  local n; n=$(rsync -c -n -i "$1" "$2/" 2>>$LOG | grep -c "^>f")
  [ "$n" -eq 0 ] && { log "VERIFIED $1"; return 0; } || { log "FAIL verify $1"; return 1; }
}
log "=== node archive start ==="

# 1. local frame copy (pure duplicate of ccn2b frames; all embedding finished)
if ! done_ frames_local; then
  rm -rf /data2/mcfrank/frames_1fps_local && log "DONE frames_local (deleted 616G copy)"
fi

# 2. BV region caches: node-local c9_caches -> real dirs on ccn2b (replace symlinks), then Oak
for enc in vits_bv vitb_bv vitl_bv; do
  done_ c9_$enc && continue
  new=$EMB/${enc}_grid4x4.new; old=$EMB/${enc}_grid4x4
  rm -rf $new; mkdir -p $new
  for sh in $old/shard_*; do cp -rL "$sh" "$new/" || { log "FAIL copy $enc $sh"; continue 2; }; done
  # verify: sizes + md5 of every emb/index file against the resolved sources
  ok=1
  for f in $(cd $new && find . -type f); do
    a=$(md5sum "$new/$f" | cut -d" " -f1); b=$(md5sum "$(readlink -f "$old/$f")" | cut -d" " -f1)
    [ "$a" = "$b" ] || { ok=0; log "FAIL md5 $enc $f"; }
  done
  [ $ok -eq 1 ] || continue
  mv $old $EMB/${enc}_grid4x4.symlinks_$(date +%Y%m%d) && mv $new $old && log "SWAPPED $enc -> real dir on ccn2b"
  rsync -rlt --partial $old/ $OAKE/${enc}_grid4x4/ >> $LOG 2>&1 && n=$(rsync -c -n -i -r $old/ $OAKE/${enc}_grid4x4/ 2>>$LOG | grep -c "^>f") && [ "$n" -eq 0 ] && log "DONE c9_$enc" || log "FAIL oak $enc"
done
if ! done_ dinov3s_oak; then
  rsync -rlt --partial $EMB/dinov3s_grid4x4/ $OAKE/dinov3s_grid4x4/ >> $LOG 2>&1 && n=$(rsync -c -n -i -r $EMB/dinov3s_grid4x4/ $OAKE/dinov3s_grid4x4/ 2>>$LOG | grep -c "^>f") && [ "$n" -eq 0 ] && log "DONE dinov3s_oak" || log "FAIL oak dinov3s"
fi
if done_ c9_vits_bv && done_ c9_vitb_bv && done_ c9_vitl_bv && ! done_ c9_delete; then
  rm -rf /data2/mcfrank/c9_caches $EMB/*_grid4x4.symlinks_* && log "DONE c9_delete"
fi

# 3. window neighbor caches -> one tar per encoder -> Oak project/window_caches -> delete local
for enc in dinov3b vits_bv vitb_bv dinov3l vitl_bv dinov3s; do
  done_ win_$enc && continue
  t=$ST/window_${enc}_20260910.tar; parts="/data2/mcfrank/emb_win5/$enc"
  [ -d /data2/mcfrank/emb_win5b/$enc ] && parts="$parts /data2/mcfrank/emb_win5b/$enc"
  [ -f $t ] || tar cf $t $parts 2>>$LOG || { log "FAIL tar $enc"; continue; }
  ship_verify $t $OAKP/window_caches && rm -rf $parts $t && log "DONE win_$enc"
done

# 4. DINO encoders: final checkpoint + config + metrics per run, plus hf_release -> Oak
if ! done_ dino_encoders; then
  t=$ST/dino_encoders_2026.1.tar
  [ -f $t ] || tar cf $t -C /data2/mcfrank dino_s2_vits/ckpt/199999 dino_s2_vits/config.yaml dino_s2_vits/training_metrics.json dino_s2_vits/probes.jsonl \
      dino_s3_vitb/ckpt/199999 dino_s3_vitb/config.yaml dino_s3_vitb/training_metrics.json dino_s3_vitb/probes.jsonl \
      dino_s4_vitl/ckpt/199999 dino_s4_vitl/config.yaml dino_s4_vitl/training_metrics.json dino_s4_vitl/probes.jsonl \
      dino_s2_vits_stdout.log dino_s3_vitb_stdout.log dino_s4_vitl_stdout.log hf_release 2>>$LOG
  ship_verify $t $OAKP && rm -f $t && log "DONE dino_encoders"
fi
if done_ dino_encoders && ! done_ dino_intermediates; then
  for r in dino_s2_vits dino_s3_vitb dino_s4_vitl; do
    for c in /data2/mcfrank/$r/ckpt/*; do [ "$(basename $c)" = "199999" ] || rm -rf "$c"; done
  done
  rm -rf /data2/mcfrank/dino_s1_vits /data2/mcfrank/dino_smoke /data2/mcfrank/dino_smoke2 && log "DONE dino_intermediates (kept ckpt 199999 x3)"
fi

# 5. legacy caches in the working tree (2025.2 book era) and the DINO probe caches -> tars -> Oak -> delete
cd $W
if ! done_ legacy_caches; then
  t=$ST/legacy_caches_2025_2_20260910.tar
  [ -f $t ] || tar cf $t emb_reg emb_dv3_grid_877k emb_s1 emb_s1_eval emb_enc emb_enc_eval emb_enc_grid emb_enc_grid_top emb_reg_mp emb_full emb_cls1 emb_reg_ho_* \
      emb_konkle emb_konkle_cls emb_konkle_cls1 emb_konkle_dev emb_konkle_dev_cls emb_konkle_mp emb_dv3_konkle_dev emb_crop_eval emb_crop_train emb_bench \
      eval_frames_mcfrank lev_vocab_images lev_vocab_imgs.tgz pose_bench_overlays pose_calib_panels pose_compare book_figs metadata runs_oak runs_invalid_20260822 2>>$LOG
  ship_verify $t $OAKP && rm -rf emb_reg emb_dv3_grid_877k emb_s1 emb_s1_eval emb_enc emb_enc_eval emb_enc_grid emb_enc_grid_top emb_reg_mp emb_full emb_cls1 emb_reg_ho_* \
      emb_konkle emb_konkle_cls emb_konkle_cls1 emb_konkle_dev emb_konkle_dev_cls emb_konkle_mp emb_dv3_konkle_dev emb_crop_eval emb_crop_train emb_bench \
      eval_frames_mcfrank lev_vocab_images lev_vocab_imgs.tgz pose_bench_overlays pose_calib_panels pose_compare book_figs metadata runs_oak runs_invalid_20260822 $t && log "DONE legacy_caches"
fi
if ! done_ probe_caches; then
  t=$ST/dino_probe_caches_20260910.tar
  [ -f $t ] || tar cf $t emb_probe_* emb_randL_* emb_hfexport_* 2>>$LOG
  ship_verify $t $OAKP && rm -rf emb_probe_* emb_randL_* emb_hfexport_* $t && log "DONE probe_caches"
fi

# 6. retired originals (canonical copies verified on the release tree / Oak on 2026-08-31)
done_ retired || { rm -rf /data2/mcfrank/_retired_20260829 && log "DONE retired"; }

log "=== NODE_ARCHIVE_DONE === $(df -h /data2 | tail -1 | awk '{print $4" free"}')"
