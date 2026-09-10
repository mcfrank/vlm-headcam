# runners/ — the shell drivers that produced the paper's runs

All run on ccn2-14 from the repo root (`bash runners/<name>.sh`; each script `cd`s to the
repo root itself). They are resumable and in-flight-guarded: a run whose `metrics.json`
exists, or whose process is alive, is skipped, so any script can be relaunched to fill gaps.
Every run lands in `runs/F_<encoder>_<condition>_s<seed>/` and is scraped into
`results/runs.parquet` by `src/scrape_runs.py`. Encoder tags: `dinov3s/dinov3b/dinov3l` =
OTS-22M/86M/304M (DINOv3 ViT-S/B/L), `vits_bv/vitb_bv/vitl_bv` = BV-22M/86M/304M.

| script | what it runs |
|---|---|
| `run_final.sh` | the main matrix for the first four encoders: random-scaling (3k–1M ×8/5/3 seeds, full ×3), ladder (full), aligned scaling (10k–170k ×3); L-OTS extras (ladder at 100k/300k, diversity 100k/300k) |
| `run_wf.sh` | no-MIL (whole-frame) control, 30k/300k/full ×5, plus `base` s3/s4 top-ups (needs `src/make_wf_caches.py` first) |
| `run_div30k.sh` | diversity sweep at the 30k budget (L-OTS) |
| `run_controls.sh` | alignment-selection controls round 1 (aligned-only, matched random, full−aligned, full−random) |
| `run_controls2.sh` | round 2 after the code audit (plain random 172k, full−matched) |
| `run_window.sh`, `run_window2.sh` | temporal ±5 s window control at 30k/300k, then 1M/full (gated on the neighbor caches) |
| `run_kchi_control.sh` | child-speech (KCHI) removal control (L-OTS) |
| `run_vitl.sh`, `run_dinov3s.sh` | the complete 113-run sequence for the two encoders added last (BV-304M, OTS-22M) |
| `win5_embed.sh`, `win5b_embed.sh`, `vitl_embed_driver.sh`, `dinov3s_driver.sh` | embedding drivers: window neighbor caches (rounds 1–2), and the full cache campaigns for the two late encoders |
| `run_joint_pilot.sh` | joint encoder+tower training pilot (SI) |
| `status.sh` | read-only status of everything queued on the node |

`archive/` holds the runners of the 2025.2-era book phases and the preview-corpus reruns
(`run_2026_*`, `run_ch8`, `run_div2`, `run_english_*`, `run_layer_pilot`, `run_phase5`,
`run_scaling_enc`, and the very early `launchers/`). None are needed for the paper.
