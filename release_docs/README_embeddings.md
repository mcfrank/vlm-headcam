# image_embeddings/dinov3b_grid4x4/ — frozen vision-encoder region features

- `emb.f16.npy` — float16, shape (1,745,489, 16, 768): a 4×4 region grid (row-major,
  NO CLS row) of facebook/dinov3-vitb16-pretrain-lvd1689m features per frame.
- `index.parquet` — row ↔ (video_id, frame_idx); covers every referent-annotation pair frame.

Produced 2026-08-24 by vlm-headcam src/embed_regions.py (register-token-aware: 1 CLS + 4
registers stripped before the 14×14 patch grid is mean-pooled 4×4). Merged from 8 strided
shards with byte-identical spot checks. Matches the R=16 readout convention of the 2025.2
`babyview_877k` caches (khaiaw's), so numbers are comparable across releases at the
cache level.
