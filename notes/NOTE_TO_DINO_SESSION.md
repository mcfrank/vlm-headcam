# From the pipeline session (2026-08-29): two asks for the C9 encoders

1. Frame caches: please land them as
   /ccn2b/dataset/babyview/2026.1/outputs/image_embeddings/{vits_bv,vitb_bv}_grid4x4/
   (merged emb.f16.npy+index.parquet, or shard_*/ subdirs — run_final.sh handles both).
2. EVAL caches (small, minutes, NATIVE loader only — not the HF export):
   Konkle test (manifests/konkle_manifest.parquet), Konkle dev
   (manifests/eval_frames_konkle_dev.parquet), and LEVANTE
   (manifests/lev_vocab_manifest.parquet), each as R=17 CLS+grid4x4 to
   emb_ch8_eval/{vits_bv,vitb_bv}_konkle{,_dev} and emb_lev_{vits_bv,vitb_bv}.
The final runner (run_final.sh, F_ families) is live and self-gates per encoder: your two
join the matrix automatically the moment the caches appear. Manifests are final
(bv26a_*, audio-filtered corpus, 1.686M pairs / 48 children).
