# From the pipeline session: your two SI requests (2026-09-02)

(1) NO-MIL: already in flight when your note arrived, superset of your spec. Families:
    F_<enc>_wf30000_s{0..4}, F_<enc>_wf300000_s{0..4}, F_<enc>_wffull_s{0..4}
  (not "meanpool" — sorry, runs were queued before your naming suggestion). Mean-over-grid
  R=1 caches for train AND eval, per your/our shared correction re drop-CLS. Paired to the
  region runs' exact manifests (rand_30000_s<s> / rand_300000_s<s> / base). Region base
  top-ups F_<enc>_base_s{3,4} land too, so full-scale pairing is n=5 both sides.
  Your "full corpus 1,682,259" = post-vocab-drop effective count of the same 1,686,105 corpus.

(2) DIV 30k: manifests built (bv26a_div30k_{1,3,10,25,48}c_s{0..4}); runs land as
    F_dinov3l_div30k_{k}c_s{s}, 5 seeds, same construction as the 100k/300k cells.

Both scrape into runs.parquet as usual; I'll commit when each family completes.

## 2026-09-02 (evening): two control families complete in runs.parquet
- No-MIL: F_<enc>_wf30000 / wf300000 / wffull (n=5) paired to F_<enc>_rand_30000_s<s> /
  rand_300000_s<s> / base (base now n=5 via _s3,_s4 top-ups). Result: null at every scale for
  all four encoders — SI figure should show paired differences around zero.
- Alignment-selection: F_<enc>_{alignedonly,matchrand,minusaligned,minusrand} (n=3 each) vs
  base. Suggested SI panel: five bars per encoder. Diagnostics for the caption in
  results/aligned_control_diagnostics.json (aligned set has 2.4x the eval-noun exposure of a
  random subset; matched control equates it). Temporal-window (win5) lands in ~1-2 days.
