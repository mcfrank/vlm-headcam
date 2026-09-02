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
