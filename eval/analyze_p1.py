import json
from pathlib import Path
R = Path("runs")

def log(name):
    p = R / name / "log.json"
    return json.loads(p.read_text()) if p.exists() else None

def best_plain(name):
    lg = log(name)
    return max(r["acc_4afc"] for r in lg) if lg else None

def boot_final(name):
    """bootstrap run: best 4AFC over boot rounds + final yardstick."""
    lg = log(name)
    if not lg:
        return None
    boot = [r for r in lg if r["phase"] == "boot"]
    warm = [r for r in lg if r["phase"] == "warmup"]
    best = max(r["acc_4afc"] for r in boot) if boot else None
    y = next((r for r in reversed(boot) if "spearman_w_clip" in r), {})
    return {"warmup_acc": warm[-1]["acc_4afc"] if warm else None,
            "boot_best": best,
            "spearman": y.get("spearman_w_clip"),
            "prec": y.get("precision_vs_clip"), "base": y.get("base_rate_clip>0.24"),
            "sel_frac": y.get("sel_frac")}

print("Oracles (CLIP-filtered, topline):")
print(f"  across (aligned_S00360001): {best_plain('P1_oracle_across')*100:.1f}")
print(f"  within (S00510002 clip>0.24): {best_plain('P1_oracle_within')*100:.1f}")

for pool, sizes in [("across", [20000, 60000, 140000]), ("within", [20000, 60000, 110000])]:
    print(f"\n=== {pool}-kid ===")
    print(f"{'size':>8} {'plain':>7} {'boot':>7} {'warmup':>7}  {'Δboot-plain':>11}  "
          f"{'ρ(w,clip)':>9} {'prec':>6} {'base':>6}")
    for n in sizes:
        plain = best_plain(f"P1_plain_{pool}_{n}")
        b = boot_final(f"P1_boot_{pool}_{n}")
        d = (b["boot_best"] - plain) if (b and plain) else None
        print(f"{n:>8} {plain*100:>7.1f} {b['boot_best']*100:>7.1f} {b['warmup_acc']*100:>7.1f}  "
              f"{d*100:>+11.1f}  {str(b['spearman']):>9} {str(b['prec']):>6} {str(b['base']):>6}")
