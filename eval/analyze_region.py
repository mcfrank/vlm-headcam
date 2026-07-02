"""Analyze region-MIL batch (R_* runs) vs the CLS-cosine P1 baselines."""
import json
from pathlib import Path
R = Path("runs")

def log(n):
    p = R / n / "log.json"
    return json.loads(p.read_text()) if p.exists() else None

def plain_best(n):
    lg = log(n)
    return max(r["acc_4afc"] for r in lg) if lg and all(r["acc_4afc"] == r["acc_4afc"] for r in lg[-1:]) else (
        max((r["acc_4afc"] for r in lg if r["acc_4afc"] == r["acc_4afc"]), default=None) if lg else None)

def boot_summ(n):
    lg = log(n)
    if not lg:
        return None
    boot = [r for r in lg if r["phase"] == "boot"]
    best = max((r["acc_4afc"] for r in boot if r["acc_4afc"] == r["acc_4afc"]), default=None)
    y = next((r for r in reversed(boot) if "spearman_w_clip" in r), {})
    return best, y.get("spearman_w_clip"), y.get("precision_vs_clip"), y.get("base_rate_clip>0.24")

print("REGION-MIL vs CLS reference (4AFC on held-out S00360001, chance 25)")
print("\nOracles / baselines (plain):")
for n, lab in [("R_oracle_across", "region-MIL oracle across (CLS ref 46.7)"),
               ("R_oracle_within", "region-MIL oracle within (CLS ref 34.9)"),
               ("R_plain_across_140000", "region-MIL unweighted across-140k (CLS ref 33.0)"),
               ("R_plain_within_110000", "region-MIL unweighted within-110k (CLS ref 32.2)")]:
    b = plain_best(n)
    print(f"  {lab:52} {b*100:.1f}" if b else f"  {lab:52} (missing)")

print("\nBootstrap (boot):  4AFC  ρ(w,clip)  precision  base   [CLS P1 ref]")
refs = {"R_boot_across_20000": 28.9, "R_boot_across_60000": 30.8, "R_boot_across_140000": 32.2,
        "R_boot_within_20000": 26.1, "R_boot_within_60000": 29.2, "R_boot_within_110000": 31.2}
for n in ["R_boot_across_20000", "R_boot_across_60000", "R_boot_across_140000",
          "R_boot_within_20000", "R_boot_within_60000", "R_boot_within_110000"]:
    s = boot_summ(n)
    if not s:
        print(f"  {n:26} (missing)"); continue
    best, rho, prec, base = s
    print(f"  {n:26} {best*100:5.1f}   {str(rho):>7}   {str(prec):>6}  {str(base):>5}   [{refs[n]}]")
