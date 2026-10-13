"""
prereg_delta_min.py
===================
Delta_min as preregistered (PREREGISTRATION.md §3b, lines 88-90): "Delta_min is derived
from the pilot's seed-level SD by simulating the sign-flip test above, with the
parametric (t_0.975,4 + t_0.80,4) * SD / sqrt(5) approximation as a cross-check."

The pilot's seed-level SD is read from reports/PILOT_REPORT.md only:
  §3, table "Paired seed-level contrasts against Backprop", row "Random − Backprop",
  column SD (sub-01, V1, 224 px, g = 1, 501 reliable voxels; the pilot ran g = 1 only),
  on which §4 ("At the observed seed-level SDs") bases its Delta_min.
The report gives a single SD for the primary contrast, so a single Delta_min applies to
every g. As a cross-check the SD is recomputed from the per-seed values the same report
lists for Random Weights and Backprop (rounded to 4 decimals there).

delta_min_signflip is imported from scripts/pilot_fit.py unchanged. §3b is re-applied with
this Delta_min to the primary contrast of results/prereg/signflip.csv (which is not
modified; its delta_min columns are supplementary: Delta_min from the primary analysis's
own seed SD). Decision as in scripts/prereg_aggregate.py: §3a needs p <= 0.05 at g = 4 and
g = 8; otherwise "The effect disappears" if Delta_min < reference_effect.at_224px, else
"The method cannot resolve it".

Output: results/prereg/delta_min_prereg.csv
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src" / "ridge"))
sys.path.insert(0, str(HERE))

from config_hash import config_hash, load_config      # noqa: E402
from pilot_fit import delta_min_signflip                # noqa: E402

PILOT = REPO / "reports" / "PILOT_REPORT.md"
PRE = REPO / "results" / "prereg"
NUM = r"([+−-]?\d+\.\d+)"


def num(s):
    return float(s.replace("−", "-").replace("+", ""))


def pilot_sd():
    text = PILOT.read_text(encoding="utf-8")
    rows = re.findall(r"^\| Random − Backprop \| \*\*" + NUM + r"\*\* \| " + NUM + r" \|", text, flags=re.M)
    if len(rows) != 1:
        raise SystemExit(f"STOP: expected one 'Random − Backprop' contrast row in {PILOT.name}, found {len(rows)}")
    sd = num(rows[0][1])

    def seeds(rule):
        m = re.findall(r"^\| \*?\*?" + re.escape(rule) + r"\*?\*? \| 5 \| .*?\| ([^|]+) \|$", text, flags=re.M)
        if len(m) != 1:
            raise SystemExit(f"STOP: per-seed row for {rule} not found uniquely")
        return np.array([num(x) for x in m[0].split()])
    d = seeds("Random Weights") - seeds("Backprop")
    return sd, float(d.std(ddof=1))


def main():
    sd, sd_check = pilot_sd()
    print(f"pilot SD (PILOT_REPORT §3, Random − Backprop, sub-01, g = 1): {sd}; "
          f"recomputed from the listed per-seed values: {sd_check:.5f}")
    dm = delta_min_signflip(sd)
    ref = float(load_config()["provenance"]["reference_effect"]["at_224px"])
    sf = pd.read_csv(PRE / "signflip.csv")
    prim = sf[(sf.kind == "contrast") & (sf.role == "primary")].set_index("grid")
    survives = all(prim.loc[g, "p_one_sided"] <= 0.05 for g in (4, 8))
    if survives:
        verdict = "3a: the effect survives"
    elif dm["simulated"] < ref:
        verdict = "3b: The effect disappears"
    else:
        verdict = "3b: The method cannot resolve it"
    rows = [dict(g=int(g), sd_source="reports/PILOT_REPORT.md §3 (Random − Backprop, sub-01, 224 px, g = 1); "
                                      "single pilot SD applied to every g",
                 sd_value=sd, sd_recomputed_from_listed_seeds=sd_check,
                 delta_min_sim=dm["simulated"], delta_min_closed=dm["closed_form"], delta_min_param=dm["parametric"],
                 reference_effect=ref, p_one_sided=float(prim.loc[g, "p_one_sided"]),
                 supplementary_delta_min_own_sd=float(prim.loc[g, "delta_min"]), verdict_3b=verdict)
            for g in sorted(prim.index)]
    out = pd.DataFrame(rows)
    out["config_hash"] = config_hash()
    out["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
    out.to_csv(PRE / "delta_min_prereg.csv", index=False)
    print(out[["g", "sd_value", "delta_min_sim", "delta_min_closed", "delta_min_param",
               "supplementary_delta_min_own_sd", "verdict_3b"]].to_string(index=False))


if __name__ == "__main__":
    main()
