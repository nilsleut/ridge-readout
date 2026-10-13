"""
prereg_aggregate.py
===================
Aggregation only (no fits) for the preregistered primary analysis
(PREREGISTRATION.md §2, §3, §5.5; config hash 206fa875cca5). Reads results/prereg/primary.csv
written by scripts/prereg_primary.py.

1. Median r over reliable voxels per subject, condition, seed and g (as written by the
   fit script); cells marked invalid by the D1 plateau gate are excluded.
2. Mean over the three subjects (config.yaml brain.aggregation.over_subjects: mean); a
   (condition, seed, g) cell enters only if all three subjects are valid.
3. Primary contrast Random - Backprop per seed (paired within seed).
4. Paired one-sided exact sign-flip test over the 5 seeds: all 2^5 = 32 sign assignments,
   statistic = mean over seeds, alternative Random - BP > 0; p = (number of assignments
   with statistic >= observed) / 32. Smallest attainable p = 1/32.
5. Delta_min per g: scripts/pilot_fit.py::delta_min_signflip (unchanged) at the seed-level
   SD of the paired difference (simulated, closed form, parametric cross-check).
6. Secondary, same method: BN-cal - BP, FA - BP, PC - BP, STDP - BP.

Decision rule, verbatim from PREREGISTRATION.md §3:
  3a "Surviving at g = 1 alone would not count as surviving. g = 1 is the paper's own
     condition re-run under a fitted readout; the claim requires it to hold at g >= 4,
     where spatial structure is retained."  Operationalised: one-sided sign-flip p <= 0.05
     (i.e. p = 1/32) for Random - BP at g = 4 AND at g = 8.
  3b otherwise: "The effect disappears" requires Delta_min to be smaller than the effect
     size the paper reports (config.yaml reference_effect.at_224px); "The method cannot
     resolve it" if Delta_min is comparable to or larger. Evaluated at g = 4 and g = 8.

Output: results/prereg/signflip.csv
"""
import itertools
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

PRE = REPO / "results" / "prereg"
BP = "Backprop"
CONTRASTS = [("Random Weights", "primary"), ("Random Weights (BN-calibrated)", "secondary"),
             ("Feedback Alignment", "secondary"), ("Predictive Coding", "secondary"), ("STDP", "secondary")]
ALPHA = 0.05
SIGNS = np.array(list(itertools.product([1, -1], repeat=5)), dtype=float)   # 32 x 5


def signflip_p(d):
    d = np.asarray(d, float)
    obs = d.mean()
    null = (SIGNS * d).mean(axis=1)
    return float((null >= obs - 1e-15).sum() / len(SIGNS))


def main():
    prim = pd.read_csv(PRE / "primary.csv")
    valid = prim[prim.valid]
    n_sub = valid.groupby(["rule", "seed_idx", "grid"]).subject.nunique()
    full = n_sub[n_sub == 3].index
    agg = valid.set_index(["rule", "seed_idx", "grid"]).loc[full].reset_index()
    mean = agg.groupby(["rule", "seed_idx", "grid"]).r_med_reliable.mean().reset_index()
    ref = float(load_config()["provenance"]["reference_effect"]["at_224px"])

    rows = []
    for g in sorted(mean.grid.unique()):
        p = mean[mean.grid == g].pivot(index="seed_idx", columns="rule", values="r_med_reliable")
        for a, role in CONTRASTS:
            d = (p[a] - p[BP]).dropna()
            if len(d) != 5:
                rows.append(dict(kind="contrast", role=role, grid=g, quantity=f"{a} - {BP}", n_seeds=len(d),
                                 note="fewer than 5 valid seeds; test not computed"))
                continue
            sd = float(d.std(ddof=1))
            dm = delta_min_signflip(sd)
            rows.append(dict(kind="contrast", role=role, grid=g, quantity=f"{a} - {BP}", mean=float(d.mean()),
                             sd=sd, n_seeds=5, seeds_positive=int((d > 0).sum()), p_one_sided=signflip_p(d),
                             delta_min=dm["simulated"], delta_min_closed=dm["closed_form"],
                             delta_min_parametric=dm["parametric"],
                             **{f"seed{i}": float(v) for i, v in zip(d.index, d.values)}))
        for rule in p.columns:
            rows.append(dict(kind="condition", grid=g, quantity=rule, mean=float(p[rule].mean()),
                             sd=float(p[rule].std(ddof=1)), n_seeds=int(p[rule].notna().sum())))
    out = pd.DataFrame(rows)

    pr = out[(out.kind == "contrast") & (out.role == "primary")].set_index("grid")
    ge4 = [g for g in (4, 8) if g in pr.index]
    survives = len(ge4) == 2 and all(pr.loc[g, "p_one_sided"] <= ALPHA for g in ge4)
    if survives:
        verdict = "3a: the effect survives"
        text = ("The resolution confound is not readout-specific; the claim holds at g >= 4, where spatial "
                "structure is retained (subject to the spatial-correspondence control, §5.3).")
    else:
        dmin = max(float(pr.loc[g, "delta_min"]) for g in ge4)
        if dmin < ref:
            verdict = "3b: the effect disappears"
            text = (f"Delta_min at g >= 4 (max {dmin:.4f}) is smaller than the effect size the paper reports "
                    f"({ref:+.3f}), so absence is informative: the effect is specific to the pooled-RDM methodology.")
        else:
            verdict = "3b: the method cannot resolve it"
            text = (f"Delta_min at g >= 4 (max {dmin:.4f}) is comparable to or larger than {ref:+.3f}; this is a "
                    "statement about the measurement, not about the effect.")
    out = pd.concat([out, pd.DataFrame([dict(
        kind="verdict", quantity=verdict, note=text,
        **{f"p_g{g}": float(pr.loc[g, "p_one_sided"]) for g in pr.index},
        reference_effect=ref, survives=survives)])], ignore_index=True)

    out["config_hash"] = config_hash()
    out["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
    out.to_csv(PRE / "signflip.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[out.kind == "contrast"][["role", "grid", "quantity", "mean", "sd", "seeds_positive", "p_one_sided",
                                        "delta_min"]].to_string(index=False))
    print(verdict, "|", text)


if __name__ == "__main__":
    main()
