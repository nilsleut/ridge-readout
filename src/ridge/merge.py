"""
merge.py
========
Duplicate check on merge, over the full result key.

Same discipline as learning_rules_v10_sweep_modal.py:665-689, which this project
already relies on: a duplicated key means the same cell was computed twice, and a
seeded deterministic pipeline must produce the same number both times. A retry that
produces a DIFFERENT value is a bug signal, not noise -- so verify every numeric column
and RAISE rather than collapsing to one row.

Key (config.yaml output.duplicate_check_key):
    arm, rule, seed, resolution, grid, layer, roi, subject, fold

Also asserts the expected 30 rule x seed combinations -- SIX rules, not five: Random
Weights (BN-calibrated) is a distinct condition.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["KEY", "RULES", "N_RULE_SEED_EXPECTED", "check_duplicates",
           "assert_rule_seed_coverage", "merge_shards"]

KEY = ["arm", "rule", "seed", "resolution", "grid", "layer", "roi", "subject", "fold"]

RULES = [
    "Random Weights",
    "Backprop",
    "Feedback Alignment",
    "Predictive Coding",
    "STDP",
    "Random Weights (BN-calibrated)",
]
SEEDS = [42, 123, 456, 789, 1337]
N_RULE_SEED_EXPECTED = len(RULES) * len(SEEDS)      # 30

ATOL = 1e-12


def check_duplicates(df: pd.DataFrame, key: Optional[Sequence[str]] = None,
                     atol: float = ATOL) -> pd.DataFrame:
    """Verify duplicated keys agree on every numeric column, then collapse them.

    Raises on disagreement. NaN in the same places counts as agreement; NaN in only
    some rows of a duplicate group does not.
    """
    key = list(key or KEY)
    missing = [k for k in key if k not in df.columns]
    if missing:
        raise ValueError(f"result frame is missing key column(s) {missing}")

    dup = df.duplicated(subset=key, keep=False)
    if not dup.any():
        return df

    num_cols = [c for c in df.columns
                if c not in key and pd.api.types.is_numeric_dtype(df[c])]
    conflicts = []
    for k, g in df[dup].groupby(key, sort=True):
        for c in num_cols:
            v = g[c].to_numpy(dtype=float)
            finite = ~np.isnan(v)
            if not (finite.all() or (~finite).all()):
                conflicts.append((k, c, list(g[c])))
            elif finite.all() and not np.allclose(v, v[0], rtol=0, atol=atol):
                conflicts.append((k, c, list(g[c])))

    n_keys = int(df[dup].groupby(key, sort=False).ngroups)
    if conflicts:
        shown = "\n".join(f"  {k}  col={c}  values={v}" for k, c, v in conflicts[:20])
        raise RuntimeError(
            f"{len(conflicts)} numeric disagreement(s) across {n_keys} duplicated "
            f"key(s): a recomputation of a deterministic cell changed its value. "
            f"Refusing to merge.\n{shown}"
            + (f"\n  ... and {len(conflicts) - 20} more" if len(conflicts) > 20 else "")
        )
    return df.drop_duplicates(subset=key, keep="last")


def assert_rule_seed_coverage(df: pd.DataFrame, expected: int = N_RULE_SEED_EXPECTED
                              ) -> None:
    """Exactly the 30 (rule, seed) combinations must be present."""
    have = set(map(tuple, df[["rule", "seed"]].drop_duplicates().to_numpy()))
    want = {(r, s) for r in RULES for s in SEEDS}
    missing, extra = sorted(want - have), sorted(have - want)
    if missing or extra:
        raise AssertionError(
            f"expected {expected} (rule, seed) combinations, found {len(have)}.\n"
            f"  missing: {missing}\n  unexpected: {extra}\n"
            "Note there are SIX rules: Random Weights (BN-calibrated) is a distinct "
            "condition, not a duplicate of Random Weights."
        )


def merge_shards(frames: Sequence[pd.DataFrame], config_hash: str,
                 key: Optional[Sequence[str]] = None,
                 require_full_coverage: bool = True) -> pd.DataFrame:
    """Concatenate shards, verify duplicates, stamp the config hash."""
    if not frames:
        raise ValueError("nothing to merge")
    df = pd.concat(list(frames), ignore_index=True)

    if "config_hash" in df.columns:
        seen = set(df["config_hash"].dropna().unique())
        if seen - {config_hash}:
            raise RuntimeError(
                f"shards carry config hashes {sorted(seen)} but the current config is "
                f"{config_hash}. Merging results computed under different frozen "
                "degrees of freedom would silently mix two studies."
            )
    df["config_hash"] = config_hash

    df = check_duplicates(df, key)
    if require_full_coverage:
        assert_rule_seed_coverage(df)
    return df.sort_values(list(key or KEY)).reset_index(drop=True)
