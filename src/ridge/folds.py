"""
folds.py
========
Cross-validation partitioning, grouped by concept.

Phase 0/B established that the 720 evaluation stimuli are 720 DISTINCT concepts, one
exemplar each (extract_fmri_rdms_720.py:89-98 takes `sorted(...)[:1]` per concept). So
concept-grouped folds are image-level folds and the grouping is a no-op today.

It is implemented and asserted anyway, for one reason: if the stimulus selection ever
changes to include multiple exemplars per concept, ungrouped folds would leak
near-duplicate information across the split and inflate every number silently. The
assertion costs nothing and converts that silent failure into a loud one.

Outer 8 folds, inner 5 (for alpha only), fixed partition, seed from config.yaml.
"""

from typing import Iterator, List, Sequence, Tuple

import numpy as np

__all__ = ["grouped_folds", "assert_no_group_crosses_fold", "outer_inner_splits"]


def grouped_folds(groups: Sequence, n_folds: int, seed: int) -> List[np.ndarray]:
    """Partition indices into n_folds, keeping every group wholly within one fold.

    Groups are shuffled with `seed` and dealt to the currently-smallest fold, which
    keeps folds balanced when group sizes are uneven. With one item per group (the
    current data) this reduces to a plain balanced shuffle.
    """
    groups = np.asarray(groups)
    n = len(groups)
    if n_folds < 2 or n_folds > n:
        raise ValueError(f"n_folds={n_folds} is not usable for n={n}")

    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))

    buckets: List[List[int]] = [[] for _ in range(n_folds)]
    sizes = np.zeros(n_folds, dtype=int)
    for gi in order:
        members = np.flatnonzero(groups == uniq[gi])
        t = int(np.argmin(sizes))
        buckets[t].extend(members.tolist())
        sizes[t] += len(members)

    folds = [np.sort(np.array(b, dtype=int)) for b in buckets]
    empty = [i for i, f in enumerate(folds) if len(f) == 0]
    if empty:
        raise ValueError(f"folds {empty} came out empty; too few groups for {n_folds} folds")
    return folds


def assert_no_group_crosses_fold(groups: Sequence, folds: Sequence[np.ndarray]) -> None:
    """Every group must live entirely inside one fold. Also checks the partition."""
    groups = np.asarray(groups)
    n = len(groups)

    seen = np.concatenate([np.asarray(f, dtype=int) for f in folds]) if folds else np.array([], int)
    if len(seen) != n or len(np.unique(seen)) != n:
        raise AssertionError(
            f"folds are not a partition: {len(seen)} indices covering "
            f"{len(np.unique(seen))} unique, expected {n} of each"
        )

    home = {}
    for fi, f in enumerate(folds):
        for g in np.unique(groups[np.asarray(f, dtype=int)]):
            if g in home:
                raise AssertionError(
                    f"group {g!r} appears in folds {home[g]} and {fi}. Near-duplicate "
                    "information would leak across the split and inflate every number."
                )
            home[g] = fi


def outer_inner_splits(groups: Sequence, n_outer: int, n_inner: int, seed: int
                       ) -> Iterator[Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]]:
    """Yield (train_idx, test_idx, inner_splits) per outer fold.

    Inner splits are built from the OUTER TRAINING SET ONLY and are used for alpha
    selection only. The outer test fold is never touched during selection.
    """
    outer = grouped_folds(groups, n_outer, seed)
    assert_no_group_crosses_fold(groups, outer)
    groups = np.asarray(groups)

    for fi, test_idx in enumerate(outer):
        train_idx = np.sort(np.concatenate([f for j, f in enumerate(outer) if j != fi]))
        # Re-derive inner folds on the training subset, still concept-grouped, with a
        # seed derived from the outer fold so the whole partition is reproducible.
        inner = grouped_folds(groups[train_idx], n_inner, seed + 1000 * (fi + 1))
        assert_no_group_crosses_fold(groups[train_idx], inner)
        inner_splits = []
        for ii, val_local in enumerate(inner):
            tr_local = np.sort(np.concatenate([f for j, f in enumerate(inner) if j != ii]))
            inner_splits.append((train_idx[tr_local], train_idx[val_local]))
        yield train_idx, test_idx, inner_splits
