# Amendment 2026-10-13: local extraction of the g = 2 rung, downstream equivalence gate

Date: 2026-10-13. Committed before any further run of the preregistered primary
analysis.

## Feature source

- **g ∈ {1, 4, 8}:** the existing Modal T4 features in `pilot_features/`, identical to
  those behind all previous results.
- **g = 2 only:** the local CPU extraction in `data/prereg_features/`.
  - Code: `extract_pilot` from `scripts/pilot_features_modal.py`, run by
    `scripts/prereg_extract.py`.
  - Changes are limited to paths, CPU instead of CUDA, `num_workers=0`, and no volume
    commit.

## The original gate failed

- **Gate:** the locally re-extracted g ∈ {1, 4, 8} arrays had to equal the Modal
  features to an absolute 1e-6.
- **Result:**
  - max absolute deviation 2.1e-5;
  - max deviation relative to the array maximum 4.6e-6;
  - minimum Pearson correlation 0.99999999999997;
  - pixel and luminance features bit-identical.
- **Attribution:** CPU versus GPU floating-point arithmetic.

## Replacement gate

Decided by Nils on 2026-10-13, after the original gate failed. It tests **downstream
equivalence**:

- **Fits:** ridge at g = 4, sub-01, all 30 checkpoints, unchanged fit configuration,
  run once with the Modal features and once with the local features.
- **Pass:** |Δ median r| ≤ 1e-4 for every checkpoint, and |Δ(Random − BP)| ≤ 1e-4
  (mean over seeds).
- **Fail:** g = 2 is dropped and reported as a deviation.

The threshold 1e-4 is one order of magnitude below the smallest reported rounding unit
(1e-3).
