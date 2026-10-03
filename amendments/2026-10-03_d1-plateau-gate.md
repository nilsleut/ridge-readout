# Amendment 2026-10-03 — D1 plateau gate (replaces D1's abort behaviour)

## Decision

1. A run **no longer aborts** on a low-edge alpha selection at a reliable voxel.
2. Every low-edge selection at a reliable voxel is **logged** with
   (condition, seed, g, voxel, outer fold).
3. After the run, a **mandatory gate** (`scripts/lower_edge_gate.py`, `gate(hits)`)
   classifies every logged selection with the procedure of step 2D
   (`reports/STEP2_LOWER_EDGE.md`), using the diagnostic grid points **1e-7, 1e-8, 1e-9**
   in addition to the standard grid. Points below 1e-9 are not used: there the solution
   is dominated by round-off eigenvalues of the dual kernel (|λ| ≤ 1.9e-12) and is not
   reliable.
   - relative gain = (MSE(1e-6) − MSE_min,diagnostic) / MSE(1e-6), on the mean inner-CV
     MSE curve of that voxel in that outer fold;
   - relative gain **< 1e-3**: **Plateau** — the selection is valid;
   - relative gain **≥ 1e-3**: the **cell is invalid**; it is excluded from every
     result and reported.
4. The selection rule (min-MSE, D2) and all scores are unchanged. The amendment is
   score-neutral: it decides only whether a cell is kept, never what number it has.
5. This replaces only D1's abort behaviour. The D1 text in `PREREGISTRATION.md` stays
   unchanged, as the record of what was decided on 2026-08-20.

## Record

- Decided by Nils on 2026-10-03, after the lower-edge diagnosis
  (`reports/STEP2_LOWER_EDGE.md`) was known.
- Evidence: 4 of 4 low-edge selections (Random Weights (BN-calibrated) s3, g = 1,
  sub-01) classified Plateau, relative gain 1.3e-5 to 1.6e-4.
- `config.yaml` is unchanged; the config hash stays `206fa875cca5`.
