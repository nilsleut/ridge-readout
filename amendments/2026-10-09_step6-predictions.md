# Step 6 predictions (fixed before runs)

Date: 2026-10-09. Committed before any step 6 script exists or runs. Exploratory.
Conventions as in step 5: per subject, then averaged; cross-run pairs; stimulus
bootstrap with 1000 resamples, seed 20261002, paired differences; variants raw and z.

## P_ridge — ridge across resolutions

**Setup.**
- GAP features, g = 1, from `$STEP5_GAP_DIR`.
- V1, reliable voxels (ridge convention of the preregistration).
- Six NATIVE resolutions plus UPSAMPLED 224; all 30 checkpoints.
- sub-01 and sub-02 primary; sub-03 reported separately.

**Criterion.** Both of the following must hold:
- Random − BP (mean over sub-01/02, mean over seeds) ≤ 0 at all six NATIVE
  resolutions;
- S_ridge = (224) − (32) ≤ 0.5 · S_raw, with S_raw = +0.0311 from step 5.

If both hold, ridge confirms z-RSA across resolutions.

## P_mech1 — moments versus input scale

- For BP, m(UPSAMPLED 224) < m(NATIVE 224) < m(NATIVE 32) in at least 4 of 5 seeds
  (V1 layer).
- For Random, the range of m over these three conditions is smaller than half the BP
  range (mean over seeds).

## P_mech2 — band-pass, causal

**Inputs.** NATIVE 224, conditions only_0-14 (coarse band only) and notch_0-14 (coarse
band removed), from the deterministic filter function in `bandpass_extract.py`.

**Predictions.**
- m_BP(notch) > m_BP(only) in 5 of 5 seeds.
- Over all cells (Random and BP × seeds × {NATIVE 32, NATIVE 224, UPSAMPLED 224,
  only_0-14, notch_0-14}), Spearman(m, d) ≥ 0.6, with d = RSA_raw − RSA_z at V1,
  subject-averaged.

## Descriptive, no rule

Robustness, filter spectra and stimulus spectra are descriptive only.

## Limits

N = 3 subjects; inference over stimuli, not over the population.
