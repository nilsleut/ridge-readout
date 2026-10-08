# Step 5 resolution predictions (fixed before runs)

Date: 2026-10-08. Committed before any step 5 script exists or runs. Exploratory.

## Hypothesis H_mom

The Random advantage at V1 and its dependence on evaluation resolution in
arXiv:2608.12408 arise from channel moments of the GAP features (mean and SD per
channel), which raw RSA with correlation distance captures and channel-wise
standardized RSA does not.

## Convention (as in paper v3)

- per subject (3), then averaged;
- all voxels of the ROI (not only reliable ones);
- cross-run pairs only (210,205);
- stimulus bootstrap with 1000 resamples, seed 20261002, paired differences from the
  same draws;
- 95% percentile CIs.

Variants: `raw`, and `z` (z-score per channel over the 720 stimuli, dead channels set
to 0).

## Rules

**R1 Reproduction (gate).**
- raw, V1, NATIVE 224: Random − BP = +0.030 ± 0.001, with CI bounds within ± 0.002 of
  [0.018, 0.042].
- raw, V1, NATIVE 32: −0.001 ± 0.001.
- Otherwise STOP.

**R2 (primary).** z, V1, NATIVE 224, Random − BP.
- Upper CI bound ≤ 0.010: H_mom explains the headline.
- Lower CI bound ≥ 0.015: H_mom does not explain it.
- Otherwise ambiguous.

**R3 Resolution slope.** S = (Random − BP at 224) − (Random − BP at 32), for raw and z.
S_z ≤ 0.5 · S_raw: the resolution effect at V1 is predominantly moment-driven.
Otherwise it is not.

**R4 LOC dissociation (prediction: survives).** z, LOC, NATIVE 224, BP − Random: if
its CI excludes 0 and the value is > 0, the LOC advantage of BP survives
standardization.

**R5 UPSAMPLED 224 (content limited to 32 px).** z, V1: upper CI bound of
Random − BP ≤ 0.010.

**M1 Mechanism index.** Per checkpoint and resolution,
m = ‖μ‖² / (‖μ‖² + Σ_k σ_k²) of the V1-layer features (μ, σ per channel over the 720
stimuli).
- Prediction: the Spearman correlation between m and d = RSA_raw − RSA_z (V1,
  subject-averaged), over all cells (checkpoint × resolution) of the conditions Random
  and BP, is ≥ 0.6.
- Also reported, without a rule: within Random and within BP separately.

## Limits

N = 3 subjects. Inference is over stimuli, not the population.
