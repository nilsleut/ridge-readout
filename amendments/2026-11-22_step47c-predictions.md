# Step 47c: decomposition of run centring. Predictions (fixed before scripts and runs)
Date: 2026-11-22. Configuration: config_step47c.yaml, config hash ea2fd24bd0a1.
Facts known before: run centring (subtracting the voxel mean over the 82 trials of a run) reduces the THINGS-720 V1 paper-cell Δ from +0.0304 to +0.0070 (step 47); the raw / correlation / center720 multiverse cell is +0.0046 (NATIVE 224, results/step7/multiverse.csv; 95 % interval −0.0034 to +0.0125); in arXiv:2611.15533 the untrained advantage was absent from the component orthogonal to the unit-mean profile.
Definitions per subject, V1, after the existing z-score: m = voxel mean over all 820 trials of the session; run mean μ_r = mean over the 82 trials of run r; run deviation d_r = μ_r − m; m̂ = m / ||m||; d_r∥ = (d_r · m̂) m̂; d_r⊥ = d_r − d_r∥.
Variants (Δ = Random − BP, paper cell, cross-run pairs, mean over seeds 0–4, then subjects):
  V0 nothing removed (= paper cell); V1 remove m only; V2 remove d_r only; V3 remove both (= run centring); V4 = V3 + d_r∥ re-added; V5 = V3 + d_r⊥ re-added.
Z1: V2 ≤ +0.015 (run deviations alone carry most of the effect).
Z2: V4 ≥ +0.020 and V5 ≤ +0.012 (the run deviations act along the mean-profile axis).
Z3: the across-run SD of the scalar d_r · m̂ is smallest in sub-03.
Interpretation fixed in advance:
  Z1 and Z2 met → the run-level part of the advantage is attributed to run-wise shifts of global activation along the voxel mean-profile axis.
  Z1 not met and V1 ≤ +0.015 → the effect of run centring is attributed mainly to removing the session mean profile (equivalent to voxel centring), not to between-run structure; the description of step 47 as "run structure" is then corrected in the manuscript.
  Any other pattern → not identified; all variants are reported.
Reported without prediction: all variants per subject; share of ||d_r||² along m̂ per subject; Spearman between the per-trial cosine of the pattern with m̂ and the image luminance, per subject, with and without run centring; LOC.
