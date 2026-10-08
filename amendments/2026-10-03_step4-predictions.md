# Step 4 predictions (fixed before runs)

Committed before any step 4 script exists or runs. Exploratory.

## Hypothesis H_scale

The RSA advantage of Random Weights at V1 (sub-01, D_raw = Random − BP = +0.0278 on the
501 reliable voxels) arises from differences in channel scale of the features.

## S1 (sub-01, g = 1)

RSA with channel-wise standardized features (z-score per channel over the 720 stimuli),
otherwise identical to `results/step2/rsa_501.csv`. D_std = Random − BP, mean over 5 seeds.

- D_std ≤ 0.010: H_scale supported.
- D_std ≥ 0.020: H_scale not supported.
- otherwise: ambiguous, reported as such.

Additional control criterion: |Random − BN-cal| under standardized RSA ≤ 0.010. If not
met, record that offsets before the ReLU contribute.

## S2 (subjects)

For sub-02 and sub-03 (reliable V1 voxels, g = 1, random folds) the pattern holds if
Random − BP under raw RSA > 0 AND Random − BP under ridge < 0. Reported per subject
separately; sub-03 has only 286 reliable voxels and is explicitly marked as weaker.
Also reported: BN-cal − BP under ridge and RSA, per subject.

## Scope

All statements concern stimulus variation in 3 subjects, not the population. The unit
of inference in this step is the seed (n = 5); no stimulus bootstrap.
