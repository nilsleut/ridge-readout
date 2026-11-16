# Step 40: predictions (fixed before scripts and runs)
Date: 2026-11-16. Configuration: config_step40.yaml, config hash 3732a848c7a8.

Motivation: three limitations named in arXiv:2611.15533 (shared-penalty joint readout; single-presentation patterns; three subjects, one dataset).

A. Banded ridge (V1, reliable voxels, g = 4 primary; g = 1 and 8 secondary if runtime allows).
Models per condition c ∈ {Random, BP} × 5 seeds: bank alone (luminance-and-contrast bank, 2g² features), network alone (32g²), joint banded (bank and network, separate penalties α_bank, α_net on the 29-point grid thinned to 15 points 1e-6…1e8, factor 10, chosen jointly per voxel by inner-CV min MSE).
Unique contribution U_c = median r(joint banded) − median r(bank alone), per seed, averaged over subjects.
B1 (sanity): joint banded is not below bank alone by more than 0.002 for any condition, seed and subject. Violation means an implementation error, not a result.
B2: U_BP > 0 in at least 4 of 5 seeds and mean U_BP > 0.
B3: U_BP − U_Random > 0 in at least 4 of 5 seed pairs and mean > 0.
All values are reported whether or not B2/B3 are met; MSE skill reported alongside.

C. Cross-validated RSA on the 100 repeated test images (12 presentations each).
Brain RDMs per subject and ROI (V1 primary, all V1 voxels; LOC secondary): (b1) correlation distance on presentation-averaged patterns; (b2) Euclidean distance on presentation-averaged patterns; (b3) cross-validated squared Euclidean distance, d_ij = mean over ordered pairs of distinct sessions (A,B) of (x_i^A − x_j^A)·(x_i^B − x_j^B).
Model RDMs (global average pooling as in the paper cell): model units raw or standardised over the 100 stimuli; correlation distance or Euclidean distance.
Comparisons: Spearman between model and brain RDM over all 4950 pairs, per subject, then mean over subjects and seeds; Random − BP with stimulus-bootstrap 95% interval.
C1: raw model units, correlation distance vs b1: Random − BP > 0 (point estimate).
C2: standardised model units, correlation distance vs b1: Random − BP ≤ 0 (point estimate).
C3: Euclidean model distance: the sign of Random − BP is the same against b2 and against b3 (raw units and standardised units each).
Note: 100 stimuli give wide intervals; C1–C3 are fixed on point estimates, intervals are reported.

D. NSD feasibility (no analysis, no prediction): go/no-go report only.
