# Step 41: predictions (fixed before scripts and runs)
Date: 2026-11-16. Configuration: config_step41.yaml, config hash 8458e04a269a.

Motivation: B1 of step 40 was violated; part C showed no untrained advantage on presentation-averaged test images.

A. Diagnosis of B1 (V1, reliable voxels, g = 4, Random and BP, 5 seeds, 3 subjects).
D-A1 (code check on real data): the joint banded path with α_net fixed at 1e12 reproduces bank-alone median r within 1e-6 in every fit. Failure means an implementation error.
D-A2 (selection variance): with a conditional one-dimensional search (α_bank fixed per voxel and outer fold at the value selected by bank alone; α_net searched over the 15-point axis by inner-CV min MSE), no fit lies below bank alone by more than 0.002.
If D-A1 and D-A2 hold, the B1 violation of step 40 is attributed to selection variance of the two-dimensional search. The unique contribution U under the conditional search is then reported as an exploratory, post hoc estimator; B2 and B3 are applied to it and labelled as such. The two-dimensional estimator of step 40 remains not interpreted.
Also reported: inner-CV MSE of the selected penalty versus outer-fold MSE (optimism), for the 2D and the conditional search.

N. Noise curve on the 100 repeated test images (V1 primary, LOC secondary; all ROI voxels; same voxel normalisation as in step 40 part C).
For k ∈ {1, 2, 3, 4, 6, 12}: per draw, choose k of the 12 sessions at random (same sessions for all images in a draw), average each image's presentations from those sessions, build the brain RDM. 20 draws per k (seed 20261116); k = 12 is a single draw. Model RDMs as in step 40 part C. Spearman over 4950 pairs, per subject, then mean over draws, subjects and seeds. Stimulus-bootstrap interval (1000 resamples, seed 20261002) for k = 1 and k = 12.
Contrast Δ(k) = Random − BP.
N1: raw units, correlation distance, V1: Δ(1) > Δ(12) (point estimates).
N2: same cell: Spearman correlation between k and Δ(k) over the six values of k is ≤ −0.8.
N3: same cell: Δ(1) > 0 (point estimate).
N4: standardised units, correlation distance, V1: Δ(k) ≤ 0 for every k (point estimates).
All other cells (Euclidean, raw and standardised; LOC) are reported without predictions.
