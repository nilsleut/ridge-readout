# Step 9 separation predictions (fixed before runs)

Date: 2026-10-10. Committed before any step 9 script exists or runs. Exploratory.
RSA convention as in steps 7 and 8: per subject, then averaged; all voxels of the ROI;
cross-run pairs; stimulus bootstrap with 1000 resamples, seed 20261002, paired
differences from the same draws. No new feature extraction: own CNN from
`$STEP5_GAP_DIR`, ImageNet models from `data/step8_gap/`, fMRI from `arm_b_out/`.

## Decomposition

For a matrix X (720 × units):
- Xc = X with the per-unit mean over the 720 stimuli subtracted;
- nᵢ = ‖Xc_i‖;
- cos θᵢⱼ = cosine between Xc_i and Xc_j.

Exact identity:

‖Xc_i − Xc_j‖² = (nᵢ − nⱼ)² + 2 nᵢ nⱼ (1 − cos θᵢⱼ).

Component RDMs:
- **norm RDM** Nᵢⱼ = |nᵢ − nⱼ| (gain magnitude);
- **angle RDM** Aᵢⱼ = 1 − cos θᵢⱼ (geometry).

The same decomposition is applied to the fMRI betas (centring per voxel over the 720
stimuli). Primary: F = center before the decomposition. Secondary: F = z (units
additionally scaled to SD 1). The RSA of a component is the Spearman correlation
between the model and the fMRI component RDM of the same kind.

## Contrasts

- Own CNN: Random − BP (5 seeds).
- ImageNet models: untrained (mean over 3 init seeds) − trained.
- Primary resolution: own CNN NATIVE 224; ImageNet 224.

## Predictions

**SEP1 (gate).** The identity holds numerically (max relative error < 1e-8) for all
models, conditions and the fMRI side. Otherwise STOP.

**SEP2 (gain magnitude).** V1, norm RDM, F = center: contrast > 0 with CI above 0 in
the own CNN and in at least 4 of 6 ImageNet models.

**SEP3 (geometry).** V1, angle RDM, F = center: contrast < 0 with CI below 0 in the own
CNN and in at least 4 of 6 ImageNet models.

**SEP4 (LOC geometry).** LOC, angle RDM, F = center: trained − untrained (own CNN:
BP − Random) > 0 with CI above 0 in the own CNN and in at least 5 of 6 ImageNet models.

**Interpretation rule.**
- If SEP2 and SEP3 are both met, the V1 advantage of untrained networks under euclidean
  distance sits in the gain magnitude, and in geometry trained networks are ahead.
- Otherwise the V1 ranking remains unidentifiable, with no attribution to gain or
  geometry.

## Descriptive, no rule

- (a) F = z for SEP2 to SEP4.
- (b) Resolution slope of the contrast for the norm and the angle RDM (own CNN:
  224 − 32; ImageNet: 448 − 224, without vit_b_16), with CI, to see in which component
  the slope from GEN4 sits.
- (c) |S_gen| under z/correlation from step 8, reported as a number per model.
- (d) Pearson over the 720 stimuli between nᵢ (model) and luminance and RMS contrast
  per condition; and between nᵢ (voxels, V1) and the same stimulus statistics.
- (e) RSA of the full euclidean RDM on Xc, to check that it matches the cell
  center/euclidean/center720 from step 7 or step 8 (deviation > 1e-6: report).

## Limits

N = 3 subjects; inference over stimuli. The labels "norm = gain, angle = geometry" are a
definition, not a biological claim.

---

## Operationalisation notes (added by the executor before any run)

1. **F applies to both sides**, as D did in step 7:
   - F = center: model units and fMRI voxels are each centred over the 720 stimuli;
   - F = z: model units and fMRI voxels are each additionally scaled to SD 1 (SD < 1e-8
     → 0). For F = z the identity and the components are computed on the z-scored
     matrix, which is centred by construction.
   - fMRI starting point: the `arm_b_out` betas, i.e. B = paper before this
     normalisation.
2. **Component RSA**: rho = ranks(model component[P]) · ranks(fMRI component[P]) on the
   cross-run pair set P of the draw (the statistic of all earlier steps), per subject,
   then the mean.
3. **(e) references**:
   - own CNN: `results/step7/multiverse.csv`, kind = rho, F = center, D = euclidean,
     B = center720, same ROI / condition / checkpoint;
   - ImageNet: `results/step8/generalization.csv`, kind = rho, same F / D / B, model,
     condition, R and ROI.
4. **(d) stimulus statistics** as in steps 7 and 8 (Y = 0.299 R + 0.587 G + 0.114 B on
   the pre-normalisation tensor; mean = luminance, SD = RMS contrast):
   - own CNN at its resolution (v12 NATIVE transform);
   - ImageNet models at R with the step 8 transform;
   - voxels: v12 NATIVE 224 stimuli, as in step 7.
5. **Slopes (b)** are bootstrapped for both F if the runtime allows (rule e of the
   task); ImageNet slopes exclude vit_b_16.
