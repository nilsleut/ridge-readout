# Step 7 offset predictions (fixed before runs)

Date: 2026-10-09. Committed before any step 7 script exists or runs. Exploratory.
Conventions as in step 5: per subject, then averaged; all voxels of the ROI; cross-run
pairs; stimulus bootstrap with 1000 resamples, seed 20261002, paired differences from
the same draws. No new ridge fits, no new feature extraction: everything comes from
`$STEP5_GAP_DIR` (including `bandpass/`) and `arm_b_out/`.

## Hypothesis H_off

Correlation and cosine distance across units are not invariant to constant per-unit
offsets (over the evaluated stimulus set); euclidean distance is. The resolution
dependence of the V1 gap in arXiv:2608.12408 arises from offsets on the model and the
brain side and is mediated by a shared per-stimulus "gain axis".

## Definitions

- **Feature normalisation** F ∈ {raw, center, scale, z}, per unit over the 720 stimuli.
- **Distance** D ∈ {correlation, cosine, euclidean}.
- **fMRI normalisation** B ∈ {paper, center720, z720}:
  - paper: betas as in `arm_b_out`, z-scored over all trials;
  - center720 and z720: per voxel over the 720 stimuli.
- **Resolution slope** S = Gap(NATIVE 224) − Gap(NATIVE 32), with Gap = Random − BP at V1.
- **Gain axis.** For a matrix X (720 × units):
  - μ = column means over the 720 stimuli; μc = μ − mean(μ);
  - x̃_i = row i, centred across units;
  - gain_i = ⟨x̃_i, μc⟩ / (‖x̃_i‖ · ‖μc‖);
  - gain RDM = |gain_i − gain_j|.

  For models on the raw features; for fMRI on the betas under B = paper.

## Predictions

**O1 (mathematical control).**
- For D = euclidean, the RDMs under F = raw and F = center are identical
  (max |Δ| < 1e-9).
- Correspondingly for B = paper vs center720 at D = euclidean, but only if the betas
  differ by offsets alone; otherwise report only.
- A violation of O1 stops the run.

**O2 (slope across the multiverse, primary).**
- S with bootstrap CI is computed for all 36 cells F × D × B at V1.
- Prediction: a CI of S above 0 together with S ≥ 0.5 · S_paper (S_paper = +0.0311)
  occurs only in cells where D ∈ {correlation, cosine} AND neither F nor B is centred
  (F ∈ {raw, scale} AND B = paper).
- In all other cells, the CI of S contains 0 or S < 0.5 · S_paper.
- Met if at most 2 of the 36 cells violate the prediction; every violation is reported
  individually.

**O3 (LOC, primary).** BP − Random at LOC, NATIVE 224, is > 0 (point estimate) in at
least 30 of the 36 cells and has a CI above 0 in at least 18.

**G1 (mechanism).**
- Cells: condition {Random, BP} × seed × resolution {six NATIVE, UPSAMPLED 224}, n = 70.
- c_gain = Pearson over the 720 stimuli between model gain and voxel gain (V1, per
  subject, then averaged).
- Prediction: Spearman(c_gain, raw RSA) ≥ 0.6, with raw RSA at F = raw,
  D = correlation, B = paper, V1.

**G2 (mechanism, causal via partialling).**
- Partial RSA = Spearman between model RDM and fMRI RDM after each has been
  rank-regressed over the pairs on its own gain RDM (residuals of the linear regression
  of the ranks).
- Prediction at F = raw, D = correlation, B = paper:
  - upper CI bound of the V1 gap at NATIVE 224 ≤ 0.010;
  - S_partial ≤ 0.5 · S_paper.

## Limits

N = 3 subjects; inference over stimuli. The sign at V1 is not interpreted as a statement
about learning rules.

---

## Operationalisation notes (added by the executor before any run)

Recorded here because the text above leaves these points open. They are fixed before
any step 7 script exists.

1. **D applies to both sides.** It is used for the model RDM (on the F-normalised
   features) and for the fMRI RDM (on the B-normalised betas). This is what makes the
   B part of O1 well-defined.
2. **Partial RSA (G2).** Per subject: the ranks of the model RDM and of the fMRI RDM
   (on the pairs of the draw) are each regressed on [1, ranks of their own gain RDM];
   the Pearson correlation of the two residual vectors is the partial Spearman ρ; then
   mean over subjects.
3. **Gain for the fMRI side.** Computed from all voxels of the ROI under B = paper, per
   subject.
4. **Stimulus luminance and RMS contrast (descriptive).** Computed as in the step 6
   stimulus spectra (`_tf` without Normalize; Y = 0.299 R + 0.587 G + 0.114 B): mean of
   Y and SD of Y per image. NATIVE 224 is used for the voxel gain; the matching
   resolution is used for the model gain at NATIVE 32 and 224.
