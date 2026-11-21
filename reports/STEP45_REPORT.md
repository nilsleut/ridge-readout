# Step 45: coarse spatial luminance (last why-test)

Predictions: `amendments/2026-11-21_step45-predictions.md` (W6, W7 and the interpretation rule),
`config_step45.yaml` (hash 347f9a0110a8). Datasets, pair sets, models, L, C, T and the partial
Spearman are those of step 44 (`config_step44.yaml`, 451099bd833e; `scripts/step44_why.py`), imported
unchanged. `config.yaml` (206fa875cca5) and the configs of steps 40–44 are unchanged.

LP-RDM: Rec. 709 grey image of `colour_statistic_control.load_rgb(path, 224)` → only_0-14 band of
arXiv:2608.12408 (`bandpass_filters.condition(x, bands(x), "only_0-14")`, including the mean) →
area-downsampled to 28 × 28 (`torch.nn.functional.interpolate`, mode `area`, i.e. 8 × 8 block
means) → Euclidean distance between the 784-dimensional vectors.

## Order of commits

| Commit | Time (local, +0100) | Content |
|---|---|---|
| `9648934` | 2026-11-21 09:41:21 | Add step 45 predictions (amendment, `config_step45.yaml`) |
| `62c1b0b` | 2026-11-21 09:43:51 | Add step 45 script with tests |
| this commit | | Add step 45 results |

The gate of Teil 0 ran before `9648934`; the run started after `62c1b0b`. Before every commit
`scripts/nsd_commit_check.py` ran (also as the pre-commit hook).

## Gate and tests (`tests/test_step45.py`)

| Check | Result |
|---|---|
| band-pass source: `src/vendor/bandpass_filters.py` is a verbatim copy of evaluation-resolution-rsa `code/bandpass/bandpass_extract.py` (`sigma`, `bands`, `condition`); sha256 of the current source equals the recorded one | passed |
| (i) gate, 20 step 40 gate images, only_0-14, Random and BP × 5 seeds, v12 Conv1: GAP features vs step 6 (`STEP5_GAP_DIR/bandpass`) | max \|d\| 0 |
| (i) RDM entries of the 190 pairs vs evaluation-resolution-rsa `data/bandpass/rdms/only_0-14` (10 RDMs) | max \|d\| 0 |
| (i) rho vs THINGS-720 V1 with the 20 rows replaced (cross-run, per subject) | max \|d\| 0 |
| (ii) LP-RDM without filter and downsampling = Euclidean distance of the grey images | max rel. \|d\| 0 |
| (iii) L + C partial through the step 45 path reproduces step 44 | +0.016276, \|d\| 8.3e-17 |

Checks inside the run: L + C at THINGS-720 V1 equals step 44 (\|d\| < 1e-12); test images without
covariates equal step 41 (−0.000159, \|d\| < 1e-9). `tests/test_step44.py --only i,ii,iii` passes.

## Predictions (`results/step45/summary.csv`)

| Rule | Criterion | Value | Result |
|---|---|---|---|
| W6 | THINGS-720 V1 Δ after partialling L, C and LP ≤ +0.010 | +0.0180 (from +0.0304) | not met |
| W7 | ρ(V1, LP) / LOSO bound: THINGS-720 > test (k = 1) and > NSD (k = 1) | 0.874 vs 0.356 and 0.170 | met |
| W3 (step 44) | | | met |
| W4 (step 44) | | | met |

**Interpretation (amendment rule): not identified.** W6 is not met. Explained fraction
1 − Δ(L + C + LP) / Δ = 1 − 0.0180 / 0.0304 = **0.408**. For comparison, L + C alone (step 44)
explains 0.464 and LP alone 0.125. Adding LP to L and C raises Δ again (+0.0163 → +0.0180): the
partial Random ρ rises slightly (0.0239 → 0.0253) while the BP ρ falls (0.0076 → 0.0073). Δ stays
positive in all 5 seeds and 3 subjects under every covariate set.

## THINGS-720 per subject (`results/step45/partial_lp.csv`)

| ROI | Subject | Δ raw | after L + C | after LP | after L + C + LP | after L + C + LP + T |
|---|---|---|---|---|---|---|
| V1 | sub-01 | +0.0330 | +0.0188 | +0.0276 | +0.0225 | +0.0223 |
| V1 | sub-02 | +0.0509 | +0.0241 | +0.0452 | +0.0253 | +0.0280 |
| V1 | sub-03 | +0.0074 | +0.0060 | +0.0070 | +0.0063 | +0.0065 |
| V1 | mean | +0.0304 | +0.0163 | +0.0266 | +0.0180 | +0.0190 |
| LOC | sub-01 | −0.0016 | −0.0007 | −0.0026 | −0.0006 | −0.0008 |
| LOC | sub-02 | −0.0242 | −0.0300 | −0.0268 | −0.0304 | −0.0315 |
| LOC | sub-03 | −0.0080 | −0.0070 | −0.0073 | −0.0071 | −0.0070 |
| LOC | mean | −0.0113 | −0.0126 | −0.0122 | −0.0127 | −0.0131 |

## LP by dataset (`results/step45/lp_by_dataset.csv`)

ρ(brain, LP): mean over subjects on the pair set of the LOSO bound (THINGS-720 P; test images all
pairs, mean over the 20 single-session draws; NSD k = 1 pair rule, mean over the 3 presentations).
LOSO bounds as in step 44 (`results/step44/luminance_by_dataset.csv`). ρ(model, LP): mean over seeds,
on the same pair sets as W3 of step 44.

| Dataset | ROI | ρ(brain, LP) | LOSO lower bound | ratio | ρ(Random, LP) | ρ(BP, LP) |
|---|---|---|---|---|---|---|
| THINGS-720 | V1 | 0.0450 | 0.0514 | 0.874 | 0.331 | 0.201 |
| THINGS test, k = 1 | V1 | 0.0121 | 0.0340 | 0.356 | 0.327 | 0.241 |
| NSD, k = 1 | V1 | 0.0456 | 0.2691 | 0.170 | 0.298 | 0.224 |
| THINGS-720 | LOC | 0.0015 | 0.0183 | 0.081 | 0.345 (Conv3) | −0.102 (Conv3) |
| THINGS test, k = 1 | LOC | 0.0022 | 0.0149 | 0.146 | 0.327 (Conv1) | 0.241 (Conv1) |
| NSD, k = 1 | LO-obj | 0.0052 | 0.0843 | 0.062 | 0.298 (Conv1) | 0.224 (Conv1) |

THINGS-720 V1 per subject: 0.056, 0.070, 0.009. Random exceeds BP in all 5 seeds in every dataset
(Random 0.25–0.39, BP −0.12–0.25). The Random − BP difference in LP sensitivity (0.07–0.13 at V1) is
smaller than for global luminance in step 44 (0.30–0.34).

## Same partialling at test images and NSD (reported without prediction)

| Dataset | ROI | Δ | after L + C | after LP | after L + C + LP |
|---|---|---|---|---|---|
| THINGS test, k = 1 | V1 | −0.0002 | −0.0014 | −0.0009 | −0.0003 |
| THINGS test, k = 1 | LOC | −0.0004 | +0.0005 | −0.0006 | +0.0013 |
| NSD, k = 1 | V1 | −0.0120 | −0.0155 | −0.0154 | −0.0095 |
| NSD, k = 1 | LO-obj | −0.0081 | −0.0089 | −0.0087 | −0.0082 |

NSD: all 8 subjects negative under every set. The NSD k = 1 Δ here uses the step 43 pair rule
(common to all subjects), not the per-subject rule of step 42; it is therefore not a reproduction of
the step 42 k = 1 value (P4: −0.0120), although it agrees with it to four decimals.

## Files

| File | Content |
|---|---|
| `results/step45/partial_lp.csv` | Δ per dataset, ROI and covariate set; THINGS-720 per subject; checks |
| `results/step45/lp_by_dataset.csv` | ρ(brain, LP), bound, ratio, ρ(model, LP) per seed and per subject |
| `results/step45/summary.csv` | W6, W7, W3/W4 of step 44, interpretation, runtimes |

Every CSV carries `config_hash` (347f9a0110a8), `config_hash_44` (451099bd833e), `git_head` and
`seed` (model seed where a row has one, else empty). Per-image LP vectors are not written; NSD values
are aggregated only.

## Runtime

166 s in total (THINGS-720 51 s, test images 8 s, NSD 107 s); limit 4 h.

## Open points

- LP is built on the grey image of `load_rgb` (PIL bilinear resize, as L and C in step 44), while
  the gate (and arXiv:2608.12408) filters the RGB tensor after torchvision `Resize`. The filter
  functions are identical; the resize implementation differs.
- `bands` is applied to a one-channel (1, 224, 224) array; `gaussian_filter` with sigma (0, σ, σ)
  filters each channel separately, so this equals filtering the grey image.
- No existing area-downsampling function was found in the repositories; `torch.nn.functional.interpolate`
  (mode `area`) was used.
- The test-image and NSD Δ values are k = 1 (single presentations).
- Log (ignored by git): `data/step45_run.log`.
