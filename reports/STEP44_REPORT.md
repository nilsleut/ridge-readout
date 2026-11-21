# Step 44: why does the THINGS-720 set show an untrained advantage?

Predictions: `amendments/2026-11-19_step44-predictions.md` (W1–W5 and the interpretation rule),
`config_step44.yaml` (hash 451099bd833e). Feasibility and gates: `reports/STEP44_SETUP.md`.
`config.yaml` (206fa875cca5) and the configs of steps 40–43 are unchanged.

Paper cell throughout: Random − BP, raw units, correlation distance, Spearman per subject, mean
over seeds 0–4, then subjects. THINGS-720: V1 ← Conv1, LOC ← Conv3, NATIVE 224, cross-run pairs
P (210,205). THINGS test images: 100 images, k = 1 (the 20 single-session draws of step 41), all
4950 pairs, GAP Conv1 for V1 and LOC. NSD: 515 images, 8 subjects, k = 1 with the step 43 pair rule
(no subject has the two presentations in the same session and run; mean 131,924 pairs), GAP Conv1
for V1 and LO-obj. Point estimates throughout; no intervals were predicted.

## Order of commits

| Commit | Time (local, +0100) | Content |
|---|---|---|
| `d958a05` | 2026-11-21 09:15:50 | Add step 44 predictions (amendment, `config_step44.yaml`, `reports/STEP44_SETUP.md`) |
| `80eef39` | 2026-11-21 09:24:06 | Add step 44 scripts with tests |
| `7777f32` | 2026-11-21 09:25:06 | Fix subset pairs in step 44 |
| this commit | | Add step 44 results |

The runs started after `7777f32`. `80eef39` built subset pairs with `common_up_pairs.pair_set`,
which accepts resamples of size 720 only; the runtime estimate stopped on it before any result
was computed. `7777f32` takes P from `pair_set` and keeps the pairs with both images in the subset,
and adds a brute-force test of the subset pairs. Before every commit `scripts/nsd_commit_check.py`
ran (also as the pre-commit hook).

## Tests (`tests/test_step44.py`, before the runs)

| Check | Result |
|---|---|
| (i) partial Spearman = first- and second-order partial-correlation formulas on permutations | passed, max \|d\| 2.7e-14 |
| (i) common cause z partialled out (n = 20,000) | passed: raw 0.484, partial +0.0088 |
| (ii) no covariate = Spearman (scipy and `rho_pairs`, with ties) | passed, max \|d\| 3.3e-16 |
| (iii) subset code with the full 720 set reproduces the paper cell | passed: 0.030445938, \|d\| 7.6e-17, 210,205 pairs |
| (iii) subset pairs = brute force (3 subsets of 100) | passed, 12,053 pairs |
| (iv) order gates (one session each, sub-01 = sub-02, sub-03 differs) | passed |
| (iv) luminance RDM vs THINGS-720 V1 = 0.054 ± 0.001 | passed: 0.053627 |

Checks inside the run: the THINGS-720 paper cell equals step 5 (0.030446, \|d\| < 1e-12); the
test-image k = 1 Δ equals step 41 (−0.000159, \|d\| < 1e-9). Existing suites unchanged and passing
(phase 2 97/97, loader 41/41, lower edge 2/2, banded 7/7, cvrsa 4/4, noise curve 4/4,
`test_nsd --skip-gate` 7/7, `test_nsd43 --only i,iii,iv` 3/3).

## Predictions (`results/step44/summary.csv`)

| Rule | Criterion | Value | Result |
|---|---|---|---|
| W1 | Δ > −0.0002 in ≥ 95 % of 1000 subsets of 100 | 97.0 % (mean +0.0302, 2.5–97.5 % [−0.0017, +0.0654]) | met |
| W2 | THINGS-720 V1 Δ after partialling L and C ≤ +0.010 | +0.0163 (from +0.0304) | not met |
| W3 | Spearman(model, L) Random > BP in 5/5 seeds, each dataset | THINGS-720 5/5, test 5/5, NSD 5/5 | met |
| W4 | ρ(V1, L) / LOSO bound: THINGS-720 > test (k = 1) and > NSD (k = 1) | 1.043 vs 0.256 and 0.075 | met |
| W5 | \|Δ(T partialled) − Δ\| < 0.005 | 0.0007 (+0.0304 → +0.0311) | met |

**Interpretation (amendment rule): not identified.** The luminance attribution requires W2, W3
and W4; W2 is not met. W5 and W1 are met, so neither "presentation order contributes" nor "sampling
variation" applies. No statement of the rule applies, which the amendment reports as not identified.

Descriptively: partialling L and C removes 46 % of the V1 advantage (+0.0304 → +0.0163; Random ρ
0.0532 → 0.0239, BP 0.0227 → 0.0076), and Δ stays positive in all 5 seeds and all 3 subjects.

## Per subject, THINGS-720 (`results/step44/partial.csv`)

| ROI | Subject | Δ raw | after L + C | after T | after L + C + T |
|---|---|---|---|---|---|
| V1 | sub-01 | +0.0330 | +0.0188 | +0.0329 | +0.0186 |
| V1 | sub-02 | +0.0509 | +0.0241 | +0.0526 | +0.0265 |
| V1 | sub-03 | +0.0074 | +0.0060 | +0.0078 | +0.0062 |
| V1 | mean | +0.0304 | +0.0163 | +0.0311 | +0.0171 |
| LOC | sub-01 | −0.0016 | −0.0007 | −0.0020 | −0.0009 |
| LOC | sub-02 | −0.0242 | −0.0300 | −0.0263 | −0.0312 |
| LOC | sub-03 | −0.0080 | −0.0070 | −0.0081 | −0.0070 |
| LOC | mean | −0.0113 | −0.0126 | −0.0121 | −0.0130 |

The positive V1 Δ comes mainly from sub-01 and sub-02 (shared order); sub-03 is small but positive.

## Luminance by dataset (`results/step44/luminance_by_dataset.csv`)

ρ(V1, L-RDM): mean over subjects on the pair set of the LOSO bound. ρ(model, L-RDM): mean over
seeds (subject-independent; THINGS-720 on P, test and NSD on all pairs).

| Dataset | ROI | ρ(brain, L) | LOSO lower bound | ratio | ρ(Random, L) | ρ(BP, L) |
|---|---|---|---|---|---|---|
| THINGS-720 | V1 | 0.0536 | 0.0514 | 1.043 | 0.615 | 0.276 |
| THINGS test, k = 1 | V1 | 0.0087 | 0.0340 | 0.256 | 0.636 | 0.318 |
| NSD, k = 1 | V1 | 0.0201 | 0.2691 | 0.075 | 0.631 | 0.336 |
| THINGS-720 | LOC | 0.0009 | 0.0183 | 0.050 | 0.674 (Conv3) | 0.016 (Conv3) |
| THINGS test, k = 1 | LOC | −0.0026 | 0.0149 | −0.173 | 0.636 (Conv1) | 0.318 (Conv1) |
| NSD, k = 1 | LO-obj | 0.0028 | 0.0843 | 0.033 | 0.631 (Conv1) | 0.336 (Conv1) |

Per seed, Random exceeds BP in all 5 seeds in every dataset and layer (range Random 0.56–0.73,
BP 0.00–0.36). The model property is the same in all three datasets; only THINGS-720 has a V1 RDM
whose luminance correlation reaches the lower bound of its noise ceiling.

## Presentation order (`results/step44/order.csv`)

| Subject | Spearman(position, L) | ρ(T-RDM, V1) | ρ(T-RDM, LOC) | ρ(T-RDM, Random Conv1) | ρ(T-RDM, BP Conv1) |
|---|---|---|---|---|---|
| sub-01 | 0.033 | −0.008 | 0.020 | 0.004 | 0.012 |
| sub-02 | 0.033 | 0.139 | 0.091 | 0.004 | 0.012 |
| sub-03 | 0.031 | 0.021 | −0.006 | −0.006 | 0.014 |

ρ(T-RDM, model): mean over seeds on P (Conv3, LOC: Random 0.009 / BP −0.013 for sub-01 and sub-02,
−0.012 / −0.001 for sub-03). The sub-02 V1 RDM carries a clear order component (0.139), but the
model RDMs barely correlate with T, so partialling T leaves Δ almost unchanged (W5).

## LOC and LO-obj (reported without prediction)

| Quantity | Value |
|---|---|
| W1 at LOC: fraction of subsets with Δ > −0.0002 | 22.9 % (mean −0.0113, 2.5–97.5 % [−0.0403, +0.0170]) |
| W2 at LOC: Δ after L + C | −0.0126 (from −0.0113) |
| W3 at LOC / LO-obj | 5/5 seeds in each dataset |
| W4 at LOC / LO-obj: ratio | THINGS-720 0.050, test −0.173, NSD 0.033 |
| W5 at LOC: \|Δ(T) − Δ\| | 0.0008 |

## Files

| File | Content |
|---|---|
| `results/step44/subsets.csv` | Δ per subset (V1, LOC; mean and per subject), pair counts, summary |
| `results/step44/partial.csv` | THINGS-720 ρ per subject, model and covariate set; Δ per subject and mean |
| `results/step44/luminance_by_dataset.csv` | ρ(model, L) per seed, ρ(brain, L), LOSO bound, ratio; checks |
| `results/step44/order.csv` | Spearman(position, L), ρ(T-RDM, brain), ρ(T-RDM, model) per subject |
| `results/step44/summary.csv` | W1–W5, interpretation, runtimes |

Every CSV carries `config_hash` (451099bd833e), `git_head` and `seed` (model seed where a row has
one, else empty; the subset seed 20261119 is in the config). Per-image L and C are not written; NSD
values are aggregated only.

## Runtime

Estimate before the run: 0.28 h (× 1.5; stop at 4 h). Actual: 178 s in total (setup 32 s, W1 98 s,
partial 28 s, luminance 18 s, order 2 s).

## Open points

- Position within the session is the row order of the THINGS-fMRI stimulus metadata (no onsets
  on disk; catch trials absent; see `STEP44_SETUP.md`).
- W3 at test images and NSD uses all pairs; at THINGS-720 the cross-run pairs P. The THINGS test
  and NSD features are GAP Conv1 for both ROIs (as in steps 41 and 42), so their LOC / LO-obj model
  values equal the V1 values.
- The test-image LOSO bound is the mean over the 20 single-session draws of the per-draw bound over
  3 subjects; the NSD bound is the k = 1 value of `results/step43/ceiling_nsd.csv`.
- The amendment file is named 2026-11-19 as specified; it was written and committed on 2026-11-21.
- Log (ignored by git): `data/step44_run.log`.
