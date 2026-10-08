# Step 4 — channel scale (S1) and the other two subjects (S2)

**Exploratory.** Predictions and criteria: `amendments/2026-10-03_step4-predictions.md`
(commit `f148085`, made before any step 4 script). Config hash `206fa875cca5`. V1,
NATIVE 224 px, Conv1, g = 1, the 30 checkpoints of `pilot_features/`; "Random" is Random
Weights without BN calibration. Values are mean ± SD over the 5 seeds, with the number of
positive seeds in brackets where given.

## S1 — RSA on channel-standardized features, sub-01 (`results/step4/std_rsa.csv`)

The pair set (210,205 cross-run pairs) and the statistic are as in step 2C, using the
same functions. The `raw` variant reproduces `results/step2/rsa_501.csv` to 1.0e-16.
Dead channels: **0** in every checkpoint and variant.

| Condition | raw | **z** (primary) | center | scale |
|---|---|---|---|---|
| Backprop | 0.0200 | 0.0417 | 0.0225 | 0.0498 |
| Random Weights | 0.0478 | 0.0266 | 0.0235 | 0.0352 |
| Random (BN-cal) | 0.0008 | 0.0250 | 0.0222 | 0.0180 |
| Feedback Alignment | 0.0015 | 0.0234 | 0.0219 | 0.0176 |
| Predictive Coding | 0.0055 | 0.0273 | 0.0235 | 0.0280 |
| STDP | 0.0233 | 0.0280 | 0.0236 | 0.0306 |

| Difference | raw | **z** | center | scale |
|---|---|---|---|---|
| Random − BP | +0.0278 ± 0.0109 (5/5) | **−0.0151 ± 0.0027 (0/5)** | +0.0010 ± 0.0027 | −0.0146 ± 0.0072 (0/5) |
| BN-cal − BP | −0.0192 ± 0.0058 | −0.0167 ± 0.0029 | −0.0003 ± 0.0014 | −0.0318 ± 0.0083 |
| Random − BN-cal | +0.0470 ± 0.0071 | **+0.0017 ± 0.0003** | +0.0013 ± 0.0018 | +0.0172 ± 0.0016 |
| FA − BP | −0.0185 ± 0.0085 | −0.0183 ± 0.0048 | −0.0007 ± 0.0041 | −0.0322 ± 0.0096 |
| PC − BP | −0.0145 ± 0.0036 | −0.0145 ± 0.0022 | +0.0009 ± 0.0012 | −0.0218 ± 0.0058 |
| STDP − BP | +0.0033 ± 0.0070 | −0.0137 ± 0.0059 | +0.0011 ± 0.0014 | −0.0191 ± 0.0131 |

**Applying S1:** D_std = −0.0151 ≤ 0.010, so **H_scale is supported**. Under z the sign
reverses (0/5 seeds positive), which matches ridge (−0.0288). The control criterion
|Random − BN-cal| = 0.0017 ≤ 0.010 is **met**.

Further observations (descriptive, no rule):
- Centring alone (`center`) collapses all six conditions to 0.022–0.024, and every
  difference becomes |Δ| ≤ 0.0013.
- Scaling alone (`scale`) already gives Random − BP < 0.
- So both the channel means and the channel SDs carry the raw Random advantage. "Scale"
  in H_scale is therefore to be read as first and second moments per channel, not SD
  alone.

**Channel scale, descriptive** (`results/step4/channel_scale.csv`; CV of the 32 channel
SDs over the 720 stimuli, mean over seeds):

| Condition | BP | Random | BN-cal | FA | PC | STDP |
|---|---|---|---|---|---|---|
| CV | 0.506 | 0.507 | 0.294 | 0.272 | 0.301 | 0.453 |

The CV of Random and BP is the same, so this summary does not separate them. Random has
the smallest channel SDs overall (min 0.057, max 0.399), which the CV does not capture.

## S2 — sub-02 and sub-03 (`results/step4/subjects.csv`)

**Checks:**
- reliable V1 voxels: 518 (sub-02) and 286 (sub-03), as expected;
- `stim_order_sub-0X.txt` is identical to `stim_order_sub-01.txt`;
- the RDM of all V1 voxels reproduces `fmri_rdm_V1_sub-0X.npy` with Pearson 1.00000000.

**Run:** the runtime estimate was 7.1 min, so both subjects were run. Ridge uses
`fit_condition` unchanged with default folds, g = 1 and D1 in counting mode; only the
data loading is per subject.

| Subject (reliable voxels) | metric | Random − BP | BN-cal − BP | Random − BN-cal |
|---|---|---|---|---|
| sub-01 (501) | ridge | −0.0288 ± 0.0082 (0/5) | −0.0305 ± 0.0108 | +0.0017 ± 0.0027 |
| | RSA raw | +0.0278 ± 0.0109 (5/5) | −0.0192 ± 0.0058 | +0.0470 ± 0.0071 |
| | RSA z | −0.0151 ± 0.0027 (0/5) | −0.0167 ± 0.0029 | +0.0017 ± 0.0003 |
| sub-02 (518) | ridge | −0.0227 ± 0.0042 (0/5) | −0.0242 ± 0.0044 | +0.0014 ± 0.0007 |
| | RSA raw | +0.0562 ± 0.0162 (5/5) | −0.0197 ± 0.0113 | +0.0758 ± 0.0116 |
| | RSA z | −0.0084 ± 0.0033 (0/5) | −0.0108 ± 0.0035 | +0.0024 ± 0.0002 |
| sub-03 (286) | ridge | −0.0113 ± 0.0059 (0/5) | −0.0079 ± 0.0062 | −0.0035 ± 0.0025 |
| | RSA raw | +0.0085 ± 0.0028 (5/5) | −0.0047 ± 0.0012 | +0.0132 ± 0.0026 |
| | RSA z | −0.0057 ± 0.0015 (0/5) | −0.0058 ± 0.0016 | +0.0001 ± 0.0001 |
| mean of 3 subjects | ridge | −0.0209 ± 0.0056 (0/5) | −0.0208 ± 0.0063 | −0.0001 ± 0.0009 |
| | RSA raw | +0.0308 ± 0.0099 (5/5) | −0.0145 ± 0.0058 | +0.0453 ± 0.0069 |
| | RSA z | −0.0097 ± 0.0023 (0/5) | −0.0111 ± 0.0025 | +0.0014 ± 0.0002 |

Ridge median r per condition (mean over seeds):

| Subject | BP | Random | BN-cal | FA | PC | STDP |
|---|---|---|---|---|---|---|
| sub-02 | 0.0964 | 0.0737 | 0.0722 | 0.0684 | 0.0747 | 0.0716 |
| sub-03 | **0.0019** | −0.0094 | −0.0060 | −0.0111 | −0.0053 | −0.0079 |

**Applying S2 (per subject):**
- **sub-02: pattern holds.** RSA raw +0.0562 > 0, ridge −0.0227 < 0.
- **sub-03: pattern holds, weaker.** RSA raw +0.0085 > 0, ridge −0.0113 < 0. The subject
  has only 286 reliable voxels. Beyond that, ridge does not predict sub-03's reliable V1
  voxels at all: median r is ≤ 0.002 in every condition. The ridge side of the pattern
  for sub-03 is therefore a difference between near-zero, partly negative, scores, and
  carries little weight.

The z-standardized RSA reverses the sign of Random − BP in all three subjects (0/5 seeds
positive in each), and Random − BN-cal under z is ≤ 0.0024 in every subject.

## Lower-edge gate (`results/step4/lower_edge_gate_subjects.csv`)

There was one hit: sub-02, Feedback Alignment seed_idx 2, voxel 867, outer fold 1, with
α_min,diag = 1e-9 and relative gain 3.0e-5. It is classified **Plateau**. No cell was
excluded. sub-03 had no hits.

## Limits

- **Inference unit is the seed** (n = 5); there is no stimulus bootstrap, so no CI over
  stimuli.
- **N = 3 subjects.** All statements concern stimulus variation in these subjects, not
  the population.
- sub-03 is weak, for the two reasons given under S2: 286 reliable voxels, and ridge r
  near zero.
- Run-grouped folds (step 3, E3) are not preregistered; this step uses only random
  folds.
- S1 was tested on sub-01 only as specified. The z variant for sub-02 and sub-03 is
  reported, but no rule was fixed for it.
- Everything is exploratory: g = 1, Conv1, 224 px only.
