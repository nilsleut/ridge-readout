# Step 2B — which "Random" is which

Read-only. All paths and line numbers refer to the source repository
`evaluation-resolution-rsa` at HEAD `7a9b374` (working copy). Exploratory; sub-01 only
where data is involved.

## 1. Conditions in the 30 v12 checkpoints

`code/learning_rules_v12_allseeds_modal.py:115-116` defines six conditions:

    RULES = ["Random Weights", "Backprop", "Feedback Alignment", "Predictive Coding", "STDP",
             "Random Weights (BN-calibrated)"]

with seeds `[42, 123, 456, 789, 1337]` (`:121`, `seed_idx` 0-4). Checkpoints are written as
`model_weights_{rule_key(rule)}_seed{seed_idx}.pt` (`:613`, `rule_key` at `:483` =
lower-case, spaces to underscores) into `/results/outputs_allseeds/checkpoints`
(`:187-189`). The 30 local files (`data/v12_checkpoints/checkpoints/`, = `CKPT_V12_DIR`):

| Label | File pattern (seed_idx 0-4) |
|---|---|
| Random Weights | `model_weights_random_weights_seed{0..4}.pt` |
| Backprop | `model_weights_backprop_seed{0..4}.pt` |
| Feedback Alignment | `model_weights_feedback_alignment_seed{0..4}.pt` |
| Predictive Coding | `model_weights_predictive_coding_seed{0..4}.pt` |
| STDP | `model_weights_stdp_seed{0..4}.pt` |
| Random Weights (BN-calibrated) | `model_weights_random_weights_(bn-calibrated)_seed{0..4}.pt` |

The two random conditions differ only in BatchNorm running statistics: both skip
training (`RANDOM_RULES`, `:117`, `:560`); the BN-calibrated one runs 50 forward-only
CIFAR batches at 32 px in train mode and asserts the weights are unchanged (`:571-586`,
`N_BN_CALIB = 50` at `:118`).

## 2. The "Random" in the headline Random − BP = +0.030 [0.018, 0.042]

**Random Weights, without BN calibration.**

- The headline comes from `results/upsampling/step4_gaps.csv` line 7
  (`persub,V1,NATIVE,224,0.030446,0.018149,0.042293,…`), per
  `results/paper_v3/NUMBERS_MANIFEST_v3.md:15-17`.
- The rule pair is fixed in `code/upsampling/step3_bootstrap.py:29`:
  `CASES = {"V1": ("Conv1", "Random Weights", "Backprop"), …}`, and the gap is
  `a − b` in `code/upsampling/step4_evaluate.py:56-59`.
- The point estimate `gap.V1.224 = 0.030446` in `results/paper_v3/numbers_v3_part1.csv:57`
  uses `("V1", "Conv1", "rnd", "bp")` with `RK["rnd"] = "Random Weights"`
  (`code/paper_v3/compute_numbers.py:44`, `:121`).

RANDOM_PAPER = `Random Weights`.

## 3. tolerance_all_rungs.csv and the pilot abort

- **tolerance_all_rungs.csv, "Random Weights"** = feature key `random`
  (`code/ridge/tolerance_measurement.py:61-67`), files `pilot_features/model_random_seed{si}_g{g}.npy`
  (`:168`). The CSV also contains `Random Weights (BN-calibrated)` as a separate rule
  (key `random_bn_calibrated`).
- **The pilot abort** was at `Random Weights (BN-calibrated) s3 g1` = key
  `random_bn_calibrated`, seed_idx 3 (`code/ridge/pilot_fit.py:55-61`, `:146`).
- Features for both were extracted by `code/ridge/pilot_features_modal.py:44-50, 79, 143, 154`
  with `load_model(key, weights_dir=/results/outputs_allseeds/checkpoints, seed_idx=si)`;
  `code/cross_species/models_v2.py:430` maps `random` -> `model_weights_random_weights.pt`
  and `:436` maps `random_bn_calibrated` -> `model_weights_random_weights_(bn-calibrated).pt`,
  both with the `_seed{si}` suffix.

**So:** the "Random Weights" of tolerance_all_rungs.csv is the same condition and the same
checkpoints as RANDOM_PAPER. The condition at which pilot_fit aborted is the *other*
random condition (BN-calibrated), a different set of checkpoints. The ridge contrasts
reported so far (Random − BP = −0.029 at g = 1) therefore use the paper's Random.

**Same files on Modal and locally:** the features were extracted from the Modal volume,
the paper RDMs from the local copy (`results/upsampling/UPSAMPLING_REPORT.md:50-56`,
downloaded with `modal volume get`). Code alone cannot prove the copies are identical;
check (ii) of step 2C does: model RDMs from `pilot_features` (g = 1) equal the stored v12
NATIVE 224 Conv1 RDMs for all 30 checkpoints (Pearson 1.00000000, max |Δ| 5.6e-7).
