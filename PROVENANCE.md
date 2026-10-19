# Provenance

## Source

| | |
|---|---|
| Source repository | `evaluation-resolution-rsa` (local working copy) |
| Source HEAD at import | `7a9b3742cf4d739400bd52d0d6eb8a53c1bd517c` |
| Base commit per `config.yaml` | `6cf64f8` (2026-08-14) |
| Config hash | `206fa875cca5` (`schema_version: 2`) |
| Import date | 2026-10-03 |

The preregistration, its amendments, `config.yaml`, `config.lock`, the reports and the
ridge code were **untracked** in the source repository. No commit there contains them.

No public registration exists; dating rests on in-text date stamps and the config hash, so the correct wording is 'criterion fixed before the analysis'.

## Files copied byte-identically

| File here | Source path | sha256 |
|---|---|---|
| `PREREGISTRATION.md` | `PREREGISTRATION.md` | `10d01322e488de8adbb7cc80842434a22d8ec2442dfc1a3c88a05acef5047956` |
| `config.yaml` | `config.yaml` | `77ee40d831802a7124e5b12a1f650895a3bf349b7a7eac41fae4813c18ca948e` |
| `config.lock` | `config.lock` | `5593e9e1c68c0fcf70c4e535f37abd864d12ed7a6dcde586df0934e6f2612abb` |
| `reports/PHASE0_RECONNAISSANCE.md` | `PHASE0_RECONNAISSANCE.md` | `ad5e2420eeb75ea513734ee77a06643a2ae4ef400234400a93b41f837cd87d8f` |
| `reports/PRECONDITIONS_P1_P4.md` | `PRECONDITIONS_P1_P4.md` | `d58ef03a931a82a225c2224902aa7fa294097c6c367697bfb20ba6f6413e459e` |
| `reports/PHASE2_REPORT.md` | `PHASE2_REPORT.md` | `e465b9aaeabc8b70624b657a41c1524d2a0519c8389759d1c855f0010057e071` |
| `reports/PHASE2B_REPORT.md` | `PHASE2B_REPORT.md` | `0c66e61fd782684f627a4e8ed4b7302b9b41345304f3081a0b3c048576d62792` |
| `reports/PILOT_REPORT.md` | `PILOT_REPORT.md` | `1945d95b7a62ae42d7afb9b9333b69cfff2adb361995e396a0bc17d077641b53` |
| `reports/TOLERANCE_REPORT.md` | `TOLERANCE_REPORT.md` | `41e04b450152e6108a635598e2e6df11f8634dbf1d29f2e341367e60d259ff4e` |
| `results/tolerance_all_rungs.csv` | `tolerance_all_rungs.csv` | `0e1cf4b72f3bb16801848371bb88779b863ee4dfa5aa44f9537fa8403baa54a0` |

These files are stored with their original CRLF line endings (`core.autocrlf=false` in
this repository), so the committed blobs hash to the values above.

## Source files of the code (originals)

Copied into `src/`, `scripts/` and `tests/`. Changes are limited to imports, `sys.path`
and path resolution, and are listed line by line in `CHANGES_FROM_ORIGIN.md`.

| Source path | sha256 of the original |
|---|---|
| `code/ridge/arm_b_betas.py` | `3305b4b511cd7e49a168318126b0463c2b617fc39b6e610e93ba5bcee70e4aa5` |
| `code/ridge/boundary_characterisation.py` | `ccfc295b81715d2f690840c02df2d2ed4ba60b0a613b9e8137434f3d4d409efd` |
| `code/ridge/check_csv_reproduction.py` | `f45bb64f38ac0f826118087a60f9dab08c3aa4d4fa4e48b56c4d5fc5e197c4ea` |
| `code/ridge/config_hash.py` | `be5fb2831222b63ca383f18c8f66ae32686607839c85806e8cdaf225a28aec97` |
| `code/ridge/determinism_gate.py` | `36b31cd19830d3dda89dd792256bfea6af564a1a0827c4ef5764cd1c4d3e4074` |
| `code/ridge/determinism_probe_modal.py` | `62c686c80313c9742030d1a541552dea1f44f07930cb34c1ba5f576b0f226843` |
| `code/ridge/extract_modal.py` | `7f2fbf27efdae7eb3265ff2bda391386c63c2abc75dfa27cc0be830eb01e753d` |
| `code/ridge/fit.py` | `437c4e03c115d514457fdfce34c8018029779b8be895ea024a0dcf639a2cbedf` |
| `code/ridge/folds.py` | `e10801a0fb34a6dff927854c25884d48097f8bca610dd972243063afad568711` |
| `code/ridge/gridpool.py` | `4fda24c0aa08694205f9c020dc7559785093156f50444d4160d2974ac802ffd4` |
| `code/ridge/merge.py` | `98fa98390ba2fe5d7e74adcc35e73e0a7efa20a4e48a3b507b890103ed7e0d97` |
| `code/ridge/pilot_features_modal.py` | `23c7e185291cf3e78060978336141297543b08b19234ff4131ea5b5ff9bb9980` |
| `code/ridge/pilot_fit.py` | `6e2deaf28eefcb51958718e2797cf0f924f4f7c301b76233ff0cdf5c12f59485` |
| `code/ridge/ridge_dual.py` | `3719d9a6fb0ee85f9451823140983de92fdd57061bb9477ccf26441fe55cfd44` |
| `code/ridge/suite_count.py` | `99b1a20ddddc85dd15d134f3433e93e0df601889e05558ee12c1af8d3cd5af2c` |
| `code/ridge/test_loader_hardening.py` | `b33c8f9a96f8ef73a34e30c45cc8383b2f237a6f1c24460b323c276fd39c0851` |
| `code/ridge/test_phase2_controls.py` | `6a09a59d8871786ed526d71e6fd685e567f7220a5e7da0e8cfb43e2c10e00395` |
| `code/ridge/tolerance_measurement.py` | `474776d496572b3a67e0b031907c690145d217318e13ce3eecfa626bac80ae3c` |
| `code/cross_species/models_v2.py` | `b3d62de84ef79955926f3762d13c15a9fb2b84f74da3a2ab0533127366bc19cc` |
| `code/bn_guard.py` | `e93ec323ffc34ac2158b4f4dd7fb71bc6a89ce73f9d72140248dcfccaa20781e` |

`code/cross_species/models_v2.py` was taken **with uncommitted changes** in the source
working copy (`git diff --stat` against the source HEAD):

```
 code/cross_species/models_v2.py | 147 ++++++++++++++++++++++++++++------------
 1 file changed, 104 insertions(+), 43 deletions(-)
```

## Data

No data is committed. `DATA_MANIFEST.txt` lists every file under `arm_b_out/` and
`pilot_features/` in the source repository (path relative to `RIDGE_DATA_DIR`, size in
bytes, sha256).

## Status of the deviations (PREREGISTRATION.md §9)

| Entry | Status |
|---|---|
| D1, alpha-boundary assertion (low edge, reliable voxels, hard per voxel) | ADOPTED 2026-08-20, signed off externally by the author |
| D1 remedy clause | AMENDED 2026-08-20: a low-edge firing requires a diagnosis first; the fix belongs in the selection rule; the tolerance form was not adopted |
| D1a, exception for the permuted control | REMOVED 2026-08-20; the supporting measurement was synthetic, not THINGS data |
| D2, selection criterion Pearson r -> MSE (reporting stays Pearson r) | ADOPTED 2026-08-20 |
| D3, alpha grid 1e-6 .. 1e8, 29 points | ADOPTED 2026-08-20 |
| Tolerance rule | Decided 2026-10-03: min-MSE stays; see `amendments/2026-10-03_tolerance-rule.md` |
| D1 abort behaviour | Replaced 2026-10-03 by a logged, post-run plateau gate (diagnostic grid 1e-7..1e-9, invalid if relative gain >= 1e-3); score-neutral; D1 text unchanged; see `amendments/2026-10-03_d1-plateau-gate.md` |

## Known stale passages in PREREGISTRATION.md / config.yaml

Left unedited on purpose; to be corrected at the next legitimate schema bump.

1. The header of `PREREGISTRATION.md` still names `schema_version: 1` and hash `8163d8c4a146`.
2. §7 still lists the alpha grid as 1e-2 .. 1e8 with 25 points (superseded by D3).
3. `config.yaml` marks `splithalf_corrected` as `UNVERIFIED`; `PILOT_REPORT.md` §7 shows
   it is reproduced exactly by the self-computed split-half (r = 1.000000).
4. `config.yaml` `reference_effect` still gives +0.044 at 224 px (paper v1); paper v3
   reports +0.030 [0.018, 0.042] on cross-run pairs per subject.

## What the measurement data is, and is not

tolerance_all_rungs.csv is a measurement with the D1 assertion disabled, has no config_hash column, and uses Random Weights without BN calibration; it is not a pilot result. pilot_fit.py aborted under config hash 206fa875cca5 at Random Weights (BN-calibrated) s3 g1, outer fold 3, after 28 of 30 conditions at g=1.

## Amendment dating

Date in the file name and text compared with the timestamp of the commit that added
the file (`git log --diff-filter=A --format=%cI -- <file>`).

| File | Date in file name | Date in text | Commit | Commit timestamp |
|---|---|---|---|---|
| `amendments/2026-10-03_tolerance-rule.md` | 2026-10-03 | 2026-10-03 | `5e968d4` | 2026-10-03T19:07:03+02:00 |
| `amendments/2026-10-03_d1-plateau-gate.md` | 2026-10-03 | 2026-10-03 (also cites 2026-08-20 for the original D1) | `0c7db7a` | 2026-10-03T20:33:12+02:00 |
| `amendments/2026-10-03_step4-predictions.md` | 2026-10-03 | none | `f148085` | 2026-10-08T17:29:16+02:00 |

`amendments/2026-10-03_step4-predictions.md`: the file name says 2026-10-03, the commit
was made on 2026-10-08. The commit timestamp is authoritative; the file is left unchanged so that the committed text stays byte-identical.

## History cleanup

The committer timestamp of `2fd215a` ("Add step 9 report", 2026-10-13) is the time of a history cleanup that merged a follow-up wording fix (one run-note sentence) into that commit; the substance of the report is unchanged.

## Step 25 motivation

Step 25 (`amendments/2026-10-16_step25-predictions.md`) was prompted by a critical review of an earlier draft, which the author obtained from an AI assistant. The motivation line of the amendment calls it an 'external critique'; the amendment file is left unchanged because its committed text is evidence of the order of events. Step 28 (`amendments/2026-10-18_step28-predictions.md`) was prompted by a second critical review of the revised draft, again obtained from an AI assistant.
