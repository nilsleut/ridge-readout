# Step 42, part 0: NSD setup

Read on 2026-11-17 from the NSD Data Manual
([main page](https://cvnlab.slite.page/p/CT9Fwl4_hc/NSD-Data-Manual)) and the public bucket
`s3://natural-scenes-dataset` (us-east-2, listed without credentials through boto3 with
unsigned requests; the AWS CLI is not installed). Manual subpages used:
[Functional data (NSD)](https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD),
[ROIs](https://cvnlab.slite.page/p/X_7BBMgghj/ROIs),
[Experiments](https://cvnlab.slite.page/p/NKalgWd__F/Experiments),
[Informational files](https://cvnlab.slite.page/p/TL2ltDZ75o/Informational-files),
[Technical notes](https://cvnlab.slite.page/p/h_T_2Djeid/Technical-notes),
[FAQ](https://cvnlab.slite.page/p/QtQPzl1xnH/FAQ),
[Terms and Conditions](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions).

NSD files are stored only under `$NSD_DIR` (outside the repository). Nothing from NSD at
voxel or RDM level enters the repository.

## Single-trial betas

| Item | Finding | Source |
|---|---|---|
| Path | `nsddata_betas/ppdata/subjAA/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_sessionBB.nii.gz` (also `.hdf5`); confirmed in the bucket (subj01: 40 sessions, 0.49 GB per nii.gz, 1.09 GB per hdf5) | Functional data (NSD); bucket |
| Content | 750 single-trial betas per session in chronological order (12 runs concatenated) | Functional data (NSD) |
| Format | int16, multiplied by 300; cast to float and divide by 300 to obtain percent signal change | Functional data (NSD), FAQ "Common pitfalls" |
| Invalid voxels | all zeros for the whole session (also outside the brain mask) | Technical notes, "On the issue of valid voxels" |
| Normalisation | the manual's noise-ceiling metrics "assume that voxel-wise beta weights are z-scored within each session and then aggregated across sessions"; z-score per voxel within session adopted | FAQ "Common pitfalls" |
| Volume size (func1pt8mm) | subj01 81 × 104 × 83 | Technical notes, "Data sizes"; ROI file header |
| Beta version | b3 = `betas_fithrf_GLMdenoise_RR` (HRF fitted per voxel, GLMdenoise, ridge regression) | Functional data (NSD) |

## Experimental design

`nsddata/experiments/nsd/nsd_expdesign.mat` (0.26 MB; Experiments page), 1-based indices:
- `masterordering` 1 × 30000: per trial, index into the subject's 10k images;
- `subjectim` 8 × 10000: 73k IDs; the first 1000 are the shared images (sorted);
- `sharedix` 1 × 1000: 73k IDs of the shared images;
- `stimpattern` 40 sessions × 12 runs × 75 slots: 0/1 stimulus trials, the same for all subjects.

Mapping trial → image, session, run: trial t (0-based, chronological) shows 73k ID
`subjectim[s, masterordering[t]]`; session = t // 750; run = position of t mod 750 in the
cumulative stimulus count of `stimpattern[session]` (runs alternate 63 and 62 stimulus trials).

## The 515 images

From `nsd_expdesign.mat`, with the completed sessions per subject [40 40 32 30 40 32 40 30]
(Technical notes, "Final numbers"): **515** shared images are shown all three times to every
subject. The manual states the same number (Technical notes; Experiments, `special100`).
For every subject, the 1545 trials of these images fall in sessions 1–30 (30 sessions per
subject).

## ROIs (func1pt8mm)

Files: `nsddata/ppdata/subjAA/func1pt8mm/roi/<name>.nii.gz`, label tables in
`nsddata/freesurfer/subjAA/label/<name>.mgz.ctab` (ROIs page; bucket).

- **V1**: `prf-visualrois.nii.gz`, labels 1 (V1v) and 2 (V1d). subj01: 1350 voxels.
- **Higher ROI**: the manual lists no object-selective ROI file. `floc-*` ROI files exist only
  for faces, words, places and bodies; there is no `floc-objects`. Candidates provided:

| Candidate | File, labels | subj01 voxels | Note |
|---|---|---|---|
| streams "lateral" | `streams.nii.gz`, label 6 | 7799 | higher-level lateral stream, anatomical (fsaverage folding plus b3 noise ceiling); from the end of LO1/LO2 to the STS; not object-selective by definition |
| streams "midlateral" | `streams.nii.gz`, label 3 | 834 | intermediate lateral ROI, borders taken from LO1/LO2 of the Wang atlas |
| Kastner2015 LO1 + LO2 | `Kastner2015.nii.gz`, labels 14, 15 | 536 | retinotopic maps, not object-selective |
| streams "lateral" ∩ objects t > 0 | label 6 and `func1pt8mm/floc_objectstval.nii.gz` > 0 | 2921 | own construction; t > 0 matches the manual's liberal threshold for the floc ROIs |

Voxel counts are taken from the ROI files only, before the grid check against the betas and
before the exclusion of invalid voxels.

**Choice (fixed by the author on 2026-11-17, before any RSA):**
- **LO-obj** (secondary ROI, used for P3): `streams` label 6 ("lateral") ∩
  `floc_objectstval` > 0, per subject in func1pt8mm. Threshold t > 0 fixed, following the
  manual's convention for the floc ROIs; no other threshold is computed.
- **lateral** (sensitivity variant, no prediction): `streams` label 6 alone.

Reason: the manual provides no object-selective floc ROI; LO-obj is the only candidate that
is both lateral occipitotemporal and object-selective.

Contrast underlying `floc_objectstval` ([Functional data (pRF, fLoc)](https://cvnlab.slite.page/p/G5dUBGBxMo/Functional-data-pRF-fLoc)):
domain contrasts compare "a given domain compared to all other stimuli"; the objects domain
consists of the categories car and instrument (`nsddata/experiments/floc/domains.tsv`,
`categories.tsv`), so the contrast is objects (car, instrument) against the other four
domains (characters: word, number; bodies: body, limb; faces: adult, child; places:
corridor, house); t from a two-sample t-test. File:
`nsddata/ppdata/subjAA/func1pt8mm/floc_objectstval.nii.gz`.

## Images

- `nsddata/stimuli/nsd/shared1000/sharedAAAA_nsdBBBBB.png`: 1000 PNG files, RGB uint8,
  425 × 425 px (checked on one file), 322 MB in total; BBBBB is the 73k ID (1-based).
  Only the 515 needed are downloaded, one file each; no partial read of
  `nsd_stimuli.hdf5` (39.6 GB) is needed.
- Shown on a grey background (127, 127, 127) (Experiments page); the PNG files are the
  square-cropped images.
- Transform as for THINGS (`scripts/pilot_features_modal.py`): `Resize(224)`,
  `CenterCrop(224)`, `ToTensor`, `Normalize`; on the square 425 px images this is a resize to
  224 px.

## Sessions and download size (sessions 1–30, nii.gz)

| Subject | Sessions needed | Size |
|---|---|---|
| subj01 | 30 | 14.69 GB |
| subj02 | 30 | 13.64 GB |
| subj03 | 30 | 14.18 GB |
| subj04 | 30 | 13.12 GB |
| subj05 | 30 | 12.11 GB |
| subj06 | 30 | 16.25 GB |
| subj07 | 30 | 11.78 GB |
| subj08 | 30 | 12.97 GB |
| total | 240 | 108.7 GB |

Processed one session at a time (about 0.5 GB on disk at any moment); stored per subject:
1545 trials × ROI voxels as float32 (a few tens of MB).

## Terms (Terms and Conditions page)

Research and education only; no distribution of the dataset or its components; cite
Allen et al., Nature Neuroscience (2021); acknowledge "Collection of the NSD dataset was
supported by NSF IIS-1822683 and NSF IIS-1822929."
