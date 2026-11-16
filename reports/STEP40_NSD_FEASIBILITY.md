# Step 40, part D: feasibility of a V1 RSA on the Natural Scenes Dataset (NSD)

Go/no-go report only (amendments/2026-11-16_step40-predictions.md, part D); no analysis,
no prediction. Checked on 2026-11-16 from the official NSD website, the AWS Open Data
registry entry and the public listing of the NSD S3 bucket. Nothing larger than 100 MB was
downloaded and no form was filled in. One file was read for the image counts below: the
11 MB design table `nsd_stim_info_merged.csv` from the public bucket (kept outside the
repository).

## Access

- NSD access requires the **NSD Data Access Agreement**, a short online form; after
  completing it, one receives a link to the NSD Data Manual
  ([naturalscenesdataset.org](https://naturalscenesdataset.org/); form:
  [forms.gle/eT4jHxaWwYUDEf2i9](https://forms.gle/eT4jHxaWwYUDEf2i9)).
- The form has to be filled in by the author himself; it was not filled in here.
- The data are hosted in the public S3 bucket `natural-scenes-dataset` (us-east-2),
  listable without an AWS account (`aws s3 ls --no-sign-request
  s3://natural-scenes-dataset/`) ([registry.opendata.aws/nsd](https://registry.opendata.aws/nsd)).
- Whether the agreement also governs the use of files that are technically readable from
  the bucket before it is signed, and the exact terms of use (citation, redistribution of
  derived data): **not verified** (the terms are in the Data Manual, which is sent after
  the form).

## Files needed for a V1 RSA with 8 subjects

Sizes from the bucket listing
([bucket root](https://natural-scenes-dataset.s3.us-east-2.amazonaws.com/?list-type=2&delimiter=/)),
1.8 mm functional space, single-trial betas version 3 (`betas_fithrf_GLMdenoise_RR`, the
version most papers use; the choice is not made here):

| File | Path in the bucket | Size |
|---|---|---|
| Betas, one file per session, nii.gz | `nsddata_betas/ppdata/subjXX/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_sessionNN.nii.gz` | see below |
| V1 ROI (`prf-visualrois`; label definitions in the Data Manual, not verified here) | `nsddata/ppdata/subjXX/func1pt8mm/roi/prf-visualrois.nii.gz` | 66 KB (subj01) |
| Higher visual cortex mask | `nsddata/ppdata/subjXX/func1pt8mm/roi/nsdgeneral.nii.gz` | 65 KB (subj01) |
| Design (image per trial, shared1000, repetitions) | `nsddata/experiments/nsd/nsd_expdesign.mat`, `nsd_stim_info_merged.csv` | 0.26 MB, 11.3 MB |
| Images (all 73,000) | `nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5` | 39.6 GB |

Betas per subject (count of `betas_sessionNN` files in the listing):

| Subject | Sessions | nii.gz | hdf5 |
|---|---|---|---|
| subj01 | 40 | 19.6 GB | 43.8 GB |
| subj02 | 40 | 18.2 GB | 45.7 GB |
| subj03 | 32 | 15.1 GB | 35.3 GB |
| subj04 | 30 | 13.1 GB | 31.6 GB |
| subj05 | 40 | 16.2 GB | 37.4 GB |
| subj06 | 32 | 17.3 GB | 39.9 GB |
| subj07 | 40 | 15.7 GB | 37.6 GB |
| subj08 | 30 | 13.0 GB | 30.2 GB |
| **Total** | 284 | **128.2 GB** | **301.5 GB** |

The images are only needed for the shared stimuli (below); whether `nsd_stimuli.hdf5` can
be read partially over HTTP (range requests) instead of being downloaded in full, or
whether the 515 images should be taken from COCO with the crop boxes of the design table:
**not verified**.

## Images shared by all subjects

From `nsd_stim_info_merged.csv`, counting a repetition as available when its trial falls
within the sessions present in the bucket (750 trials per session):

- `shared1000`: 1000 images planned for all subjects.
- Seen at least once by all 8 subjects: **907** images.
- Seen in all 3 repetitions by all 8 subjects: **515** images (subjects 3, 4, 6 and 8
  completed 32, 30, 32 and 30 of 40 sessions).

An analysis on images common to all subjects is therefore possible: 515 images with three
repetitions each (cross-validated distances over 3 repetitions, not 12 as in THINGS), or
907 with at least one.

## Compute

Measured in this step (step 40, CPU, this machine):

- Conv1 extraction with the existing code: 10 checkpoints (Random Weights and Backprop × 5
  seeds) plus the control band took 16 s for 20 images and 39 s for 100 images, so about
  3–4 min for 515 images and about 6 min for 907.
- RSA with a 1000-resample stimulus bootstrap: see the part C runtime in
  `reports/STEP40_REPORT.md` (100 images, 3 subjects). With 515 images the number of pairs
  grows from 4,950 to 132,355 (27×), and 8 instead of 3 subjects; estimated a few hours on
  8 cores.
- Not computed here: the V1 extraction from the volumes (reading 284 session files and
  masking), estimated at well under a minute per session once downloaded.
- Download time for 128 GB depends on the bandwidth: **not verified** (at 20 MB/s about
  1.8 h, at 5 MB/s about 7 h).

The `bench/` directory named in the task does not exist in this repository; the estimates
use the timings measured in this step and the per-fit times stored in
`results/step25/ridge_skill.csv` (median 36 s per ridge fit at g = 4).

## Disk space

`df -h C:` on 2026-11-16: 953 GB total, 107 GB free. The full nii.gz betas (128 GB) do
**not** fit; the hdf5 betas (302 GB) even less. Processing one session at a time (download
about 0.4–0.5 GB, extract V1 voxels of the shared trials, delete) needs under 1 GB at any
moment, plus a few hundred MB for the extracted V1 data and the 515 images.

## Go / no-go

**Go, conditional on the access agreement.** Reasons:
- the data are public in a listable bucket, and the files needed are identified;
- 515 images are seen three times by all 8 subjects, enough for an RSA with a stimulus
  bootstrap and a cross-validated distance (over 3 repetitions);
- feature extraction and RSA are cheap on this CPU (minutes to hours);
- the only hard constraint, disk space, is met by streaming one session at a time.

Open before starting: the terms of the agreement (**not verified**), the beta version and
voxel space, the image source (partial hdf5 read or COCO), and a new amendment with
predictions, since this would be a new analysis.

## Plan (after the agreement)

1. Fill in the NSD Data Access Agreement (author) and read the Data Manual, including the
   terms of use and the recommended beta version. (1 day, including the wait for the link.)
2. Commit an amendment with predictions and the analysis choices (beta version, V1 from
   `prf-visualrois`, 515 shared images, normalisation, RSA cells as in step 40 C). (Half a day.)
3. Download design, ROI files and the 515 images (partial hdf5 read or COCO). (1–2 h.)
4. Stream the 284 beta sessions: per session download, select the shared trials and V1
   voxels, save, delete. (2–8 h, bandwidth-bound; unattended.)
5. Extract Conv1 features for the 515 images (10 checkpoints). (Under 10 min.)
6. Compute the RSA cells with the stimulus bootstrap and the cross-validated distances,
   write results and report. (A few hours.)
