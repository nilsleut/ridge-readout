# Phase 2b — selection criterion measured; D1 still open

> **RESOLVED SINCE WRITING.** D1, D2 and D3 were adopted by the author on 2026-08-20 —
> D1 in a **stricter** form than either option discussed here. Config is now
> `schema_version: 2`, hash `206fa875cca5`. See `PREREGISTRATION.md` §9 and
> `PILOT_REPORT.md`. The header below describes the state at the time of writing.

Config: **`schema_version: 1`, hash `8163d8c4a146`** — reverted, verified against
`config.lock`. The operative code path is `select_alpha_per_voxel(mode="per_voxel")`,
which matches it. **No fitting run has produced a number.** Nothing below is adopted.

---

## 0. The sign-off failure

You are right and I will not qualify it. I called a question, treated the reply as
authorisation on a frozen degree of freedom, then wrote "Resolved — option (b) selected"
and bumped `schema_version` to 2 in the same document that first raised the problem. That
is the executing agent signing off on itself.

Reverted in full: `config.yaml` is restored to schema 1 and re-hashes to `8163d8c4a146`.
`PREREGISTRATION.md` §9 now carries a standing header marking D1 and D2 **PROPOSED, NOT
ADOPTED**, and records the false approval rather than deleting it. The code default went
back to per-voxel; the alternatives are reachable only through an explicit `mode=`
argument used by the measurement harness.

---

## 1. Task 1 — the criterion is the root cause, confirmed

`code/ridge/boundary_characterisation.py`. **Real Arm A geometry throughout:** n = 720,
outer 8 → 630 training, inner 5 → 504 per inner fit. 600 voxels per population, α grid
1e−2 … 1e8, 25 points. Reliability computed exactly as Arm B will compute it — two
independent half-averages, Spearman–Brown — threshold 0.2.

Fractions are **never summed across sides**.

| Population (reliable/600) | criterion | **lo·rel** | hi·rel | lo·unrel | hi·unrel | median α |
|---|---|---|---|---|---|---|
| pure noise (0) | Pearson r | — | — | 0.437 | 0.467 | 1.0e3 |
| pure noise (0) | **MSE** | — | — | **0.000** | 0.527 | 1.0e8 |
| weak, below threshold (3) | Pearson r | 0.208 | 0.625 | 0.399 | 0.473 | 2.6e3 |
| weak, below threshold (3) | **MSE** | **0.000** | 0.417 | **0.000** | 0.283 | 2.6e3 |
| mixed, spans threshold (50) | Pearson r | 0.212 | 0.062 | 0.344 | 0.311 | 3.8e2 |
| mixed, spans threshold (50) | **MSE** | **0.000** | **0.000** | **0.000** | 0.125 | 3.8e2 |
| moderate, mostly reliable (395) | Pearson r | 0.267 | 0.005 | 0.245 | 0.105 | 1.5e2 |
| moderate, mostly reliable (395) | **MSE** | **0.000** | **0.000** | **0.000** | **0.000** | 5.6e1 |

**The prediction holds, and more sharply than stated.**

- Under **MSE the low-side rate among reliable voxels is exactly 0.000 in every
  population** — including pure noise, where it is 0.000 among unreliable voxels too.
- Under **Pearson r it is 0.21–0.27 in every population that has reliable voxels at all**,
  including the one with 395 of them and abundant signal. r-selection manufactures a ~25%
  low-boundary alarm rate that is independent of whether the grid is adequate.
- Under MSE, the boundary selections that do occur sit **entirely on the high side and
  entirely in unreliable voxels**: 0.527 (pure noise) → 0.283 (weak) → 0.125 (mixed) →
  0.000 (moderate). That is the estimator working. A no-signal voxel selects α_max, and
  the median selected α for pure noise is 1.0e8 — the top of the grid, the correct answer,
  arrived at stably.

The old summed rates (91.1% / 56.8%) reproduce under r (0.437 + 0.467 = 0.904 for pure
noise) and collapse under MSE.

**Why:** ridge's α controls the *scale* of the prediction far more than its direction, and
Pearson r is scale-invariant. Measured directly in Task 2: under r the per-voxel "optimum"
spans **1e−6 to 1e12** — the entire 18-decade probe grid. There is no optimum for `argmax`
to find, so it finds an endpoint.

**Not adopted.** Prepared, measured, reported.

---

## 2. Task 2 — a threshold derived from the signal case

Grid-truncation experiment: 600 voxels, SNR 0.10–0.60, optimum located on a wide probe
grid (1e−6 … 1e12, 61 points), then three grids built relative to it.

| criterion | grid | **lo·rel** | hi·rel | lo·unrel | hi·unrel | any·rel | any·all |
|---|---|---|---|---|---|---|---|
| Pearson r | generous | 0.133 | 0.002 | 0.152 | 0.104 | 0.134 | 0.195 |
| Pearson r | just contains optimum | 0.127 | 0.001 | 0.147 | 0.107 | 0.128 | 0.191 |
| Pearson r | **truncated past optimum** | 0.000 | 0.278 | 0.000 | 0.287 | 0.278 | 0.283 |
| **MSE** | generous | **0.000** | 0.000 | 0.000 | 0.020 | 0.000 | 0.010 |
| **MSE** | just contains optimum | **0.000** | 0.007 | 0.000 | 0.463 | 0.007 | 0.236 |
| **MSE** | **truncated past optimum** | **1.000** | 0.000 | 0.966 | 0.019 | 1.000 | 0.993 |

**Under Pearson r, no threshold exists.** The true optimum spans the whole probe grid
(1e−6 … 1e12), so "a grid that just contains the optimum" is not constructible, and
generous (0.134), just-contains (0.128) and truncated (0.278) are not separable by any
cut. That is stronger than "the threshold was calibrated wrongly": under r-selection the
quantity a threshold would have to detect does not exist.

**Under MSE the separation is total.** The low-side fraction among reliable voxels is
**0.000** for both adequate grids and **1.000** for the truncated one. The optimum span is
1e−6 … 5.0e2, i.e. actually localised.

**Threshold candidate: `lo·rel > 0.05`**, on MSE selection. The margin is 0.000 vs 1.000,
so anything in (0.01, 0.9) behaves identically; 0.05 is visibly clear of zero without
approaching the failure case. **Derived from the signal case, as required** — it is the
rate a *truncated* grid produces, not the rate noise produces.

Two things the table settles about the withdrawn 0.95 candidate:

- It was calibrated to *not fire* (above the 0.911 null rate), which made it permissive by
  construction. Confirmed: an adequate MSE grid yields any·all = 0.236 and a truncated one
  0.993, so 0.95 sits inside the failure region rather than between the two.
- **Summing the sides destroys the instrument.** At "just contains the optimum" under MSE,
  hi·unrel = 0.463 and any·all = 0.236 — driven entirely by unreliable voxels correctly
  choosing maximal regularisation. The same grid gives lo·rel = 0.000. The side-split and
  the reliability restriction are both load-bearing.

**Not adopted.** The criterion change and any surviving assertion need your sign-off,
together.

---

## 3. Task 3 — the two reconciliations

### 3.1 — you are right; the verification ran on the wrong geometry

`df = 209` was read correctly as n_train − 1, but n_train was 210 because that synthetic
used n = 240, not the real n = 720. Re-run at the real Arm A geometry:

| | n_train | df_max | expected | d/n_train |
|---|---|---|---|---|
| outer fold | **630** | 629.0 | 629 | **3.25** |
| inner fold | **504** | 503.0 | 503 | **4.06** |
| *(previous claim)* | *210* | *209* | *209* | *9.75* |

So the g = 8 rung runs at d/n ≈ 3.3–4.1, not 9.8. The dual form is confirmed correct at
the real shape — df caps exactly at n_train − 1, the −1 coming from per-feature centring.
The sentence "verified on synthetic data at the real shapes" was false for that row and is
withdrawn; it is now true.

### 3.2 — 78 vs 83: both were accurate, at different moments, which is the problem

`PHASE2_REPORT.md` §2 was written when the suite had 78 checks. The summary was written
after the (now-reverted) rescope tests were added, at 83. Neither was wrong when written,
and the document was internally inconsistent by the time it was sent — the same drift the
config-hash discipline exists to prevent in data, occurring in prose.


<!--SUITE-->**138 checks, all passing** (phase2 controls 97, loader hardening 41; determinism-gate self-test pass) against config `206fa875cca5`<!--/SUITE-->

The suite now covers per-voxel mode as operative, the side-split statistics, `mode="none"`, and the
un-adopted population-fraction mode as a proposal.

---

## 4. Task 4 — P1 scoped down

The "gross weight-space divergence, near-identical representational geometry" observation
is marked in `PRECONDITIONS_P1_P4.md` as a **methods footnote at most**: it rests on one
re-run pair per rule, concerns only nondeterminism-induced divergence, and is a by-product
of a reproducibility check rather than a claim about weight space and function in general.
It does not enter the paper as a finding. No further compute; **the FA-under-Adam run is
out of scope and will not be run.**

---

## 5. Built in parallel, not run

`code/ridge/arm_b_betas.py` — Arm B trial-averaged betas and split-half reliability.
Neither depends on D1 or on the criterion question.

- ROI masking and per-voxel z-scoring across all 9,840 trials **before** stimulus
  selection, matching `extract_fmri_rdms_720.py:55-78`, so the arms differ only in
  stimulus set and averaging.
- All **462** balanced splits enumerated, not sampled — verified; deterministic, no seed.
- Spearman–Brown correction on the 6-rep half-averages.
- Hard failures if a subject does not yield exactly 100 test stimuli, or exactly 12
  repetitions each.

Not executed: it touches the 47 GB H5 files and produces no fitted number.

---

## 6. Open, awaiting your decision

**D1** (α-boundary assertion scope) and **D2** (inner-CV selection criterion: Pearson r →
MSE, with Pearson r retained as the *reported* metric for Brain-Score comparability) are
decided together, by you.

My reading of the measurement, offered as a reading and not as a decision: D2 is the
substantive change, and D1 largely dissolves once D2 is made — under MSE selection the
low-side reliable rate is 0.000 on every adequate grid, so even the original per-voxel
rule would pass on that side. What the per-voxel rule cannot survive is the *high* side,
where α_max is the correct answer for a no-signal voxel and not an alarm at all.

No fitting run produces a number before you decide.
