# Amendment 2026-10-03 — alpha-selection tolerance rule

## Decision

Alpha selection stays min-MSE (D2); tol-0.1 % and other tolerance forms are not adopted.

## Reason

min-MSE is the simplest rule, it is already implemented, and it adds no degree of
freedom. A tolerance rule would add a new frozen parameter (the tolerance value) that
`TOLERANCE_REPORT.md` measured at only three settings.

## Record

- Decided by Nils on 2026-10-03, after `reports/TOLERANCE_REPORT.md` and
  `results/tolerance_all_rungs.csv` were known.
- `config.yaml` is unchanged; the config hash stays `206fa875cca5`.

## Consequence

The D1 assertion stays in force. `scripts/pilot_fit.py` still aborts at
Random Weights (BN-calibrated) s3 g1 until the diagnosis required by the amended D1
remedy clause (`PREREGISTRATION.md` §9) has been made. That diagnosis is not part of
this amendment.
