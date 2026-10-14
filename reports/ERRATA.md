# Errata

Corrections to reports that are left unchanged as the record of what was written at the
time. The paper follows the data.

## 2026-10-14 — `reports/PREREG_PRIMARY.md`, control band

`reports/PREREG_PRIMARY.md` (section "Instruments (as implemented)") states that the
luminance and contrast control bank predicts V1 "as well as or better than every network
except Backprop". According to the results, it lies above every network condition,
including Backprop, at every g.

Median r over reliable V1 voxels, mean over the three subjects (luminance and contrast
bank from `results/prereg/instruments_control_band.csv`; network conditions from
`results/prereg/primary.csv`, mean over seeds):

| g | luminance and contrast | Backprop (highest network condition) |
|---|---|---|
| 1 | 0.0810 | 0.0642 |
| 2 | 0.0874 | 0.0699 |
| 4 | 0.1055 | 0.0860 |
| 8 | 0.1105 | 0.0877 |

The report is left unchanged; the paper follows the data.
