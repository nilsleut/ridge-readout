"""
check_luminance_defs.py
=======================
Read-only check of the two luminance definitions in this repository, on the 720
evaluation stimuli at 224 px:

  Rec. 709  Y = 0.2126 R + 0.7152 G + 0.0722 B   control band (scripts/pilot_features_modal.py:187)
  Rec. 601  Y = 0.299 R + 0.587 G + 0.114 B      stimulus statistics of steps 7-9
                                                 (scripts/step7_gain.py:110-125)

Both are image means of Y on the [0, 1] tensor after Resize(224) -> CenterCrop(224) ->
ToTensor, without Normalize.

Sources, nothing re-implemented:
  - Rec. 709: the Rec. 709 expression is inline in the Modal function
    `extract_pilot` and cannot be imported. Its exact output is stored instead: channel 0
    of the g = 1 luminance + contrast control features (`control_lumcontrast_g1.npy`,
    channels [luminance, contrast], pool_grid at g = 1 = image mean), read with
    `pilot_fit.load_features`. These are the values that entered the control band.
  - Rec. 601: `step7_gain.stimulus_stats()[224][0]`, imported unchanged.
Both use the stimulus order of stim_order_sub-01.txt.

Output: results/checks/luminance_defs.csv
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src" / "ridge"))
sys.path.insert(0, str(REPO / "src" / "vendor"))

from config_hash import config_hash      # noqa: E402
from pilot_fit import load_features      # noqa: E402
from step7_gain import stimulus_stats    # noqa: E402

OUT = REPO / "results" / "checks"


def main():
    lc = load_features("control_lumcontrast", 1)
    if lc.shape != (720, 2):
        raise SystemExit(f"STOP: control_lumcontrast_g1 has shape {lc.shape}, expected (720, 2)")
    y709 = lc[:, 0].astype(np.float64)
    y601 = np.asarray(stimulus_stats()[224][0], dtype=np.float64)
    if y601.shape != (720,):
        raise SystemExit(f"STOP: stimulus_stats()[224] mean has shape {y601.shape}, expected (720,)")
    pr, sr = pearsonr(y709, y601)[0], spearmanr(y709, y601)[0]
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    rows = [dict(quantity="pearson", value=float(pr)), dict(quantity="spearman", value=float(sr)),
            dict(quantity="mean_rec709", value=float(y709.mean())),
            dict(quantity="mean_rec601", value=float(y601.mean())),
            dict(quantity="max_abs_diff", value=float(np.abs(y709 - y601).max()))]
    df = pd.DataFrame(rows)
    df["n_stimuli"] = 720
    df["rec709_source"] = "pilot_features/control_lumcontrast_g1.npy channel 0 (scripts/pilot_features_modal.py:187)"
    df["rec601_source"] = "scripts/step7_gain.py stimulus_stats()[224][0]"
    df["config_hash"] = config_hash()
    df["git_head"] = head
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "luminance_defs.csv", index=False)
    print(df[["quantity", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
