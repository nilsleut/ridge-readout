# Vendored from evaluation-resolution-rsa: code/bandpass/bandpass_extract.py
# Source HEAD: 7a9b3742cf4d739400bd52d0d6eb8a53c1bd517c
# sha256 of the original: b35e679965ff4bac1fc942debbb5aedd318ee5bb5aa4cc873cbefac32788ded6
# Taken verbatim: constants PX, CUTS, BANDS, GREY and the functions sigma, bands, condition.
# Not taken: main(), rdm_path, the output paths. No local changes.
"""Octave bands by difference of Gaussians (sigma convention of the blur test): a Gaussian
low-pass L_r has amplitude response 0.5 at r/2 cycles per image, sigma(r) =
sqrt(ln 2 / (2 pi^2)) * 2 * 224 / r ~= 84 / r px.

  band "0-14" = L14   coarse residual, contains the image mean (DC)
  only   the band alone (bands without DC shifted by the CIFAR-10 channel-mean grey)
  notch  native minus the band; for 0-14 shifted by the same constant grey.
Filter on the [0,1] image after Resize(224) -> CenterCrop(224) -> ToTensor, then Normalize.
"""
import math

import numpy as np
from scipy.ndimage import gaussian_filter

PX = 224
CUTS = [112, 56, 28, 14]
BANDS = ["112-224", "56-112", "28-56", "14-28", "0-14"]
GREY = np.array([0.4914, 0.4822, 0.4465], np.float32)[:, None, None]


def sigma(r):
    return math.sqrt(math.log(2) / (2 * math.pi ** 2)) * 2 * PX / r


def bands(x):
    """x: (3, H, W) float32 in [0,1] -> dict band -> array (sums back to x exactly)."""
    L = {r: gaussian_filter(x, sigma=(0, sigma(r), sigma(r)), mode="reflect", truncate=4.0) for r in CUTS}
    b = {"112-224": x - L[112], "56-112": L[112] - L[56], "28-56": L[56] - L[28],
         "14-28": L[28] - L[14], "0-14": L[14]}
    return b


def condition(x, b, cond):
    kind, band = cond.split("_", 1)
    if kind == "only":
        return b[band] + (0 if band == "0-14" else GREY)
    return x - b[band] + (GREY if band == "0-14" else 0)
