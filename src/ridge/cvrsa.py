"""
cvrsa.py
========
RDMs and comparisons for step 40, part C (amendments/2026-11-16_step40-predictions.md):
RSA on the 100 repeated THINGS-fMRI test images.

Brain RDMs from reps (n_stim, n_sessions, n_voxel), one presentation per session:
  b1  correlation distance on the presentation-averaged patterns
  b2  Euclidean distance on the presentation-averaged patterns
  b3  cross-validated squared Euclidean distance
        d_ij = mean over ordered pairs of distinct sessions (A, B) of
               (x_i^A - x_j^A) . (x_i^B - x_j^B)
      computed with sum_{A != B} D^A . D^B = |sum_s D^s|^2 - sum_s |D^s|^2.
      Noise that is independent across sessions has expectation 0, so d_ij is unbiased
      for the squared distance of the true patterns and can be negative.

Model RDMs from (n_stim, n_units) features, raw or standardised over the stimuli;
correlation or Euclidean distance. Comparison: Spearman over the condensed pairs.
Stimulus bootstrap: resample the stimuli with replacement, keep the pairs of resampled
positions whose stimuli differ (self-pairs dropped), same draws for every comparison.
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

__all__ = ["cv_sq_euclid", "brain_rdms", "model_rdm", "spearman", "boot_draws",
           "condensed_index", "boot_spearman"]


def cv_sq_euclid(reps: np.ndarray) -> np.ndarray:
    """Condensed cross-validated squared Euclidean distances (scipy pdist order)."""
    reps = np.asarray(reps, np.float64)
    n, S, _ = reps.shape
    if S < 2:
        raise ValueError("need at least two sessions")
    total = reps.sum(axis=1)                                   # (n, v)
    G = total @ total.T                                        # |sum_s D^s|^2 via Gram
    g = np.diag(G)
    sumsq = g[:, None] + g[None, :] - 2 * G
    within = np.zeros((n, n))
    for s in range(S):
        X = reps[:, s, :]
        Gs = X @ X.T
        gs = np.diag(Gs)
        within += gs[:, None] + gs[None, :] - 2 * Gs
    D = (sumsq - within) / (S * (S - 1))
    iu = np.triu_indices(n, 1)
    return D[iu]


def brain_rdms(reps: np.ndarray) -> dict:
    avg = np.asarray(reps, np.float64).mean(axis=1)
    return {"b1": pdist(avg, "correlation"), "b2": pdist(avg, "euclidean"), "b3": cv_sq_euclid(reps)}


def model_rdm(X: np.ndarray, normalisation: str, distance: str) -> np.ndarray:
    X = np.asarray(X, np.float64)
    if normalisation == "z":
        sd = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(sd < 1e-8, 1.0, sd)
        X[:, sd < 1e-8] = 0.0
    elif normalisation != "raw":
        raise ValueError(normalisation)
    out = pdist(X, distance)
    if not np.isfinite(out).all():
        raise ValueError("non-finite model RDM")
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def condensed_index(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """Index into a scipy condensed vector for pairs i < j."""
    i, j = np.minimum(i, j), np.maximum(i, j)
    return n * i - i * (i + 1) // 2 + (j - i - 1)


def boot_draws(n: int, n_boot: int, seed: int):
    """Per resample: condensed indices of the pairs of distinct resampled stimuli."""
    rng = np.random.default_rng(seed)
    a, b = np.triu_indices(n, 1)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        si, sj = idx[a], idx[b]
        keep = si != sj
        out.append(condensed_index(si[keep], sj[keep], n))
    return out


def boot_spearman(model: np.ndarray, brain: np.ndarray, draws) -> np.ndarray:
    """Spearman for the identity draw (row 0) and every bootstrap draw."""
    vals = [spearman(model, brain)]
    vals += [spearman(model[k], brain[k]) for k in draws]
    return np.array(vals)
