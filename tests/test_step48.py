"""
test_step48.py
==============
Checks for step 48 (amendments/2026-11-22_step48-predictions.md, config_step48.yaml).
All must pass before the step 48 runs.

  (i)   T-global reproduces the paper cell (+0.030446, = step 47 to 1e-12); N-session from the
        stored raw betas and session statistics reproduces step 42 exactly (subj01 session 1, the
        session on disk before the reload; the run repeats this for every session and compares all
        rho values with step 42)
  (ii)  synthetic patterns: common profile c * p plus a stimulus part with per-trial amplitude b_i;
        Spearman(correlation-distance RDM, amplitude RDM b_i + b_j) rises with c (near 0 at c = 0)
        and centring the patterns over trials removes it; the same with the gain RDM 2 - g_i - g_j
  (iii) image statistics on synthetic images: CE 0 for a uniform image, CE of a grating scales with
        contrast^2, C of a grating = amplitude / sqrt(2); OFF, ON, DD for areas of known brightness;
        with the image's own mean OFF = ON; L and C of image_stats equal step44 lum_contrast
        (first 10 THINGS-720 images)
  (iv)  g and g^M on synthetic data equal the hand-computed cosines with the mean row

Run: python tests/test_step48.py [--only i,ii,iii,iv]
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nsd_common as C                                       # noqa: E402,F401
import image_stats as IS                                     # noqa: E402
import step48_amplitude as AM                                # noqa: E402
import step48_conventions as S48                             # noqa: E402

FAILS, N = [], 0


def check(name, ok, info=""):
    global N
    N += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def test_reproduce():
    import pandas as pd
    import step44_why as W
    from scipy.spatial.distance import pdist
    import nsd_raw_fetch as RF
    br = []
    for s in S48.SUBJ:
        t = S48.things_data(s)
        br.append(pdist(t["p720_T-global"][:, S48.local_rois(s)["V1"]], "correlation"))
    d, _ = W.delta_on(W.things720_models("V1"), br, W.crossrun_index())
    ref = pd.read_csv(REPO / "results" / "step47" / "run_centring.csv")
    r47 = float(ref[(ref.kind == "delta") & (ref.roi == "V1") & (ref.variant == "raw")].delta.iloc[0])
    check("(i) T-global = paper cell", abs(d - r47) <= 1e-12 and abs(d - 0.030446) < 5e-7, f"{d:+.9f} vs {r47:+.9f}")
    z = np.load(RF.session_file("subj01", 1))
    ref42 = np.load(C.nsd_dir("betas", "subj01") / "session01.npz")
    Z = RF.zscore(z["Q"].astype(np.float64) / C.cfg()["betas"]["scale_divisor"], z["mu"], z["sd"], z["valid"]).astype(np.float32)
    check("(i) N-session from raw + session statistics = step 42 (subj01 session 1)",
          np.array_equal(Z, ref42["Z"]) and np.array_equal(z["valid"], ref42["valid"]), f"Z {Z.shape}")
    M, S = RF.pooled_stats([np.array([1.0, 2.0]), np.array([3.0, 0.0])], [np.array([1.0, 0.5]), np.array([2.0, 0.5])])
    x = np.r_[np.full(50, 0.0), np.full(50, 2.0), np.full(50, 1.0), np.full(50, 5.0)]   # session 1: mean 1, sd 1; 2: mean 3, sd 2
    check("(i) pooled mean / SD from session statistics", abs(M[0] - x.mean()) < 1e-12 and abs(S[0] - x.std()) < 1e-12,
          f"{M[0]:.3f} {S[0]:.3f}")


def test_profile():
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    n, v = 200, 300
    p = rng.normal(1.0, 1.0, v)
    b = rng.uniform(0.5, 2.0, n)
    U = rng.normal(size=(n, v))
    i, j = np.triu_indices(n, 1)
    A = b[i] + b[j]
    rho, rho_g, rho_c = [], [], []
    for c in (0.0, 0.5, 1.0, 2.0, 4.0):
        X = c * p[None, :] + b[:, None] * U
        D = pdist(X, "correlation")
        g = AM.gain(X)
        rho.append(spearmanr(D, A)[0])
        rho_g.append(spearmanr(D, 2 - g[i] - g[j])[0])
        rho_c.append(spearmanr(pdist(X - X.mean(axis=0), "correlation"), A)[0])
    check("(ii) rho(corr-distance RDM, amplitude RDM) rises with the profile", bool(np.all(np.diff(rho) > 0)) and abs(rho[0]) < 0.15
          and rho[-1] > 0.9, " ".join(f"{x:+.3f}" for x in rho))
    check("(ii) the same with the gain RDM 2 - g_i - g_j", bool(np.all(np.diff(rho_g[1:]) > 0)) and rho_g[-1] > 0.9,
          " ".join(f"{x:+.3f}" for x in rho_g))
    check("(ii) centring over trials removes it", max(abs(x) for x in rho_c) < 0.15, " ".join(f"{x:+.3f}" for x in rho_c))


def test_image_stats():
    px = 200
    yy, xx = np.mgrid[0:px, 0:px].astype(np.float64)
    uni = np.full((px, px), 0.4)
    check("(iii) CE of a uniform image is 0", IS.contrast_energy(uni) < 1e-20, f"{IS.contrast_energy(uni):.1e}")
    gr = {c: 0.5 + c * np.sin(2 * np.pi * 0.1 * xx) for c in (0.1, 0.2, 0.4)}
    ce = {c: IS.contrast_energy(g) for c, g in gr.items()}
    r1, r2 = ce[0.2] / ce[0.1], ce[0.4] / ce[0.2]
    check("(iii) CE of a grating scales with contrast^2", abs(r1 - 4) < 1e-9 and abs(r2 - 4) < 1e-9, f"ratios {r1:.6f} {r2:.6f}")
    sd = float(gr[0.2].astype(np.float32).std())
    check("(iii) C of a grating = amplitude / sqrt(2)", abs(sd - 0.2 / np.sqrt(2)) < 1e-6, f"{sd:.6f}")
    half = np.where(xx < px / 2, 0.2, 0.8)
    q = np.where((xx < px / 2) & (yy < px / 2), 0.9, 0.4)
    o1, o2, o3 = IS.off_on(half, 0.5), IS.off_on(np.full((px, px), 0.3), 0.5), IS.off_on(q, 0.5)
    ok = (np.allclose(o1, (0.15, 0.15, 0.5)) and np.allclose(o2, (0.2, 0.0, 1.0))
          and np.allclose(o3, (0.075, 0.1, 0.075 / 0.175)))
    check("(iii) OFF, ON, DD for areas of known brightness", ok, f"{np.round(o1, 4)} {np.round(o2, 4)} {np.round(o3, 4)}")
    own = IS.off_on(q, float(q.mean()))
    check("(iii) with the image's own mean OFF = ON", abs(own[0] - own[1]) < 1e-12, f"{own[0]:.4f} {own[1]:.4f}")
    st = IS.image_stats([half, q, uni])
    check("(iii) reference = mean L of the set", abs(st["ref"] - np.mean([0.5, 0.525, 0.4])) < 1e-6, f"{st['ref']:.4f}")
    import step44_why as W
    paths = W.things720_images()[:10]
    s = IS.image_stats([IS.grey(p) for p in paths])
    L, Cn = W.lum_contrast(paths)
    check("(iii) L and C equal step 44 lum_contrast", np.array_equal(s["L"], L) and np.array_equal(s["C"], Cn),
          f"max |d| {np.abs(s['L'] - L).max():.1e} {np.abs(s['C'] - Cn).max():.1e}")


def test_gain():
    rng = np.random.default_rng(5)
    X = rng.normal(0.5, 1.0, size=(40, 30))
    m = X.mean(axis=0)
    hand = np.array([float(x @ m / np.sqrt(x @ x) / np.sqrt(m @ m)) for x in X])
    check("(iv) g = cosine with the mean pattern", np.abs(AM.gain(X) - hand).max() < 1e-14, f"max |d| {np.abs(AM.gain(X) - hand).max():.1e}")
    u = np.zeros(30)
    u[0] = 1.0
    W_ = np.zeros((4, 30))
    W_[:, 1:] = rng.normal(size=(4, 29))
    a = np.array([1.0, 2.0, 3.0, 4.0])
    W_ -= W_.mean(axis=0)
    F = a[:, None] * u[None, :] + W_                          # mean row = 2.5 u
    exp = a / np.linalg.norm(F, axis=1)
    check("(iv) g^M on features with a known mean axis", np.abs(AM.gain(F) - exp).max() < 1e-14, " ".join(f"{x:.3f}" for x in AM.gain(F)))


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    for key, fn in (("i", test_reproduce), ("ii", test_profile), ("iii", test_image_stats), ("iv", test_gain)):
        if only is None or key in only:
            fn()
    print(f"  all {N} checks passed" if not FAILS else f"  {len(FAILS)} of {N} checks FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
