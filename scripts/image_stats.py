"""
image_stats.py
==============
Step 48 (amendments/2026-11-22_step48-predictions.md, config_step48.yaml): per-image statistics
on the step 44 image transform (colour_statistic_control.load_rgb: Resize 224 + CenterCrop 224,
RGB in [0, 1]; grey = Rec. 709 weights).

  L    mean grey value (= step44_why.lum_contrast L, MEANLUM)
  C    SD of the grey values (= step44_why.lum_contrast C, RMS contrast)
  CE   contrast energy: mean over pixels and filters of the squared responses of a Gabor bank of
       4 orientations x 4 spatial frequencies (the formula of the existing control bank,
       pilot_features_modal.gabor_bank: cosine phase, sigma = 1 / (3 sf), orientations 0/45/90/135;
       spatial frequencies 0.05, 0.10, 0.20, 0.40 cycles / px = the existing three plus one octave
       below; kernels 41 x 41, mean subtracted so that a uniform image gives 0; reflect padding)
  OFF  mean over pixels of max(0, ref - x);  ON  mean over pixels of max(0, x - ref)
  DD   OFF / (OFF + ON)
  ref  = mean grey value over all images of the dataset (with the image's own mean, OFF = ON
       for every image and DD = 0.5; see reports/STEP48_SETUP.md)

Per-image values are kept in memory or under RIDGE_DATA_DIR / NSD_DIR only; the repository gets
aggregates.
"""
import numpy as np
from scipy.signal import fftconvolve

REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
SFS = (0.05, 0.10, 0.20, 0.40)
THETAS = (0.0, 45.0, 90.0, 135.0)
KSIZE = 41
STATS = ["L", "C", "CE", "OFF", "ON", "DD"]


def gabor_bank(sfs=SFS, thetas=THETAS, ks=KSIZE):
    """(n_filters, ks, ks) float64; the existing control-bank formula, mean subtracted."""
    yy, xx = np.meshgrid(np.arange(ks) - ks // 2, np.arange(ks) - ks // 2, indexing="ij")
    xx, yy = xx.astype(np.float64), yy.astype(np.float64)
    out = []
    for sf in sfs:
        for th in thetas:
            t = np.deg2rad(th)
            xr = xx * np.cos(t) + yy * np.sin(t)
            yr = -xx * np.sin(t) + yy * np.cos(t)
            sig = 1.0 / (3.0 * sf)
            k = np.exp(-(xr ** 2 + yr ** 2) / (2 * sig ** 2)) * np.cos(2 * np.pi * sf * xr)
            out.append(k - k.mean())
    return np.stack(out)


_BANK = gabor_bank()


def grey(path, px=224):
    import step44_why as W
    CS = W.colour_module()
    return (CS.load_rgb(path, px) @ REC709).astype(np.float64)


def contrast_energy(g, bank=None):
    bank = _BANK if bank is None else bank
    h = bank.shape[-1] // 2
    gp = np.pad(g, h, mode="reflect")
    return float(np.mean([np.mean(fftconvolve(gp, k, mode="valid") ** 2) for k in bank]))


def off_on(g, ref):
    off = float(np.maximum(0.0, ref - g).mean())
    on = float(np.maximum(0.0, g - ref).mean())
    return off, on, (off / (off + on) if off + on > 0 else np.nan)


def image_stats(greys):
    """greys: list of (H, W) grey images of one dataset. Returns {stat: (n,) array}."""
    L = np.array([float(g.astype(np.float32).mean()) for g in greys])
    ref = float(L.mean())
    out = {s: np.empty(len(greys)) for s in STATS}
    for i, g in enumerate(greys):
        out["L"][i] = L[i]
        out["C"][i] = float(g.astype(np.float32).std())
        out["CE"][i] = contrast_energy(g)
        out["OFF"][i], out["ON"][i], out["DD"][i] = off_on(g, ref)
    out["ref"] = ref
    return out


def dataset_images(name):
    """Image paths of 'things720', 'things_test' or 'nsd515' (the orders of the analyses)."""
    if name == "things720":
        import step44_why as W
        return W.things720_images()
    if name == "things_test":
        import step40_common as S40
        from v12_lift import namespaces
        ns12, _ = namespaces()
        imgs = [ns12["find_img"](n) for n in S40.test_stimuli()]
        if any(p is None for p in imgs):
            raise SystemExit("STOP: test image missing")
        return imgs
    if name == "nsd515":
        import nsd_features as F
        return F.image_paths()
    raise ValueError(name)


def stats_for(name):
    """{stat: (n,)} for a dataset, with L and C checked against step44_why.lum_contrast."""
    import step44_why as W
    paths = dataset_images(name)
    out = image_stats([grey(p) for p in paths])
    L, Cn = W.lum_contrast(paths)
    dl, dc = float(np.abs(out["L"] - L).max()), float(np.abs(out["C"] - Cn).max())
    if dl > 1e-6 or dc > 1e-6:
        raise SystemExit(f"STOP: {name}: L / C differ from step 44 (max |d| {dl:.1e}, {dc:.1e})")
    out["check_L"], out["check_C"] = dl, dc
    return out
