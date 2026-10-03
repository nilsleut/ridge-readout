"""
test_phase2_controls.py
=======================
The regression controls, written and run BEFORE the pipeline they protect.

Every assertion in the ridge pipeline gets a negative test here that deliberately
triggers it. An assertion that cannot fail proves nothing -- the same rule
bn_guard.py:66-71 states about vacuous passes, applied to the whole of Phase 2.

Covered:
  1. g=1 pooling IS the paper's x.mean([2,3])
  2. equal feature dimensionality across resolutions within a g
  3. float32, never float16
  4. per-image cell shuffle preserves dim + per-channel distribution, destroys layout
  5. permuted labels -> r ~ 0                       (the pipeline's negative test)
  6. alpha -> infinity -> r -> 0
  7. alpha -> 0 with d > n -> train interpolation, poor test r
  8. alpha at a grid boundary -> AlphaBoundaryError, not a warning
  9. no concept crosses a fold boundary
 10. duplicate check raises on disagreement, expects 30 rule x seed combos
 11. standardization uses TRAIN statistics only
 12. hardened loader raises on a missing checkpoint

Run:  python code/ridge/test_phase2_controls.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src" / "ridge"))
sys.path.insert(0, str(_HERE.parent / "src" / "vendor"))

from gridpool import (pool_grid, shuffle_cells_per_image, expected_dim,   # noqa: E402
                      assert_g1_is_original, assert_equal_dim_across_resolutions,
                      assert_float32)
from ridge_dual import (DualRidge, alpha_grid, standardize, effective_df,   # noqa: E402
                        pearson_columns, r2_columns, mse_columns,
                        select_alpha_per_voxel, AlphaBoundaryError, boundary_sides)
from folds import grouped_folds, assert_no_group_crosses_fold, outer_inner_splits  # noqa: E402
from merge import check_duplicates, assert_rule_seed_coverage, RULES, KEY  # noqa: E402


class R:
    def __init__(self):
        self.fail = []
        self.n = 0

    def ok(self, name, cond, detail=""):
        self.n += 1
        if not cond:
            self.fail.append(f"{name}{': ' + detail if detail else ''}")

    def raises(self, name, exc, fn):
        self.n += 1
        try:
            fn()
        except exc:
            return
        except Exception as e:
            self.fail.append(f"{name}: raised {type(e).__name__} ({e}), expected {exc.__name__}")
            return
        self.fail.append(f"{name}: did NOT raise {exc.__name__}")


def synth(n=200, d=40, v=6, snr=1.0, seed=0):
    """A design with real signal, so 'r ~ 0' tests are not trivially satisfied."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, v))
    Y = X @ W * snr + rng.standard_normal((n, v))
    return X, Y


# ── 1-4: grid ladder ────────────────────────────────────────────────────────

def t_gridpool(r: R):
    assert_g1_is_original()
    r.ok("g=1 is the paper's x.mean([2,3])", True)

    # Explicit, per shape, in case the helper is ever loosened.
    for s in (16, 32, 48, 64, 80, 112):
        x = torch.randn(4, 32, s, s)
        r.ok(f"pool_grid(x,1) bit-identical at {s}", torch.equal(pool_grid(x, 1), x.mean([2, 3])))

    # Widths constant across resolution within a g -- the instrument.
    for g in (1, 2, 4, 8):
        dims = {}
        for px in (32, 64, 96, 128, 160, 224):
            x = torch.randn(3, 32, px // 2, px // 2)
            dims[px] = pool_grid(x, g).shape[1]
        r.ok(f"g={g} width constant across 6 resolutions = {expected_dim(32, g)}",
             set(dims.values()) == {expected_dim(32, g)}, str(dims))
        assert_equal_dim_across_resolutions(dims, g)

    # Negative: a width that varies must be fatal.
    r.raises("varying width across resolutions raises", AssertionError,
             lambda: assert_equal_dim_across_resolutions({32: 32, 224: 64}, 1))
    # Negative: a single resolution must not pass vacuously.
    r.raises("single-resolution dim check is not a vacuous pass", AssertionError,
             lambda: assert_equal_dim_across_resolutions({224: 32}, 1))
    # Negative: g finer than the map would upsample.
    r.raises("g larger than the map raises", ValueError,
             lambda: pool_grid(torch.randn(2, 32, 4, 4), 8))

    # float32 enforcement.
    assert_float32(torch.randn(2, 8))
    r.ok("float32 accepted", True)
    r.raises("float16 features rejected", AssertionError,
             lambda: assert_float32(torch.randn(2, 8).half()))

    # Cell shuffle: preserves dimensionality and the per-channel value multiset,
    # destroys the cell<->index correspondence.
    for g in (4, 8):
        x = pool_grid(torch.randn(16, 32, 32, 32), g)
        s = shuffle_cells_per_image(x, g, seed=7)
        r.ok(f"shuffle g={g} preserves width", s.shape == x.shape)
        a = x.view(16, 32, g * g).sort(dim=2).values
        b = s.view(16, 32, g * g).sort(dim=2).values
        r.ok(f"shuffle g={g} preserves per-channel value multiset", torch.allclose(a, b))
        r.ok(f"shuffle g={g} actually changes the layout", not torch.equal(s, x))
        # Different images must get DIFFERENT permutations, else it is a fixed
        # permutation and ridge is invariant to it -- the control would be vacuous.
        v = x.view(16, 32, g * g)
        w = s.view(16, 32, g * g)
        perm_of = lambda i: [int((v[i, 0] == w[i, 0, j]).nonzero()[0][0]) for j in range(g * g)]
        r.ok(f"shuffle g={g} uses a different permutation per image",
             perm_of(0) != perm_of(1))
    r.raises("cell shuffle at g=1 raises", ValueError,
             lambda: shuffle_cells_per_image(torch.randn(4, 32), 1, seed=0))


# ── 5-8, 11: ridge ──────────────────────────────────────────────────────────

def t_ridge(r: R):
    X, Y = synth(n=200, d=40, v=6, seed=1)
    tr, te = slice(0, 150), slice(150, 200)
    Xtr, Xte = standardize(X[tr], X[te])
    m = DualRidge().fit(Xtr, Y[tr])
    rp = m.r_path(Xte, Y[te])
    r.ok("baseline: signal is recovered", rp.max() > 0.5, f"max r={rp.max():.3f}")

    # 5. Permuted labels -> r ~ 0. THE pipeline's negative test.
    rng = np.random.default_rng(0)
    Yp = Y.copy()
    rng.shuffle(Yp)                       # break the stimulus<->response pairing
    mp = DualRidge().fit(Xtr, Yp[tr])
    rpp = mp.r_path(Xte, Yp[te])
    r.ok("permuted labels give r ~ 0", abs(np.median(rpp)) < 0.15,
         f"median r={np.median(rpp):.3f}")
    r.ok("permuted labels well below intact signal", rpp.max() < rp.max(),
         f"{rpp.max():.3f} vs {rp.max():.3f}")

    # 6. alpha -> infinity: over-regularization destroys predictivity.
    #
    # The Phase 2 spec words this as "alpha -> inf -> r -> 0". Measured, that is FALSE
    # for Pearson r, and the reason is worth recording rather than papering over:
    # r is SCALE-INVARIANT. As alpha grows the prediction collapses onto the training
    # mean, but the surviving deviation -- of order 1/alpha -- keeps its DIRECTION, and
    # r measures only direction. At alpha=1e12 this pipeline returns |r| ~ 0.96 while
    # the predictions are constant to 1e-6.
    #
    # So the assertion is tested on the two quantities that do go to zero: the
    # prediction's departure from the training mean, and R^2. Anyone who later "fixes"
    # the pipeline to make r -> 0 would be introducing a bug.
    big = DualRidge(alphas=np.array([1e2, 1e12])).fit(Xtr, Y[tr])
    preds = big.predict_path(Xte)
    r.ok("alpha -> inf predicts the train mean",
         np.allclose(preds[-1], Y[tr].mean(axis=0), atol=1e-6))
    r.ok("alpha -> inf drives predicted variance to 0",
         preds[-1].std(axis=0).max() < 1e-6, f"max sd={preds[-1].std(axis=0).max():.2e}")
    from ridge_dual import r2_columns
    r2_big = r2_columns(preds[-1], Y[te])
    # R^2 does not land on exactly 0 either, and the reason is not a defect: it is
    # measured against the TEST mean while the collapsed model predicts the TRAIN
    # mean, so the limit is -n*(ybar_train - ybar_test)^2 / SS_tot -- slightly
    # negative. The exact statement of "the model has collapsed to its intercept" is
    # therefore an identity against the constant train-mean predictor, which is what
    # is asserted here rather than an arbitrary threshold.
    const_pred = np.broadcast_to(Y[tr].mean(axis=0), preds[-1].shape)
    r2_const = r2_columns(const_pred, Y[te])
    r.ok("alpha -> inf collapses to exactly the intercept model",
         np.allclose(r2_big, r2_const, atol=1e-9),
         f"max|R2 - R2_const|={np.abs(r2_big - r2_const).max():.2e}")
    r.ok("the collapsed R^2 is <= 0 (never explains variance)", r2_big.max() <= 1e-9,
         f"max R2={r2_big.max():.2e}")
    r2_sane = r2_columns(preds[0], Y[te])
    r.ok("R^2 at a sane alpha is well above the over-regularized one",
         np.median(r2_sane) > np.median(r2_big) + 0.1,
         f"{np.median(r2_sane):.3f} vs {np.median(r2_big):.3f}")
    # And document the scale-invariance explicitly, so it cannot be mistaken for a bug.
    rb = big.r_path(Xte, Y[te])
    r.ok("Pearson r does NOT go to 0 under over-regularization (scale-invariant)",
         abs(rb[-1]).max() > 0.5, f"max|r|={abs(rb[-1]).max():.2e}")

    # 7. alpha -> 0 with d > n: interpolates train, generalizes poorly.
    Xw, Yw = synth(n=60, d=400, v=4, snr=0.3, seed=2)
    trw, tew = slice(0, 40), slice(40, 60)
    Xwtr, Xwte = standardize(Xw[trw], Xw[tew])
    mw = DualRidge(alphas=np.array([1e-10, 1e3])).fit(Xwtr, Yw[trw])
    r_train = mw.r_path(Xwtr, Yw[trw])
    r_test = mw.r_path(Xwte, Yw[tew])
    r.ok("alpha -> 0, d>n: interpolates the training fold",
         np.median(r_train[0]) > 0.999, f"median train r={np.median(r_train[0]):.6f}")
    r.ok("alpha -> 0, d>n: test r is worse than at a sane alpha",
         np.median(r_test[0]) < np.median(r_test[1]),
         f"{np.median(r_test[0]):.3f} vs {np.median(r_test[1]):.3f}")

    # df must be monotone decreasing in alpha and bounded by min(d, n).
    lam = mw.lam_
    df = effective_df(lam, np.array([1e-10, 1e0, 1e10]))
    r.ok("df decreases with alpha", df[0] > df[1] > df[2], str(df))
    r.ok("df <= min(d, n)", df[0] <= min(400, 40) + 1e-6, f"df={df[0]:.2f}")

    # 8. D1 boundary assertion: LOW edge only, RELIABLE voxels only, one aborts.
    #    D3 grid: 1e-6 .. 1e8, 29 points -- edges have MOVED, so these tests are
    #    re-derived against the new grid rather than carried over.
    a = alpha_grid()
    r.ok("D3: grid is 1e-6..1e8 with 29 points",
         len(a) == 29 and abs(a[0] - 1e-6) < 1e-18 and abs(a[-1] - 1e8) < 1e-6,
         f"{a[0]:.0e}..{a[-1]:.0e} n={len(a)}")
    r.ok("D3: 2 alphas per decade", abs((len(a) - 1) / 14 - 2.0) < 1e-9)
    n_v = 20

    # LOW edge in a reliable voxel -> abort. One voxel is enough.
    c_lo = np.zeros((len(a), n_v)); c_lo[len(a) // 2, :] = 1.0
    c_lo[len(a) // 2, 3] = 0.0; c_lo[0, 3] = 1.0
    r.raises("D1: ONE reliable voxel at the LOW edge aborts", AlphaBoundaryError,
             lambda: select_alpha_per_voxel(c_lo, a, context="t-lo"))

    # HIGH edge is NEVER an alarm, even at 100%.
    c_hi = np.zeros((len(a), n_v)); c_hi[-1, :] = 1.0
    idx, sel, st = select_alpha_per_voxel(c_hi, a, context="t-hi")
    r.ok("D1: 100% at the HIGH edge does NOT abort", st["hi_rel"] == 1.0)
    r.ok("D1: high-edge selection returns alpha_max", np.all(sel == a[-1]))

    # LOW edge in an UNRELIABLE voxel is not an alarm either.
    reliable = np.ones(n_v, bool); reliable[3] = False
    idx, sel, st = select_alpha_per_voxel(c_lo, a, reliable=reliable, context="t-lo-unrel")
    r.ok("D1: LOW edge in an unreliable voxel does not abort",
         st["lo_unrel"] == 1.0 and st["lo_rel"] == 0.0)

    # Interior selection passes, all four cells zero.
    c_mid = np.zeros((len(a), n_v)); c_mid[len(a) // 2, :] = 1.0
    idx, sel, st = select_alpha_per_voxel(c_mid, a, context="t-mid")
    r.ok("D1: interior selection passes", np.all(idx == len(a) // 2))
    r.ok("D1: reliable cells are zero on an interior selection",
         st["lo_rel"] == 0.0 and st["hi_rel"] == 0.0)
    # With reliable=None every voxel is reliable, so the unreliable cells are
    # computed over an EMPTY set. nan is the honest value there: reporting 0.0
    # would claim a measurement that was never made.
    r.ok("D1: empty unreliable subpopulation reports nan, not 0.0",
         np.isnan(st["lo_unrel"]) and np.isnan(st["hi_unrel"]))
    for k in ("lo_rel", "hi_rel", "lo_unrel", "hi_unrel", "n_lo_rel",
              "median_selected_alpha", "alpha_min", "alpha_max"):
        r.ok(f"D1: stats carry {k}", k in st)

    # All four cells split correctly.
    c_mix = np.zeros((len(a), 10)); c_mix[len(a) // 2, :] = 1.0
    c_mix[len(a) // 2, :4] = 0.0; c_mix[0, :2] = 1.0; c_mix[-1, 2:4] = 1.0
    rel4 = np.zeros(10, bool); rel4[2:7] = True      # voxels 0,1 (low) unreliable
    idx, sel, st = select_alpha_per_voxel(c_mix, a, reliable=rel4,
                                          assert_low_boundary=False)
    r.ok("D1: lo_unrel cell", abs(st["lo_unrel"] - 0.4) < 1e-9, str(st["lo_unrel"]))
    r.ok("D1: hi_rel cell", abs(st["hi_rel"] - 0.4) < 1e-9, str(st["hi_rel"]))
    r.ok("D1: lo_rel cell zero here", st["lo_rel"] == 0.0)

    # Empty reliable mask must not pass vacuously.
    r.raises("D1: empty reliable mask aborts rather than passing vacuously",
             AlphaBoundaryError,
             lambda: select_alpha_per_voxel(c_mid, a, reliable=np.zeros(n_v, bool),
                                            context="t-empty"))

    # D2: MSE is the selection criterion; argmin MSE == argmax R^2 exactly.
    rngm = np.random.default_rng(11)
    pred = rngm.standard_normal((40, 5)); true = rngm.standard_normal((40, 5))
    from ridge_dual import mse_columns as _mse
    mse = _mse(pred, true)
    r.ok("D2: mse_columns matches a direct computation",
         np.allclose(mse, ((true - pred) ** 2).mean(axis=0)))
    r.ok("D2: argmin MSE == argmax R2 on a fixed fold",
         np.all(np.argmin(np.stack([_mse(pred, true), _mse(pred * 2, true)]), axis=0)
                == np.argmax(np.stack([r2_columns(pred, true),
                                       r2_columns(pred * 2, true)]), axis=0)))

    # 11. Standardization uses TRAIN statistics only.
    A = np.random.default_rng(3).standard_normal((50, 5)) * 3 + 7
    B = np.random.default_rng(4).standard_normal((20, 5)) * 11 - 2
    As, Bs = standardize(A, B)
    r.ok("train fold standardizes to mean 0", np.allclose(As.mean(0), 0, atol=1e-10))
    r.ok("train fold standardizes to sd 1", np.allclose(As.std(0), 1, atol=1e-10))
    r.ok("test fold is NOT standardized to its own stats",
         not np.allclose(Bs.mean(0), 0, atol=1e-6),
         "test was rescaled by its own mean -- that is leakage")
    # Zero-variance features must not produce NaN.
    Z = np.ones((10, 3)); Z[:, 1] = np.arange(10)
    Zs, = standardize(Z)
    r.ok("dead features become 0, not NaN", np.isfinite(Zs).all() and np.allclose(Zs[:, 0], 0))

    # pearson_columns must not emit NaN for constant columns.
    c = pearson_columns(np.ones((10, 2)), np.random.default_rng(0).standard_normal((10, 2)))
    r.ok("constant column gives r=0, not NaN", np.isfinite(c).all() and np.allclose(c, 0))


# ── 9: folds ────────────────────────────────────────────────────────────────

def t_folds(r: R):
    # The real case: 720 concepts, one exemplar each.
    g1 = np.arange(720)
    f = grouped_folds(g1, 8, seed=20260818)
    assert_no_group_crosses_fold(g1, f)
    r.ok("720 singleton concepts -> 8 folds partition exactly",
         sum(len(x) for x in f) == 720 and len(np.unique(np.concatenate(f))) == 720)
    r.ok("folds are balanced at 90 each", {len(x) for x in f} == {90},
         str(sorted(len(x) for x in f)))

    # The case the assertion exists for: multiple exemplars per concept.
    g2 = np.repeat(np.arange(120), 6)
    f2 = grouped_folds(g2, 8, seed=1)
    assert_no_group_crosses_fold(g2, f2)
    r.ok("multi-exemplar concepts stay within one fold", True)

    # Negative: a hand-made split that breaks a concept must be caught.
    bad = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    r.raises("concept crossing a fold boundary is caught", AssertionError,
             lambda: assert_no_group_crosses_fold(np.array([0, 0, 0, 0, 1, 1]), bad))
    # Negative: a non-partition must be caught.
    r.raises("non-partition folds caught", AssertionError,
             lambda: assert_no_group_crosses_fold(np.arange(6),
                                                  [np.array([0, 1]), np.array([1, 2])]))

    # Outer/inner: the outer test fold must never appear in any inner split.
    n_out = 0
    for tr, te, inner in outer_inner_splits(np.arange(200), 8, 5, seed=42):
        n_out += 1
        r.ok("outer train/test disjoint", len(np.intersect1d(tr, te)) == 0)
        for itr, iva in inner:
            if len(np.intersect1d(itr, te)) or len(np.intersect1d(iva, te)):
                r.ok("inner splits never touch the outer test fold", False)
                break
        else:
            r.ok("inner splits never touch the outer test fold", True)
    r.ok("8 outer folds produced", n_out == 8, str(n_out))


# ── 10: merge ───────────────────────────────────────────────────────────────

def t_merge(r: R):
    base = {k: "x" for k in KEY}
    base.update(dict(seed=42, resolution=224, grid=1, fold=0))

    same = pd.DataFrame([{**base, "r": 0.5}, {**base, "r": 0.5}])
    out = check_duplicates(same)
    r.ok("identical duplicates collapse to one row", len(out) == 1)

    diff = pd.DataFrame([{**base, "r": 0.5}, {**base, "r": 0.5 + 1e-9}])
    r.raises("disagreeing duplicates RAISE, not collapse", RuntimeError,
             lambda: check_duplicates(diff))

    # atol is 1e-12, so a 1e-11 difference must still be caught.
    tiny = pd.DataFrame([{**base, "r": 0.5}, {**base, "r": 0.5 + 1e-11}])
    r.raises("1e-11 disagreement caught at atol=1e-12", RuntimeError,
             lambda: check_duplicates(tiny))

    # NaN in some rows but not others is a disagreement.
    nan_mix = pd.DataFrame([{**base, "r": np.nan}, {**base, "r": 0.5}])
    r.raises("NaN in only some duplicate rows caught", RuntimeError,
             lambda: check_duplicates(nan_mix))

    r.ok("six rules expected, not five", len(RULES) == 6, str(len(RULES)))
    full = pd.DataFrame([{"rule": ru, "seed": s} for ru in RULES
                         for s in [42, 123, 456, 789, 1337]])
    assert_rule_seed_coverage(full)
    r.ok("full 30-combination coverage accepted", True)
    r.raises("missing Random (BN-calibrated) is caught", AssertionError,
             lambda: assert_rule_seed_coverage(
                 full[full.rule != "Random Weights (BN-calibrated)"]))
    r.raises("a missing seed is caught", AssertionError,
             lambda: assert_rule_seed_coverage(full[full.seed != 1337]))


# ── 12: loader ──────────────────────────────────────────────────────────────

def t_loader(r: R):
    from models_v2 import load_model
    with tempfile.TemporaryDirectory() as td:
        r.raises("hardened loader raises on a missing checkpoint", FileNotFoundError,
                 lambda: load_model("backprop", weights_dir=Path(td)))


def main() -> int:
    r = R()
    for fn in (t_gridpool, t_ridge, t_folds, t_merge, t_loader):
        fn(r)
    print("Phase 2 regression controls")
    print("-" * 70)
    if r.fail:
        for f in r.fail:
            print(f"  FAIL  {f}")
        print(f"\n{len(r.fail)} failure(s) of {r.n} checks")
        return 1
    print(f"  all {r.n} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
