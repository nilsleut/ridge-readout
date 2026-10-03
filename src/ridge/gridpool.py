"""
gridpool.py
===========
The grid ladder (config.yaml `grid_ladder`) and the per-image cell-shuffle control.

THE ONE THING THAT MATTERS HERE
-------------------------------
At g = 1 this module must reproduce the paper's condition EXACTLY. It does that by
calling the paper's own expression -- `x.mean([2, 3])`, the tensor method, as written
at learning_rules_v10_sweep_modal.py:169 and models_v2.py:127 -- and not by routing
g = 1 through the general adaptive-pooling path.

Phase 0 measured the two to be bit-identical on CPU/float32 at all six Conv1 shapes,
but that was not tested on a T4, and the working rule is to call the original rather
than rely on an equivalence. `pool_grid(x, 1)` therefore dispatches to the original
expression, and `assert_g1_is_original()` proves the dispatch is real rather than
assumed.

DIMENSIONALITY
--------------
C * g^2, constant across resolution by construction: 32, 128, 512, 2048 for
g = 1, 2, 4, 8. That constancy IS the instrument -- it is what removes the capacity
confound from the resolution axis -- so `assert_equal_dim_across_resolutions` treats a
violation as fatal.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

__all__ = ["pool_grid", "shuffle_cells_per_image", "expected_dim",
           "assert_g1_is_original", "assert_equal_dim_across_resolutions",
           "assert_float32"]


def expected_dim(channels: int, g: int) -> int:
    return channels * g * g


def pool_grid(x: torch.Tensor, g: int) -> torch.Tensor:
    """Pool a feature map (N, C, H, W) to (N, C*g*g).

    g == 1 calls the paper's own expression. g > 1 uses adaptive average pooling to a
    fixed g x g grid, flattened channel-major so that feature index c*g^2 + p is
    channel c at spatial cell p -- the layout `shuffle_cells_per_image` relies on.
    """
    if x.dim() != 4:
        raise ValueError(f"expected (N, C, H, W), got {tuple(x.shape)}")
    if g < 1:
        raise ValueError(f"g must be >= 1, got {g}")
    if g > min(x.shape[2], x.shape[3]):
        # Pooling to a grid finer than the map would upsample, which would make the
        # rung mean something different at different resolutions -- exactly the
        # confound the fixed grid exists to remove.
        raise ValueError(
            f"g={g} exceeds the feature-map side {x.shape[2]}x{x.shape[3]}; that would "
            "upsample rather than pool. Check config.yaml grid_ladder."
        )

    if g == 1:
        # THE PAPER'S CALL, verbatim. Do not 'simplify' this to the general path.
        return x.mean([2, 3])

    return F.adaptive_avg_pool2d(x, g).flatten(1)


def shuffle_cells_per_image(x: torch.Tensor, g: int, seed: int) -> torch.Tensor:
    """Spatial-correspondence control (config.yaml `spatial_correspondence_control`).

    Permutes the g^2 spatial cells with A DIFFERENT PERMUTATION FOR EACH IMAGE, the
    same permutation across channels within an image.

    Why per-image: ridge is invariant to a fixed reordering of features, so a single
    permutation applied to every image would change nothing and the control would be
    vacuous. A per-image permutation destroys the correspondence between feature index
    and retinotopic location across images while preserving, exactly:
      * dimensionality C*g^2,
      * the multiset of values in every channel of every image,
      * the within-image cross-channel structure at each cell.
    So any drop relative to the unshuffled rung is attributable to spatial
    correspondence rather than to capacity or to the activation distribution.

    Takes the pooled (N, C*g*g) tensor, not the raw map.
    """
    if g < 2:
        raise ValueError(f"cell shuffling is meaningless at g={g} (one cell)")
    n, d = x.shape
    cells = g * g
    if d % cells:
        raise ValueError(f"width {d} is not a multiple of g^2={cells}")
    c = d // cells

    v = x.view(n, c, cells)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # One permutation per image, shared across channels.
    perms = torch.stack([torch.randperm(cells, generator=gen) for _ in range(n)])
    idx = perms.unsqueeze(1).expand(n, c, cells).to(x.device)
    return torch.gather(v, 2, idx).reshape(n, d)


# ─────────────────────────────────────────────────────────────────────────────
# Assertions -- each has a negative test in test_phase2_controls.py
# ─────────────────────────────────────────────────────────────────────────────

def assert_g1_is_original(channels: int = 32, sides=(16, 32, 48, 64, 80, 112),
                          batch: int = 8) -> None:
    """Prove pool_grid(x, 1) IS the paper's expression, on this device and dtype.

    Compares against `x.mean([2, 3])` computed inline. Bit-identity, not allclose: if
    the dispatch were ever rerouted through adaptive_avg_pool2d this must fail even
    when the two happen to agree numerically on some machines.
    """
    for s in sides:
        x = torch.randn(batch, channels, s, s)
        if not torch.equal(pool_grid(x, 1), x.mean([2, 3])):
            raise AssertionError(
                f"pool_grid(x, 1) is not bit-identical to x.mean([2,3]) at {s}x{s}. "
                "The g=1 rung no longer reproduces the paper's condition."
            )


def assert_equal_dim_across_resolutions(dims_by_res: Dict[int, int], g: int,
                                        channels: int = 32) -> None:
    """Every resolution must give the same width at a given g."""
    want = expected_dim(channels, g)
    bad = {r: d for r, d in dims_by_res.items() if d != want}
    if bad:
        raise AssertionError(
            f"g={g}: expected width {want} at every resolution, got {bad}. "
            "Dimensionality varying with resolution reintroduces the capacity "
            "confound the fixed grid exists to remove; every number for this "
            "condition would be uninterpretable."
        )
    if len(dims_by_res) < 2:
        # A single resolution cannot demonstrate constancy -- refuse to pass vacuously.
        raise AssertionError(
            f"g={g}: only {len(dims_by_res)} resolution(s) supplied; the check would "
            "be vacuous. Supply all six."
        )


def assert_float32(x: torch.Tensor, what: str = "features") -> None:
    if x.dtype != torch.float32:
        raise AssertionError(
            f"{what} must be float32, got {x.dtype}. float16 would put quantisation "
            "noise on the same order as the effects being measured (~1e-3)."
        )
