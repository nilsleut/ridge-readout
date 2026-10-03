"""
test_loader_hardening.py
========================
Negative tests for every raise added to models_v2.load_model (Change 4), plus
positive tests that the six real checkpoints still load.

An assertion that cannot fail proves nothing -- that is how the original BatchNorm
defect hid (bn_guard.py:66-71 makes the same point about vacuous passes). So every
raise here gets a test that deliberately triggers it.

Run:
    python code/ridge/test_loader_hardening.py                     # negative tests only
    python code/ridge/test_loader_hardening.py --weights <dir>     # + real checkpoints
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src" / "vendor"))
sys.path.insert(0, str(_HERE.parent / "src" / "ridge"))

import models_v2  # noqa: E402
from models_v2 import load_model, _resolve_weight_file, _WEIGHT_FILES  # noqa: E402
import paths  # noqa: E402

DEFAULT_WEIGHTS = paths.ckpt_dir()

ALL_RULES = ["random", "backprop", "feedback_alignment", "predictive_coding",
             "stdp", "random_bn_calibrated"]


class Results:
    def __init__(self):
        self.failures = []
        self.n = 0

    def check(self, name, cond, detail=""):
        self.n += 1
        if not cond:
            self.failures.append(f"{name}{(': ' + detail) if detail else ''}")

    def expect_raises(self, name, exc_types, fn):
        """The negative test primitive: fn MUST raise one of exc_types."""
        self.n += 1
        try:
            fn()
        except exc_types:
            return
        except Exception as e:
            self.failures.append(
                f"{name}: raised {type(e).__name__} ({e}), expected one of "
                f"{[t.__name__ for t in exc_types]}"
            )
            return
        self.failures.append(f"{name}: did NOT raise (silent fallback still present)")


# ─────────────────────────────────────────────────────────────────────────────
# Negative tests -- one per raise
# ─────────────────────────────────────────────────────────────────────────────

def negative_tests(r: Results):
    # 1. models_v2.py:497-498 (was): missing file -> warned and returned a
    #    randomly initialised model. Must now raise.
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td)
        for rule in ["random", "backprop", "feedback_alignment", "predictive_coding"]:
            r.expect_raises(
                f"missing checkpoint raises [{rule}]",
                (FileNotFoundError,),
                lambda rule=rule: load_model(rule, weights_dir=empty),
            )
        # Same for the STDP branch, which had its own separate fallback.
        r.expect_raises(
            "missing checkpoint raises [stdp]",
            (FileNotFoundError,),
            lambda: load_model("stdp", weights_dir=empty),
        )

    # 2. A weights_dir that does not exist at all -- the defect this module's
    #    docstring records as having actually shipped (stale hard-coded path made
    #    every rule fall back to random init).
    r.expect_raises(
        "nonexistent weights_dir raises",
        (FileNotFoundError,),
        lambda: load_model("backprop", weights_dir=Path("/definitely/not/here")),
    )

    # 3. models_v2.py:460 (was): _seed0.pt preferred, silently falling back when
    #    absent. A requested seed that does not exist must raise, never substitute
    #    a different seed's weights.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # Only seed 3 present; asking for seed 1 must not return seed 3.
        torch.save({"x": torch.zeros(1)},
                   d / _WEIGHT_FILES["backprop"].replace(".pt", "_seed3.pt"))
        r.expect_raises(
            "requesting an absent seed raises (no cross-seed substitution)",
            (FileNotFoundError,),
            lambda: load_model("backprop", weights_dir=d, seed_idx=1),
        )
        # And the un-suffixed name must not satisfy a seed_idx=2 request either.
        torch.save({"x": torch.zeros(1)}, d / _WEIGHT_FILES["backprop"])
        r.expect_raises(
            "un-suffixed file does not satisfy seed_idx=2",
            (FileNotFoundError,),
            lambda: load_model("backprop", weights_dir=d, seed_idx=2),
        )
        # seed_idx=0 IS allowed to resolve to the legacy un-suffixed name.
        r.check(
            "seed_idx=0 resolves to legacy un-suffixed name",
            _resolve_weight_file(d, "backprop", 0).name == _WEIGHT_FILES["backprop"],
        )

    # 4. Incomplete STDP checkpoint: conv-only. v1 warned and continued, giving
    #    identity BN and a random FC1. Must now raise.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        torch.save(
            {"conv1.weight": torch.zeros(32, 3, 3, 3),
             "conv2.weight": torch.zeros(64, 32, 3, 3),
             "conv3.weight": torch.zeros(128, 64, 3, 3)},
            d / _WEIGHT_FILES["stdp"],
        )
        r.expect_raises(
            "conv-only STDP checkpoint raises (would give identity BN + random FC1)",
            (KeyError,),
            lambda: load_model("stdp", weights_dir=d),
        )

    # 5. Partial / mismatched state dict for an nn.Module rule must raise via
    #    strict=True rather than loading some tensors and leaving others random.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        torch.save({"conv1.0.weight": torch.zeros(32, 3, 3, 3)},
                   d / _WEIGHT_FILES["backprop"])
        r.expect_raises(
            "partial state_dict raises (strict=True)",
            (RuntimeError,),
            lambda: load_model("backprop", weights_dir=d),
        )

    # 6. Unknown rule still raises, and the new sixth rule is registered.
    r.expect_raises(
        "unknown rule raises",
        (ValueError,),
        lambda: load_model("not_a_rule", weights_dir=DEFAULT_WEIGHTS),
    )
    r.check("random_bn_calibrated is registered in _WEIGHT_FILES",
            "random_bn_calibrated" in _WEIGHT_FILES)
    r.check("all six conditions are loadable rules",
            set(ALL_RULES) <= set(_WEIGHT_FILES),
            f"missing {set(ALL_RULES) - set(_WEIGHT_FILES)}")

    # 7. The loader must never return a model still in training mode.
    r.check("assert_bn_eval is imported by models_v2",
            hasattr(models_v2, "assert_bn_eval"))


# ─────────────────────────────────────────────────────────────────────────────
# Positive tests -- the real checkpoints must still load
# ─────────────────────────────────────────────────────────────────────────────

def positive_tests(r: Results, weights_dir: Path):
    import torch.nn as nn

    for rule in ALL_RULES:
        try:
            m = load_model(rule, weights_dir=weights_dir)
        except Exception as e:
            r.check(f"real checkpoint loads [{rule}]", False,
                    f"{type(e).__name__}: {e}")
            continue
        r.check(f"real checkpoint loads [{rule}]", True)

        # BN must be in eval mode -- the original defect, checked directly.
        bns = [mm for mm in (m.modules() if isinstance(m, nn.Module)
                             else [o for o in vars(m).values() if isinstance(o, nn.Module)])
               if isinstance(mm, nn.BatchNorm2d)]
        if not isinstance(m, nn.Module):
            bns = []
            for o in vars(m).values():
                if isinstance(o, nn.Module):
                    bns += [mm for mm in o.modules() if isinstance(mm, nn.BatchNorm2d)]
        r.check(f"BN layers found [{rule}]", len(bns) > 0)
        r.check(f"all BN in eval mode [{rule}]",
                all(not b.training for b in bns))

        # A forward pass must produce the documented Conv1 shape.
        c1, c2, c3, h1 = m.get_features(torch.randn(2, 3, 224, 224))
        r.check(f"Conv1 pooled dim is 32 [{rule}]", tuple(c1.shape) == (2, 32),
                f"got {tuple(c1.shape)}")

    # random vs random_bn_calibrated must be DIFFERENT models -- they share
    # convolutional weights but differ in stored BN statistics. If they came out
    # identical, the sixth condition would be a silent duplicate.
    try:
        a = load_model("random", weights_dir=weights_dir)
        b = load_model("random_bn_calibrated", weights_dir=weights_dir)
        conv_same = torch.equal(a.conv1[0].weight, b.conv1[0].weight)
        bn_same = torch.equal(a.conv1[1].running_var, b.conv1[1].running_var)
        r.check("random and bn-calibrated share conv weights", conv_same)
        r.check("random and bn-calibrated differ in BN stats", not bn_same)
    except Exception as e:
        r.check("random vs bn-calibrated comparison", False, f"{type(e).__name__}: {e}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=None,
                   help=f"checkpoint dir for positive tests (default {DEFAULT_WEIGHTS})")
    p.add_argument("--skip-positive", action="store_true")
    a = p.parse_args()

    r = Results()
    print("loader hardening tests")
    print("-" * 70)
    negative_tests(r)
    n_neg = r.n
    print(f"  negative tests (every raise triggered deliberately): {n_neg}")

    if not a.skip_positive:
        wd = a.weights or DEFAULT_WEIGHTS
        if wd.exists():
            positive_tests(r, wd)
            print(f"  positive tests (real checkpoints in {wd.name}): {r.n - n_neg}")
        else:
            print(f"  positive tests SKIPPED: {wd} not present")

    print("-" * 70)
    if r.failures:
        for f in r.failures:
            print(f"  FAIL  {f}")
        print(f"\n{len(r.failures)} failure(s) of {r.n} checks")
        return 1
    print(f"  all {r.n} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
