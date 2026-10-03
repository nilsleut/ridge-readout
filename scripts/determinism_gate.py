"""
determinism_gate.py
===================
The seed-42 bit-identity gate for the 30-checkpoint retrain.

WHY THIS EXISTS, AND WHY IT RUNS BEFORE THE RETRAIN
---------------------------------------------------
The retrain regenerates seed 42, for which six checkpoints already exist. Comparing
them afterwards, by eye, would let the interpretation be chosen after the result is
known. So the comparison is a test with BOTH branches written down first:

  identical      -> training is deterministic. Seed-level SD measures variation in
                    initialization, which is what the design intends the unit of
                    inference to be.
  not identical  -> training carries a nondeterministic component, and every +/- in
                    the paper and in this work has a second, currently unlabelled
                    variance source. Report per-tensor magnitude and STOP.

Both outcomes are usable. Neither is a failure of the gate.

WHAT MAKES A MISMATCH SERIOUS
-----------------------------
The repo already documents (README.md "Determinism"; paper Methods) that convolution
backward is nondeterministic on GPU, so gradient-trained conditions reproduce only to
about 1e-3 at V1, while conditions that never backpropagate through a convolution
(random, predictive coding, STDP) reproduce bit-identically.

That gives the mismatch two very different meanings, so this module classifies rather
than just diffing:

  * a mismatch confined to Backprop / Feedback Alignment is CONSISTENT with what is
    already documented;
  * a mismatch in Random Weights, Random Weights (BN-calibrated), Predictive Coding or
    STDP CONTRADICTS the documented claim, and is the more serious of the two findings.

Tensors are additionally grouped conv / bn / fc, because "the conv weights are
bit-identical and only the readout drifted" and "the conv weights moved" are different
facts and the conv weights are what the V1 claim rests on.

USAGE
-----
    python code/ridge/determinism_gate.py \
        --reference /path/to/outputs_bnfix/checkpoints \
        --candidate /path/to/outputs_allseeds/checkpoints \
        --seed-idx 0

Exit codes:  0 = identical (branch 1)
             2 = mismatch, report emitted (branch 2)
             1 = the gate could not run (missing files, key mismatch, ...)

Self-test (synthetic, exercises both branches, no checkpoints needed):
    python code/ridge/determinism_gate.py --self-test
"""

import argparse
import json
import sys
from pathlib import Path

import torch

__all__ = ["RULES", "RULE_KEY", "EXPECT_BIT_IDENTICAL", "compare_checkpoint",
           "run_gate", "GateResult"]

# Canonical rule list. SIX rules, not five: Random Weights (BN-calibrated) is a
# distinct condition (num_batches_tracked = 50; identity BN would be 0).
RULES = [
    "Random Weights",
    "Backprop",
    "Feedback Alignment",
    "Predictive Coding",
    "STDP",
    "Random Weights (BN-calibrated)",
]

N_SEEDS = 5
N_CHECKPOINTS_EXPECTED = len(RULES) * N_SEEDS   # 30


def RULE_KEY(rule: str) -> str:
    """Same transform the sweep uses for filenames (v10:434)."""
    return rule.lower().replace(" ", "_")


# Rules that never backpropagate through a convolution. The repo claims these
# reproduce bit-identically; a mismatch here contradicts published documentation.
EXPECT_BIT_IDENTICAL = {
    "Random Weights",
    "Random Weights (BN-calibrated)",
    "Predictive Coding",
    "STDP",
}
# Rules for which ~1e-3 drift is already documented.
DOCUMENTED_NONDETERMINISTIC = {"Backprop", "Feedback Alignment"}


def _tensor_group(key: str) -> str:
    """conv / bn / fc / other. Covers all four naming schemes (Phase 0/F)."""
    k = key.lower()
    if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
        return "bn"
    # BN affine params live at conv{n}.1.{weight,bias} for BP/Random/FA, and at
    # bn{n}.{weight,bias} for PC/STDP.
    if k.startswith("bn") or ".1.weight" in k or ".1.bias" in k:
        return "bn"
    if k.startswith("fc"):
        return "fc"
    # conv1.0.weight (BP/Random), conv1.0.W/.b/.B_feedback (FA),
    # W1.weight / P2.weight (PC), conv1.weight (STDP).
    if k.startswith("conv") or k.startswith("w") or k.startswith("p"):
        return "conv"
    return "other"


class GateResult:
    """Structured verdict, so both branches are machine-readable."""

    def __init__(self):
        self.per_rule = {}          # rule -> dict
        self.errors = []            # gate could not run

    @property
    def all_identical(self) -> bool:
        """Every rule that WAS compared came out identical.

        Note this says nothing about rules that were not compared -- see
        `is_complete`. Conflating "nothing mismatched" with "everything matched"
        is precisely the vacuous pass this project has been bitten by before.
        """
        return bool(self.per_rule) and all(
            r["identical"] for r in self.per_rule.values()
        )

    @property
    def is_complete(self) -> bool:
        """All six rules were actually compared, with no gate errors."""
        return not self.errors and set(self.per_rule) == set(RULES)

    @property
    def any_mismatch(self) -> bool:
        return any(not r["identical"] for r in self.per_rule.values())

    @property
    def contradicts_documentation(self) -> list:
        """Rules that mismatched but are documented as bit-identical."""
        return [
            rule for rule, r in self.per_rule.items()
            if not r["identical"] and rule in EXPECT_BIT_IDENTICAL
        ]

    @property
    def consistent_with_documentation(self) -> list:
        return [
            rule for rule, r in self.per_rule.items()
            if not r["identical"] and rule in DOCUMENTED_NONDETERMINISTIC
        ]

    def to_dict(self) -> dict:
        return {
            "all_identical": self.all_identical,
            "contradicts_documentation": self.contradicts_documentation,
            "consistent_with_documentation": self.consistent_with_documentation,
            "errors": self.errors,
            "per_rule": self.per_rule,
        }


def compare_checkpoint(ref: dict, cand: dict) -> dict:
    """Compare two state dicts tensor by tensor at tolerance 0.0.

    Returns a dict with `identical`, per-tensor diffs for every tensor that is not
    bit-identical, and a per-group (conv/bn/fc) summary. Raises ValueError on a key
    mismatch: differing key sets mean the two checkpoints are not the same model, and
    that is a gate failure, not a tolerance question.
    """
    ref_keys, cand_keys = set(ref), set(cand)
    if ref_keys != cand_keys:
        only_ref = sorted(ref_keys - cand_keys)
        only_cand = sorted(cand_keys - ref_keys)
        raise ValueError(
            f"key sets differ; not the same model. "
            f"only in reference: {only_ref}; only in candidate: {only_cand}"
        )

    diffs = {}
    groups = {}
    for k in sorted(ref_keys):
        a, b = ref[k], cand[k]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            if a != b:
                diffs[k] = {"group": _tensor_group(k), "non_tensor": [a, b]}
            continue
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch at {k}: {tuple(a.shape)} vs {tuple(b.shape)}")

        # Bit-identity is the requirement. torch.equal, not allclose.
        if torch.equal(a, b):
            continue

        af, bf = a.double(), b.double()
        absdiff = (af - bf).abs()
        ref_norm = af.norm().item()
        g = _tensor_group(k)
        diffs[k] = {
            "group": g,
            "shape": tuple(a.shape),
            "max_abs_diff": absdiff.max().item(),
            "mean_abs_diff": absdiff.mean().item(),
            "rel_norm_diff": ((af - bf).norm().item() / ref_norm) if ref_norm > 0 else float("inf"),
            "n_elements_differing": int((af != bf).sum().item()),
            "n_elements": a.numel(),
        }
        gg = groups.setdefault(g, {"n_tensors": 0, "max_abs_diff": 0.0, "max_rel_norm_diff": 0.0})
        gg["n_tensors"] += 1
        gg["max_abs_diff"] = max(gg["max_abs_diff"], diffs[k]["max_abs_diff"])
        gg["max_rel_norm_diff"] = max(gg["max_rel_norm_diff"], diffs[k]["rel_norm_diff"])

    return {
        "identical": not diffs,
        "n_tensors": len(ref_keys),
        "n_tensors_differing": len(diffs),
        "groups_affected": groups,
        "per_tensor": diffs,
    }


def _ref_path(ref_dir: Path, rule: str) -> Path:
    """Existing checkpoints carry no seed suffix (they are seed 0 implicitly)."""
    return ref_dir / f"model_weights_{RULE_KEY(rule)}.pt"


def _cand_path(cand_dir: Path, rule: str, seed_idx: int) -> Path:
    """Retrained checkpoints are seed-suffixed."""
    return cand_dir / f"model_weights_{RULE_KEY(rule)}_seed{seed_idx}.pt"


def run_gate(ref_dir: Path, cand_dir: Path, seed_idx: int = 0,
             rules=None) -> GateResult:
    """Run the gate over every rule. Never raises for a mismatch — that is branch 2."""
    rules = RULES if rules is None else rules
    res = GateResult()
    for rule in rules:
        rp, cp = _ref_path(Path(ref_dir), rule), _cand_path(Path(cand_dir), rule, seed_idx)
        if not rp.exists():
            res.errors.append(f"{rule}: reference missing at {rp}")
            continue
        if not cp.exists():
            res.errors.append(f"{rule}: candidate missing at {cp}")
            continue
        ref = torch.load(rp, map_location="cpu", weights_only=True)
        cand = torch.load(cp, map_location="cpu", weights_only=True)
        try:
            res.per_rule[rule] = compare_checkpoint(ref, cand)
        except ValueError as e:
            res.errors.append(f"{rule}: {e}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Reporting — both branches, written out explicitly
# ─────────────────────────────────────────────────────────────────────────────

def report(res: GateResult) -> int:
    print("=" * 78)
    print("SEED-42 DETERMINISM GATE")
    print("=" * 78)

    if res.errors:
        print("\nGATE COULD NOT RUN for some rules:")
        for e in res.errors:
            print(f"  ! {e}")

    if not res.per_rule:
        print("\nNo rule could be compared. Gate inconclusive.")
        return 1

    print(f"\n{'rule':<34} {'verdict':<16} {'tensors differing':>18}")
    print("-" * 78)
    for rule in RULES:
        if rule not in res.per_rule:
            print(f"{rule:<34} {'NOT COMPARED':<16} {'-':>18}")
            continue
        r = res.per_rule[rule]
        verdict = "bit-identical" if r["identical"] else "MISMATCH"
        print(f"{rule:<34} {verdict:<16} "
              f"{r['n_tensors_differing']:>7} / {r['n_tensors']:<8}")

    # ── Branch 1 ────────────────────────────────────────────────────────────
    if res.all_identical and res.is_complete:
        print("\n" + "-" * 78)
        print("BRANCH 1: IDENTICAL.")
        print("-" * 78)
        print("Training is deterministic for all six rules at seed 42. Seed-level SD\n"
              "therefore measures variation in initialization, which is what the design\n"
              "intends the unit of inference to be. Proceed.")
        return 0

    # ── Partial: no mismatch, but not everything was compared ───────────────
    # This is NOT branch 2. Reporting an incomplete run as a mismatch would be a
    # false alarm; reporting it as a pass would be a vacuous one.
    if not res.any_mismatch:
        compared = [r for r in RULES if r in res.per_rule]
        missing = [r for r in RULES if r not in res.per_rule]
        print("\n" + "-" * 78)
        print("PARTIAL: no mismatch, but the gate is INCOMPLETE.")
        print("-" * 78)
        print(f"Compared and bit-identical ({len(compared)}/{len(RULES)}): "
              f"{', '.join(compared)}")
        print(f"Not compared ({len(missing)}): {', '.join(missing)}")
        print("\nNo verdict on the rules that were not compared. The gate is not\n"
              "satisfied until all six are present.")
        return 3

    # ── Branch 2 ────────────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("BRANCH 2: NOT IDENTICAL.")
    print("-" * 78)
    print("Training carries a nondeterministic component. Every +/- in the paper and in\n"
          "this work therefore has a second variance source that is currently\n"
          "unlabelled. Per-tensor magnitudes follow. STOPPING.\n")

    for rule in RULES:
        r = res.per_rule.get(rule)
        if r is None or r["identical"]:
            continue
        # NB: the repo's documented "~1e-3" figure is about rho at V1, NOT about
        # weight tensors. Weight-space divergence and output divergence are
        # different quantities: 40 epochs of Adam amplify a 1-ULP difference in
        # conv backward into a substantially different point in weight space that
        # still computes nearly the same function. So a large weight difference
        # here is not itself evidence against the documented claim -- only
        # check_csv_reproduction.py tests that claim. Do not read the tag as a
        # magnitude judgement.
        tag = ("CONTRADICTS documented bit-identity"
               if rule in EXPECT_BIT_IDENTICAL else
               "in the documented-nondeterministic set; magnitude judged on rho, "
               "not on weights -- see check_csv_reproduction.py")
        print(f"  {rule}  [{tag}]")
        for g, gg in sorted(r["groups_affected"].items()):
            print(f"    group {g:<6} {gg['n_tensors']:>2} tensor(s)  "
                  f"max|diff|={gg['max_abs_diff']:.3e}  "
                  f"max rel-norm={gg['max_rel_norm_diff']:.3e}")
        for k, d in sorted(r["per_tensor"].items(),
                           key=lambda kv: -kv[1].get("max_abs_diff", 0))[:6]:
            if "max_abs_diff" not in d:
                continue
            print(f"      {k:<28} max|diff|={d['max_abs_diff']:.3e}  "
                  f"rel-norm={d['rel_norm_diff']:.3e}  "
                  f"{d['n_elements_differing']}/{d['n_elements']} elems")
        print()

    if res.contradicts_documentation:
        print("  ** The mismatch includes rules documented as reproducing")
        print("     bit-identically: " + ", ".join(res.contradicts_documentation))
        print("     This CONTRADICTS README.md 'Determinism' and the paper's Methods,")
        print("     and is the more serious of the two possible findings.")
    elif res.consistent_with_documentation:
        print("  ** The mismatch is confined to gradient-trained rules")
        print("     (" + ", ".join(res.consistent_with_documentation) + "),")
        print("     which is the qualitative pattern the documented nondeterminism of")
        print("     convolution backward on GPU predicts. It is still a second variance")
        print("     source and must be labelled as one.")
        print()
        print("     This gate does NOT establish the SIZE of the consequence. Weight-")
        print("     space divergence over 40 epochs of Adam can be large while the")
        print("     computed function, and hence rho, stays close. The documented")
        print("     ~1e-3 figure is about rho. Run check_csv_reproduction.py, and")
        print("     read the effect on the headline gap, before concluding anything")
        print("     about magnitude.")
    return 2


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: exercises BOTH branches on synthetic checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "conv1.0.weight": torch.randn(4, 3, 3, 3, generator=g),
        "conv1.1.weight": torch.randn(4, generator=g),
        "conv1.1.running_mean": torch.zeros(4),
        "conv1.1.num_batches_tracked": torch.tensor(0),
        "fc1.weight": torch.randn(8, 4, generator=g),
    }


def self_test() -> int:
    failures = []

    # Branch 1: identical inputs must report identical.
    a, b = _synthetic(0), _synthetic(0)
    r = compare_checkpoint(a, b)
    if not r["identical"] or r["n_tensors_differing"] != 0:
        failures.append("identical checkpoints were not reported identical")

    # Branch 2a: a one-element perturbation must be caught. This is the negative
    # test for the assertion: a gate that cannot fail proves nothing.
    c = _synthetic(0)
    c["conv1.0.weight"] = c["conv1.0.weight"].clone()
    c["conv1.0.weight"][0, 0, 0, 0] += 1e-7
    r = compare_checkpoint(a, c)
    if r["identical"]:
        failures.append("a 1e-7 perturbation was NOT caught (tolerance is not 0.0)")
    elif r["per_tensor"]["conv1.0.weight"]["group"] != "conv":
        failures.append("perturbed conv tensor was not grouped as conv")
    elif r["per_tensor"]["conv1.0.weight"]["n_elements_differing"] != 1:
        failures.append("element-differing count wrong")

    # Branch 2b: the strongest possible negative test — a ONE-ULP change, the
    # smallest difference float32 can represent, must still be caught. (A naive
    # "* (1 + 1e-12)" would not test anything: 1e-12 is below float32 epsilon
    # ~1.19e-7, so the multiply is a no-op and the tensors really are identical.
    # nextafter moves by exactly one representable step.)
    d = _synthetic(0)
    d["fc1.weight"] = d["fc1.weight"].clone()
    d["fc1.weight"][0, 0] = torch.nextafter(
        d["fc1.weight"][0, 0], torch.tensor(float("inf"))
    )
    if torch.equal(a["fc1.weight"], d["fc1.weight"]):
        failures.append("self-test bug: nextafter did not change the tensor")
    r = compare_checkpoint(a, d)
    if r["identical"]:
        failures.append("a one-ULP change was NOT caught (tolerance is not 0.0)")
    elif r["per_tensor"]["fc1.weight"]["group"] != "fc":
        failures.append("perturbed fc tensor was not grouped as fc")

    # Key mismatch must raise, not silently pass.
    e = _synthetic(0)
    del e["fc1.weight"]
    try:
        compare_checkpoint(a, e)
        failures.append("key-set mismatch did not raise")
    except ValueError:
        pass

    # Shape mismatch must raise.
    f = _synthetic(0)
    f["fc1.weight"] = torch.randn(9, 4)
    try:
        compare_checkpoint(a, f)
        failures.append("shape mismatch did not raise")
    except ValueError:
        pass

    # Group classification over all four real naming schemes (Phase 0/F).
    expect = {
        "conv1.0.weight": "conv", "conv1.0.W": "conv", "conv1.0.b": "conv",
        "conv1.0.B_feedback": "conv", "W1.weight": "conv", "P2.weight": "conv",
        "conv1.weight": "conv",
        "conv1.1.weight": "bn", "conv1.1.bias": "bn", "bn1.weight": "bn",
        "bn1.running_mean": "bn", "bn1.running_var": "bn",
        "conv1.1.num_batches_tracked": "bn",
        "fc1.weight": "fc", "fc2.bias": "fc",
    }
    for k, want in expect.items():
        got = _tensor_group(k)
        if got != want:
            failures.append(f"_tensor_group({k!r}) = {got!r}, expected {want!r}")

    # GateResult classification must separate the two mismatch meanings.
    res = GateResult()
    res.per_rule = {
        "Random Weights": {"identical": False, "n_tensors": 1,
                           "n_tensors_differing": 1, "groups_affected": {},
                           "per_tensor": {}},
        "Backprop": {"identical": False, "n_tensors": 1,
                     "n_tensors_differing": 1, "groups_affected": {},
                     "per_tensor": {}},
    }
    if res.contradicts_documentation != ["Random Weights"]:
        failures.append("contradicts_documentation misclassified")
    if res.consistent_with_documentation != ["Backprop"]:
        failures.append("consistent_with_documentation misclassified")
    if res.all_identical:
        failures.append("all_identical true despite mismatches")

    # An empty result must not read as a pass.
    if GateResult().all_identical:
        failures.append("empty GateResult reported all_identical (vacuous pass)")
    if GateResult().is_complete:
        failures.append("empty GateResult reported is_complete (vacuous pass)")

    # A PARTIAL run -- some rules compared and identical, the rest not compared --
    # must be neither a pass nor a mismatch. Getting this wrong once already
    # printed "BRANCH 2: NOT IDENTICAL" for a run in which nothing mismatched.
    partial = GateResult()
    partial.per_rule = {"Random Weights": {"identical": True, "n_tensors": 22,
                                           "n_tensors_differing": 0,
                                           "groups_affected": {}, "per_tensor": {}}}
    partial.errors = ["Backprop: candidate missing"]
    if not partial.all_identical:
        failures.append("partial run: compared rule was identical but all_identical False")
    if partial.is_complete:
        failures.append("partial run reported is_complete")
    if partial.any_mismatch:
        failures.append("partial run reported a mismatch where there was none")

    # A complete, all-identical run must be complete.
    full = GateResult()
    full.per_rule = {r: {"identical": True, "n_tensors": 1, "n_tensors_differing": 0,
                         "groups_affected": {}, "per_tensor": {}} for r in RULES}
    if not (full.all_identical and full.is_complete and not full.any_mismatch):
        failures.append("complete all-identical run misclassified")

    if N_CHECKPOINTS_EXPECTED != 30:
        failures.append(f"expected 30 checkpoints, constant says {N_CHECKPOINTS_EXPECTED}")

    print("determinism_gate self-test")
    if failures:
        for f_ in failures:
            print(f"  FAIL  {f_}")
        print(f"\n{len(failures)} failure(s)")
        return 1
    print(f"  all checks passed (both branches exercised; "
          f"{len(expect)} key-naming cases)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--reference", type=Path, help="dir with the six existing checkpoints")
    p.add_argument("--candidate", type=Path, help="dir with the retrained checkpoints")
    p.add_argument("--seed-idx", type=int, default=0)
    p.add_argument("--json", type=Path, help="also write the verdict as JSON")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()

    if a.self_test:
        return self_test()
    if not a.reference or not a.candidate:
        p.error("--reference and --candidate are required (or use --self-test)")

    res = run_gate(a.reference, a.candidate, a.seed_idx)
    code = report(res)
    if a.json:
        a.json.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
        print(f"\nverdict -> {a.json}")
    return code


if __name__ == "__main__":
    sys.exit(main())
