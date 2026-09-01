"""Regenerate every reported number from a clean checkout.

One entry point for the whole result set, so a reader can check any figure in
the README without reconstructing the command that produced it. Each stage
prints the command it runs before running it, and the script stops at the first
failure rather than carrying on and producing a partial table that looks whole.

Stages
------
``verify``      dataset structure and ground-truth well-formedness
``tests``       the unit suite (fast, and gates everything after it)
``train``       the headline model
``evaluate``    detection and localization metrics
``ablations``   all six studies, multi-seed
``slots``       the Phase 3a isolation gate
``latency``     the Phase 5 measurements
``figures``     the qualitative figures

Usage::

    py -3.12 scripts/reproduce_all.py --root data/mvtec_loco_256 --seeds 0,1,2
    py -3.12 scripts/reproduce_all.py --stages verify,tests --dry-run
    py -3.12 scripts/reproduce_all.py --quick        # smoke-sized, for CI
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

PY = [sys.executable]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_STAGES = ["verify", "tests", "train", "evaluate", "ablations",
              "slots", "latency", "figures"]


def run(cmd: list[str], label: str, dry_run: bool) -> float:
    """Run one stage, echoing the exact command first."""
    printable = " ".join(str(c) for c in cmd)
    print(f"\n{'=' * 78}\n[{label}]\n  {printable}\n{'=' * 78}", flush=True)

    if dry_run:
        print("  (dry run -- not executed)")
        return 0.0

    started = time.time()
    result = subprocess.run(cmd, cwd=REPO)
    elapsed = time.time() - started

    if result.returncode != 0:
        raise SystemExit(
            f"\nSTAGE FAILED: {label} (exit {result.returncode}).\n"
            f"Stopping here -- a partial result set is worse than none, because "
            f"it looks complete."
        )

    print(f"  [{label}] ok in {elapsed/60:.1f} min", flush=True)
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate every reported number")
    parser.add_argument("--root", default="data/mvtec_loco_256",
                        help="dataset root (use the pre-resized cache for speed)")
    parser.add_argument("--category", default="pushpins")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--name", default="reproduce")
    parser.add_argument("--stages", default="all",
                        help=f"comma-separated subset of {ALL_STAGES}, or 'all'")
    parser.add_argument("--quick", action="store_true",
                        help="tiny schedule for a smoke check, not for reporting")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stages = (ALL_STAGES if args.stages == "all"
              else [s.strip() for s in args.stages.split(",") if s.strip()])
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {ALL_STAGES}")

    epochs = ["train.epochs=3", "train.warmup_epochs=1"] if args.quick else []
    seeds = "0" if args.quick else args.seeds
    ckpt = os.path.join("checkpoints", args.name, args.category, "final.pt")

    common = [f"data.root={args.root}", f"data.category={args.category}",
              "data.num_workers=4"]

    print(f"Reproducing Logical-JEPA results")
    print(f"  root={args.root}  category={args.category}  seeds={seeds}")
    print(f"  stages: {', '.join(stages)}")
    if args.quick:
        print("  QUICK MODE -- results are smoke checks, not reportable numbers")

    timings: dict[str, float] = {}

    if "verify" in stages:
        cmd = PY + ["scripts/verify_dataset.py", "--root", args.root]
        # Only demand all five categories when pointed at a real LOCO tree.
        if "loco" in args.root and "synth" not in args.root:
            cmd.append("--strict-loco")
        timings["verify"] = run(cmd, "verify dataset", args.dry_run)

    if "tests" in stages:
        timings["tests"] = run(PY + ["-m", "pytest", "tests/", "-q"],
                               "unit tests", args.dry_run)

    if "train" in stages:
        timings["train"] = run(
            PY + ["train.py", "--config", "configs/loco.yaml", "--name", args.name,
                  "--set", *common, *epochs],
            "train headline model", args.dry_run)

    if "evaluate" in stages:
        timings["evaluate"] = run(
            PY + ["evaluate.py", "--checkpoint", ckpt,
                  "--config", "configs/loco.yaml", "--set", *common],
            "evaluate (detection + localization)", args.dry_run)

    if "ablations" in stages:
        timings["ablations"] = run(
            PY + ["run_ablations.py", "--config", "configs/ablations.yaml",
                  "--study", "all", "--seeds", seeds,
                  "--set", f"experiment.name={args.name}_ablations", *common,
                  "train.save_interval=0", *epochs],
            "all ablation studies", args.dry_run)

    if "slots" in stages:
        timings["slots"] = run(
            PY + ["train_slots.py", "--checkpoint", ckpt,
                  "--num-slots", "4,6,8,12",
                  "--epochs", "10" if args.quick else "150",
                  "--set", *common],
            "slot bottleneck isolation gate", args.dry_run)

    if "latency" in stages:
        timings["latency"] = run(
            PY + ["scripts/benchmark_latency.py", "--checkpoint", ckpt,
                  "--resolutions", "256,384", "--batch-sizes", "1,4"],
            "latency benchmark", args.dry_run)

    if "figures" in stages:
        timings["figures"] = run(
            PY + ["visualize.py", "--checkpoint", ckpt, "--mode", "all",
                  "--config", "configs/loco.yaml", "--set", *common],
            "qualitative figures", args.dry_run)

    print("\n" + "=" * 78)
    print("REPRODUCTION COMPLETE")
    print("=" * 78)
    for stage, seconds in timings.items():
        print(f"  {stage:<34s} {seconds/60:6.1f} min")
    print(f"  {'TOTAL':<34s} {sum(timings.values())/60:6.1f} min")
    print("\nResults under outputs/ ; per-phase findings in outputs/*/SUMMARY.md")


if __name__ == "__main__":
    main()
