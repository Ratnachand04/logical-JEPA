"""Measure inference latency across resolution and sweep settings.

Everything here is *measured*, never projected. The sweep dominates inference
cost and it scales with the number of window positions, which itself grows with
the patch grid -- so a resolution change moves latency by much more than the
encoder FLOPs alone suggest, and the only honest way to report it is to run it.

Phase 5 targets <100 ms/image from a 258 ms baseline. This script produces the
table that says whether that was hit, and which knob got it there.

Usage::

    py -3.12 scripts/benchmark_latency.py --checkpoint <path>
    py -3.12 scripts/benchmark_latency.py --checkpoint <path> --resolutions 256,384
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from masking.sweep_mask import SweepMaskBank  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.logging_utils import get_logger, save_json  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_sweep(model, bank, device, batch_size=1, chunk=16, repeats=5,
               warmup=2, amp=False) -> dict:
    """Median-of-repeats latency for one full sweep.

    Median rather than mean: the first timed run after a configuration change
    often catches a kernel autotune, and one such outlier would dominate a mean.
    """
    images = torch.randn(batch_size, 3, model.img_size, model.img_size, device=device)

    def once():
        if amp and device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast("cuda", dtype=dtype):
                model.anomaly_grids(images, bank, chunk=chunk)
        else:
            model.anomaly_grids(images, bank, chunk=chunk)

    for _ in range(warmup):
        once()
    _sync(device)

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        once()
        _sync(device)
        timings.append(time.perf_counter() - start)

    timings.sort()
    total = timings[len(timings) // 2]

    return {
        "total_ms": round(total * 1000, 1),
        "per_image_ms": round(total / batch_size * 1000, 1),
        "batch_size": batch_size,
        "chunk": chunk,
        "amp": amp,
        "sweep_configs": len(bank),
    }


def build_at_resolution(cfg, img_size: int, device):
    """A freshly initialised model at a given resolution.

    Weights are random: this measures *cost*, which depends on shape and not on
    what the weights contain. Loading a 256px checkpoint into a 512px model
    would fail on the position table anyway.
    """
    resized = Config(cfg.to_dict())
    resized.set_path("data.img_size", img_size)
    return build_model(resized).to(device).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Logical-JEPA inference")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint supplying the architecture; a default "
                             "config is used when omitted")
    parser.add_argument("--config", default="configs/loco.yaml")
    parser.add_argument("--resolutions", default="256,384,512")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="outputs/phase5_latency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out, exist_ok=True)
    logger = get_logger("benchmark", os.path.join(args.out, "latency.log"))

    if args.checkpoint and os.path.isfile(args.checkpoint):
        from evaluate import load_model_from_checkpoint
        _model, cfg, _ = load_model_from_checkpoint(args.checkpoint, device)
        logger.info(f"Architecture from {args.checkpoint}")
    else:
        from utils.config import load_config
        cfg = load_config(args.config)
        logger.info(f"Architecture from {args.config}")

    if device.type == "cuda":
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("Running on CPU -- latency will not be representative")

    resolutions = [int(r) for r in args.resolutions.replace(",", " ").split()]
    batch_sizes = [int(b) for b in args.batch_sizes.replace(",", " ").split()]
    base_windows = cfg.get_path("anomaly.sweep_windows", [2, 4, 6])

    rows = []

    for img_size in resolutions:
        model = build_at_resolution(cfg, img_size, device)
        grid = model.grid_size
        logger.info(f"\n{'='*66}\nresolution {img_size} -> {grid}x{grid} grid "
                    f"({grid*grid} tokens)\n{'='*66}")

        # Scale the sweep windows with the grid so the *physical* window size
        # stays constant; otherwise a bigger grid silently means finer windows
        # and the comparison measures two changes at once.
        scale = grid / 16
        windows = sorted({max(2, int(round(w * scale))) for w in base_windows})

        configs = [
            ("full sweep", windows, None),
            ("single scale", [windows[len(windows) // 2]], None),
            ("single scale, coarse stride", [windows[len(windows) // 2]],
             [max(2, windows[len(windows) // 2])]),
        ]

        for label, win, strides in configs:
            bank = SweepMaskBank(grid, windows=win, strides=strides, device=device)

            for batch_size in batch_sizes:
                for amp in (False, True) if device.type == "cuda" else (False,):
                    try:
                        result = time_sweep(
                            model, bank, device, batch_size=batch_size,
                            chunk=args.chunk, repeats=args.repeats, amp=amp,
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"  OOM: {label} bs={batch_size} amp={amp}")
                        torch.cuda.empty_cache()
                        continue

                    result.update(img_size=img_size, grid=grid, windows=win,
                                  config=label)
                    rows.append(result)
                    logger.info(
                        f"  {label:<28s} bs={batch_size} amp={str(amp):<5s} "
                        f"windows={win} configs={result['sweep_configs']:>4d}  "
                        f"{result['per_image_ms']:>7.1f} ms/image"
                    )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- summary ------------------------------------------------------ #
    logger.info("\n" + "=" * 66 + "\nSUMMARY (ms per image)\n" + "=" * 66)
    logger.info(f"{'res':>5s} {'config':<28s} {'bs':>3s} {'amp':>5s} "
                f"{'cfgs':>5s} {'ms/img':>8s}")
    for row in rows:
        logger.info(
            f"{row['img_size']:5d} {row['config']:<28s} {row['batch_size']:3d} "
            f"{str(row['amp']):>5s} {row['sweep_configs']:5d} "
            f"{row['per_image_ms']:8.1f}"
        )

    under_target = [r for r in rows if r["per_image_ms"] < 100.0]
    logger.info(f"\nConfigurations under the 100 ms/image target: "
                f"{len(under_target)}/{len(rows)}")
    if under_target:
        best = min(under_target, key=lambda r: r["per_image_ms"])
        logger.info(
            f"  fastest: {best['per_image_ms']} ms/image "
            f"({best['config']}, {best['img_size']}px, bs={best['batch_size']}, "
            f"amp={best['amp']})"
        )
    else:
        fastest = min(rows, key=lambda r: r["per_image_ms"])
        logger.info(
            f"  TARGET MISSED -- fastest measured is {fastest['per_image_ms']} "
            f"ms/image ({fastest['config']}, {fastest['img_size']}px)"
        )

    save_json({"rows": rows, "device": str(device)},
              os.path.join(args.out, "latency.json"))
    logger.info(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
