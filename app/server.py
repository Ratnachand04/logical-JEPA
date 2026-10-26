"""Flask inference server for the Logical-JEPA web demo.

Serves a static HTML/CSS/JS frontend and a small JSON API. The model, sweep bank
and scorer are the *same* objects ``evaluate.py`` uses, so a verdict shown in
the browser is produced by exactly the pipeline the reported metrics measure.

Usage::

    py -3.12 app/server.py --checkpoint checkpoints/<name>/<category>/final.pt
    py -3.12 app/server.py --checkpoint <path> --port 8080

Endpoints
---------
``GET  /``                 the single-page frontend
``GET  /api/status``       model / calibration / device info
``GET  /api/samples``      test images available for one-click inspection
``GET  /api/sample-image`` the raw bytes of one of those samples
``POST /api/predict``      score an uploaded image or a named sample
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402
from PIL import Image  # noqa: E402

from anomaly.scoring import Calibration, build_scorer  # noqa: E402
from datasets.mvtec_loco import MVTecLOCO  # noqa: E402
from datasets.transforms import build_eval_transform, to_numpy_image  # noqa: E402
from evaluate import build_sweep_bank, load_model_from_checkpoint  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.logging_utils import get_logger  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_UPLOAD_MB = 16

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

logger = get_logger("server")

# Populated by main(); a single shared inference context.
STATE: dict = {
    "model": None,
    "scorer": None,
    "cfg": None,
    "device": None,
    "transform": None,
    "test_set": None,
    "checkpoint": None,
    "calibrated": False,
}


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def encode_png(array: np.ndarray) -> str:
    """RGB or grayscale uint8 array -> base64 data URI."""
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def colorize(heat: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Anomaly map -> RGB uint8 using a perceptually ordered colormap.

    ``lo``/``hi`` come from the calibration so heatmaps are comparable between
    images: without a fixed range, a perfectly normal image is stretched to full
    saturation and looks as alarming as a defective one.
    """
    lo = float(heat.min()) if lo is None else lo
    hi = float(heat.max()) if hi is None else hi

    norm = np.clip((heat - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    coloured = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def blend(image: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return (image * (1 - alpha) + heat_rgb * alpha).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_inference(pil_image: Image.Image) -> dict:
    """Score one PIL image and build every visual the frontend shows."""
    started = time.perf_counter()

    scorer = STATE["scorer"]
    device = STATE["device"]
    calib = scorer.calibration

    tensor = STATE["transform"](pil_image.convert("RGB")).unsqueeze(0).to(device)
    out = scorer.predict(tensor)

    heat = out["maps"][0, 0].float().cpu().numpy()
    rgb = to_numpy_image(tensor[0])

    heat_rgb = colorize(heat, calib.map_lo, calib.map_hi)
    overlay = blend(rgb, heat_rgb, 0.5)

    score = float(out["score"][0])
    z = float(out["z_score"][0])
    is_anom = bool(out["is_anomalous"][0])

    # Per-scale grids, upsampled for display. These are what let a viewer see
    # *why* the verdict came out the way it did: a hit only at the large scale
    # is the signature of a logical anomaly.
    scales = {}
    for window, grid in out["grids"].items():
        g = grid[0].float().cpu().numpy()
        big = cv2.resize(g, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        scales[str(window)] = {
            "image": encode_png(colorize(big)),
            "mean": float(g.mean()),
            "max": float(g.max()),
            "role": "structural (local defects)" if window <= 3
                    else "logical (component layout)",
        }

    # Cardinality channel (Phase 3c), when the scorer uses it: one card, the
    # mismatch between expected and observed component mass, averaged over scales.
    cardinality = None
    if out.get("card_grids"):
        g = torch.stack([grid[0].float() for grid in out["card_grids"].values()]).mean(0)
        g = g.cpu().numpy()
        big = cv2.resize(g, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        cardinality = {
            "image": encode_png(colorize(big)),
            "mean": float(g.mean()),
            "max": float(g.max()),
            "role": "cardinality (expected vs observed component mass)",
            "mode": scorer.cardinality,
        }

    # Peak location, in pixels, for the "where" readout.
    peak = np.unravel_index(int(np.argmax(heat)), heat.shape)

    return {
        "verdict": "ANOMALOUS" if is_anom else "NORMAL",
        "is_anomalous": is_anom,
        "score": score,
        "z_score": z,
        "confidence": float(out["confidence"][0]),
        "threshold": float(calib.threshold),
        "calibrated": STATE["calibrated"],
        "peak": {"y": int(peak[0]), "x": int(peak[1])},
        "images": {
            "input": encode_png(rgb),
            "heatmap": encode_png(heat_rgb),
            "overlay": encode_png(overlay),
        },
        "scales": scales,
        "cardinality": cardinality,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/status")
def api_status():
    """Model metadata shown in the header, plus calibration state."""
    model = STATE["model"]
    cfg = STATE["cfg"]
    scorer = STATE["scorer"]

    if model is None:
        # Compare-only mode: no category model, but the reference model may be up.
        if REFERENCE["scorer"] is not None:
            return jsonify({"ready": False, "error": "No category model loaded",
                            "reference": reference_info()})
        return jsonify({"ready": False, "error": "No model loaded"}), 503

    encoder = model.context_encoder.num_parameters()
    predictor = model.predictor.num_parameters()
    trainable = sum(p.numel() for p in model.trainable_parameters())
    train_images = cfg.get_path("data.num_train_images", None)

    return jsonify({
        "ready": True,
        "checkpoint": STATE["checkpoint"],
        "category": cfg.get_path("data.category"),
        "device": str(STATE["device"]),
        "calibrated": STATE["calibrated"],
        "model": {
            "grid": f"{model.grid_size}x{model.grid_size}",
            "img_size": model.img_size,
            "patch_size": model.patch_size,
            "embed_dim": model.embed_dim,
            "encoder_params": encoder,
            "predictor_params": predictor,
            "trainable_params": trainable,
            # Few-shot models: where the encoder started and what was adapted.
            "pretrained": model.pretrained,
            "finetune": model.finetune,
            "train_images": train_images,
        },
        "masking": {
            "strategy": cfg.get_path("masking.strategy"),
            "mode": cfg.get_path("masking.mode"),
        },
        "inference": {
            "sweep_windows": list(scorer.mask_bank.scales()),
            "sweep_configs": len(scorer.mask_bank),
            "distance": scorer.distance or model.loss_kind,
            "fusion": scorer.fusion,
            "aggregation": scorer.aggregation,
            "deviation": scorer.deviation,
            "cardinality": scorer.cardinality,
            "memory": scorer.memory,
            "memory_images": len(scorer.memory_paths),
        },
        "calibration": scorer.calibration.to_dict(),
        "reference": reference_info(),
    })


@app.route("/api/samples")
def api_samples():
    """Test images grouped by defect type, for one-click inspection."""
    test_set = STATE["test_set"]
    if test_set is None:
        return jsonify({"samples": [], "note": "no test split available"})

    limit = int(request.args.get("limit", 8))
    grouped: dict[str, list] = {}

    for i, sample in enumerate(test_set.samples):
        bucket = grouped.setdefault(sample.defect_type, [])
        if len(bucket) < limit:
            bucket.append({
                "index": i,
                "name": os.path.basename(sample.image_path),
                "defect_type": sample.defect_type,
                "label": sample.label,
            })

    return jsonify({"category": test_set.category, "groups": grouped})


@app.route("/api/sample-image")
def api_sample_image():
    """Raw bytes of one test image, for the sample thumbnails."""
    test_set = STATE["test_set"]
    if test_set is None:
        return jsonify({"error": "no test split available"}), 404

    try:
        index = int(request.args.get("index", ""))
        sample = test_set.samples[index]
    except (ValueError, IndexError):
        return jsonify({"error": "invalid sample index"}), 400

    # send_file resolves a relative path against the Flask app root (app/), not
    # the working directory the server was launched from, so absolutise it.
    return send_file(os.path.abspath(sample.image_path), mimetype="image/png")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Score an uploaded file, a base64 data URI, or a test-set sample index."""
    if STATE["scorer"] is None:
        return jsonify({"error": "No model loaded"}), 503

    try:
        image = None
        ground_truth = None

        if "file" in request.files:
            upload = request.files["file"]
            if not upload.filename:
                return jsonify({"error": "Empty filename"}), 400
            image = Image.open(io.BytesIO(upload.read()))

        else:
            payload = request.get_json(silent=True) or {}

            if "index" in payload and STATE["test_set"] is not None:
                sample = STATE["test_set"].samples[int(payload["index"])]
                image = Image.open(sample.image_path)
                ground_truth = sample.defect_type

            elif "image" in payload:
                raw = payload["image"]
                if "," in raw:
                    raw = raw.split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(raw)))

        if image is None:
            return jsonify({"error": "No image supplied"}), 400

        result = run_inference(image)
        if ground_truth is not None:
            result["ground_truth"] = ground_truth
            result["correct"] = (ground_truth != "good") == result["is_anomalous"]

        return jsonify(result)

    except Exception as exc:  # noqa: BLE001 - surface the message to the UI
        logger.exception("prediction failed")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": f"Image exceeds the {MAX_UPLOAD_MB} MB limit"}), 413


# --------------------------------------------------------------------------- #
def load_state(checkpoint: str, overrides: list[str], calibrate: bool) -> None:
    """Build the shared inference context once at startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    override_cfg = Config()
    for item in overrides:
        import yaml
        key, _, value = item.partition("=")
        override_cfg.set_path(key.strip(), yaml.safe_load(value.strip()))

    model, cfg, payload = load_model_from_checkpoint(checkpoint, device, override_cfg)
    bank = build_sweep_bank(cfg, model, device)
    scorer = build_scorer(cfg, model, bank)

    category = cfg.get_path("data.category")
    root = cfg.get_path("data.root", "data/mvtec_loco")
    img_size = cfg.get_path("data.img_size", 256)

    # Prefer the calibration (and memory bank) saved by evaluate.py; fall back
    # to fitting them so the demo still gives a meaningful verdict.
    calib_path = os.path.join(os.path.dirname(checkpoint), "calibration.json")
    calibrated = False

    bank_path = os.path.join(os.path.dirname(checkpoint), "memory_bank.pt")

    if os.path.isfile(calib_path):
        with open(calib_path, encoding="utf-8") as handle:
            scorer.calibration = Calibration.from_dict(json.load(handle))
        calibrated = True
        logger.info(f"Loaded calibration from {calib_path}")
        if scorer.memory != "only" and not scorer.calibration.scale_stats:
            # Fitted by a memory-only evaluation: it has no JEPA statistics.
            calibrated = False
            logger.warning("The saved calibration has no JEPA statistics; refitting")
        if calibrated and scorer.uses_memory:
            # A calibration fitted without the memory channel cannot score with it.
            if os.path.isfile(bank_path) and scorer.calibration.mem_stats:
                scorer.load_memory_bank(torch.load(bank_path, weights_only=True), device)
                logger.info(f"Loaded memory bank from {bank_path}")
            else:
                calibrated = False
                logger.warning("anomaly.memory is on but the saved calibration has no "
                               "memory bank; refitting")
    if not calibrated and calibrate:
        try:
            from datasets.mvtec_loco import build_dataloaders, build_normal_loader

            train_loader, val_loader, _ = build_dataloaders(cfg, category, num_workers=0)
            # Statistics (and the memory bank) from the training normals --
            # in a few-shot run, only the K training images -- as evaluate.py does.
            scorer.calibrate(val_loader or train_loader, device,
                             cfg.get_path("eval.sigma_threshold", 3.0),
                             stats_loader=build_normal_loader(cfg, category, "train",
                                                              num_workers=0))
            calibrated = True
            logger.info(f"Fitted calibration on {scorer.calibration.n_samples} normal images")
        except Exception as exc:  # noqa: BLE001
            if scorer.uses_memory:
                raise SystemExit(f"Could not fit the memory bank ({exc}).") from exc
            logger.warning(f"Could not calibrate ({exc}); verdicts will be uncalibrated")
    elif not calibrated:
        if scorer.uses_memory:
            raise SystemExit("anomaly.memory is on but no memory bank is available. "
                             "Run evaluate.py with this config first, or start with --calibrate.")
        logger.warning("No calibration available -- run evaluate.py for meaningful verdicts")

    try:
        test_set = MVTecLOCO(root, category, "test", img_size)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning(f"No test split for samples panel ({exc})")
        test_set = None

    STATE.update({
        "model": model, "scorer": scorer, "cfg": cfg, "device": device,
        "transform": build_eval_transform(img_size),
        "test_set": test_set, "checkpoint": checkpoint, "calibrated": calibrated,
    })

    logger.info(model.describe())
    logger.info(f"Sweep: {bank.describe()}  device={device}")
    logger.info(f"Checkpoint epoch {payload.get('epoch', '?')}, category '{category}'")


# --------------------------------------------------------------------------- #
# Reference comparison (any product: a correct part vs the part to check)
# --------------------------------------------------------------------------- #
REFERENCE: dict = {"scorer": None, "payload": None, "checkpoint": None, "cache": {}}
REFERENCE_CACHE_SIZE = 4


def reference_info() -> dict:
    scorer = REFERENCE["scorer"]
    if scorer is None:
        return {"available": False}
    payload = REFERENCE["payload"]
    threshold = scorer.calibration.get("threshold", float("inf"))
    return {
        "available": True,
        "checkpoint": REFERENCE["checkpoint"],
        "backbone": scorer.model.backbone,
        "products_trained": len(payload.get("products", [])),
        "holdout": payload.get("holdout", []),
        "predictor_params": scorer.model.predictor.num_parameters(),
        "windows": list(scorer.bank.scales()),
        "calibrated": bool(np.isfinite(threshold)),
        "threshold": float(threshold) if np.isfinite(threshold) else None,
    }


def _prepared_references(files) -> object:
    """Encode references once per distinct set of uploads (small LRU cache)."""
    import hashlib

    blobs = [f.read() for f in files]
    key = hashlib.sha1(b"".join(hashlib.sha1(b).digest() for b in blobs)).hexdigest()
    cache = REFERENCE["cache"]
    if key in cache:
        cache[key] = cache.pop(key)          # mark as most recently used
        return cache[key]
    images = [Image.open(io.BytesIO(b)) for b in blobs]
    state = REFERENCE["scorer"].prepare(images)
    cache[key] = state
    while len(cache) > REFERENCE_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    return state


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Compare an uploaded current image with one or more reference images."""
    scorer = REFERENCE["scorer"]
    if scorer is None:
        return jsonify({"error": "No reference model loaded (start the server with "
                                 "--reference-checkpoint)"}), 503
    refs = [f for f in request.files.getlist("reference") if f and f.filename]
    current = request.files.get("current")
    if not refs or current is None or not current.filename:
        return jsonify({"error": "Upload at least one reference image and one current image"}), 400
    if len(refs) > 4:
        refs = refs[:4]
    try:
        started = time.perf_counter()
        try:
            sensitivity = float(request.form.get("sensitivity", 1.0))
        except ValueError:
            sensitivity = 1.0
        sensitivity = min(max(sensitivity, 0.25), 4.0)
        state = _prepared_references(refs)
        out = scorer.compare(Image.open(current), state, sensitivity=sensitivity)
        heat = out["map"]
        threshold = out["threshold"]
        calibrated = bool(np.isfinite(threshold))
        # Colour scale: saturate where the map reaches this product's decision
        # level, so the hottest colour means "would be flagged on its own".
        level = scorer.normal_level(state)
        hi = threshold * level if (calibrated and out["rule"] == "ratio") else (
            threshold if calibrated else float(heat.max()))
        lo = min(float(heat.min()), 0.0)
        heat_rgb = colorize(heat, lo, max(hi, lo + 1e-6))
        overlay = blend(out["aligned"], heat_rgb, 0.5)
        peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
        anomalous = bool(out["is_anomalous"])
        return jsonify({
            "verdict": ("DIFFERENT" if anomalous else "MATCHES") if calibrated else "SCORED",
            "is_anomalous": anomalous,
            "calibrated": calibrated,
            "score": float(out["score"]),
            "relative_score": float(out["relative_score"]),
            "decision_value": float(out["decision_value"]),
            "rule": out["rule"],
            "sensitivity": sensitivity,
            "threshold": float(threshold) if calibrated else None,
            "alignment": out["alignment"],
            "n_references": len(state.images),
            "peak": {"y": int(peak[0]), "x": int(peak[1])},
            "images": {
                "reference": encode_png(state.images[0]),
                "current": encode_png(out["aligned"]),
                "heatmap": encode_png(heat_rgb),
                "overlay": encode_png(overlay),
            },
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        })
    except Exception as exc:  # noqa: BLE001 - surface the message to the UI
        logger.exception("comparison failed")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def load_reference_state(checkpoint: str) -> None:
    """Load the reference-conditioned model and its calibration."""
    from anomaly.reference_scoring import ReferenceScorer, load_calibration
    from train_reference import load_reference_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, payload = load_reference_model(checkpoint, device)
    node = cfg.get("scoring", {})
    calibration = load_calibration(checkpoint)
    REFERENCE.update(
        scorer=ReferenceScorer(model, device, windows=tuple(node.get("windows", [4])),
                               sigma=node.get("smooth_sigma", 4.0),
                               top_k_ratio=node.get("top_k_ratio", 0.01),
                               n_pseudo=node.get("n_pseudo", 8), align=node.get("align", True),
                               calibration=calibration),
        payload=payload, checkpoint=checkpoint, cache={})
    logger.info(model.describe())
    if not np.isfinite(calibration["threshold"]):
        logger.warning("No reference_calibration.json -- comparisons are scored but get no "
                       "verdict. Run scripts/eval_reference.py --calibrate.")
    logger.info(f"Reference model trained on {len(payload.get('products', []))} products; "
                f"held out: {payload.get('holdout') or 'none'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logical-JEPA web demo server")
    parser.add_argument("--checkpoint", default=None,
                        help="category model (single-image inspection)")
    parser.add_argument("--reference-checkpoint", default=None,
                        help="reference-conditioned model (compare a part with a reference)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="skip fitting a calibration when none is saved")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint and not args.reference_checkpoint:
        raise SystemExit("Give --checkpoint, --reference-checkpoint, or both.")
    if args.checkpoint:
        load_state(args.checkpoint, args.set, calibrate=not args.no_calibrate)
    if args.reference_checkpoint:
        load_reference_state(args.reference_checkpoint)

    logger.info(f"Serving Logical-JEPA at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
