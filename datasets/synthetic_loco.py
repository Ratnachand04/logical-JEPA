"""A synthetic stand-in for MVTec LOCO AD.

MVTec LOCO AD requires manual download and registration, so this module
generates a small dataset with the *same directory layout, the same anomaly
taxonomy and the same ground-truth format*. Everything downstream -- loader,
training, sweep, metrics, ablations, web demo -- runs unchanged on either one.

The synthetic category is a ``screw_board``: a product body on a board with four
screws at fixed mounting points, plus a colour-coded indicator strip. That is the
minimum structure that admits genuinely *logical* anomalies:

===========================  ==========================================
``missing_screw``            a mounting point is empty
``extra_screw``              a screw where none belongs
``misplaced_screw``          right count, wrong location
``wrong_count``              two mounting points emptied
``swapped_indicator``        indicator strip has the wrong colour order
===========================  ==========================================

None of these change local texture: every pixel still looks like a legitimate
screw or board. They are only detectable by reasoning about the arrangement,
which is exactly the regime this project targets. Structural anomalies
(``scratch``, ``contamination``, ``chip``) are generated too, so the
structural-vs-logical split can be measured.

This is a *development* dataset. Reported results must come from real MVTec
LOCO AD; the README says so explicitly.
"""

from __future__ import annotations

import json
import os
import random

import cv2
import numpy as np

CATEGORY = "screw_board"

IMG_SIZE = 320                     # generated at 320, resized to 256 by the loader
BOARD_COLOR = (78, 84, 92)
PRODUCT_COLOR = (150, 148, 140)
SCREW_COLOR = (196, 198, 205)
INDICATOR_COLORS = [(60, 180, 75), (255, 190, 40), (40, 90, 220)]

# The four canonical mounting points, in fractions of the image size.
MOUNTS = [(0.20, 0.20), (0.80, 0.20), (0.20, 0.80), (0.80, 0.80)]

LOGICAL_TYPES = ("missing_screw", "extra_screw", "misplaced_screw",
                 "wrong_count", "swapped_indicator")
STRUCTURAL_TYPES = ("scratch", "contamination", "chip")

# Which way each defect moves the amount of content in its region. Phase 1's
# below-chance logical result was attributed to *removal* (an emptied region is
# easier to predict), so the two directions are reported separately.
DEFECT_DIRECTION = {
    "missing_screw": "removal",
    "wrong_count": "removal",
    "extra_screw": "addition",
    "misplaced_screw": "rearrangement",
    "swapped_indicator": "rearrangement",
    "scratch": "structural",
    "contamination": "structural",
    "chip": "structural",
}

# Written beside the category so evaluation can break results down by subtype.
SUBTYPE_MANIFEST = "defect_subtypes.json"


# --------------------------------------------------------------------------- #
# Primitive drawing
# --------------------------------------------------------------------------- #
def _noise(img: np.ndarray, rng: random.Random, sigma: float = 4.0) -> np.ndarray:
    """Sensor noise plus a mild global lighting shift.

    Both are nuisance factors a real acquisition has, and both must be learned
    as *normal* so the model does not flag brightness changes as anomalies.
    """
    gain = 1.0 + rng.uniform(-0.06, 0.06)
    out = img.astype(np.float32) * gain
    out += np.random.normal(0.0, sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_screw(img: np.ndarray, cx: int, cy: int, radius: int, rng: random.Random) -> None:
    """A screw head: bright disc, darker rim, cross slot at a random angle."""
    cv2.circle(img, (cx, cy), radius, SCREW_COLOR, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, (120, 122, 128), 2, lineType=cv2.LINE_AA)

    angle = rng.uniform(0, np.pi)
    for theta in (angle, angle + np.pi / 2):
        dx, dy = int(np.cos(theta) * radius * 0.7), int(np.sin(theta) * radius * 0.7)
        cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy),
                 (95, 97, 103), 2, lineType=cv2.LINE_AA)


def _draw_base(rng: random.Random) -> np.ndarray:
    """Board + product body + indicator strip, with small fixture jitter."""
    img = np.full((IMG_SIZE, IMG_SIZE, 3), BOARD_COLOR, dtype=np.uint8)

    # Faint board grain so the model cannot rely on a perfectly flat background.
    grain = np.random.normal(0, 3, (IMG_SIZE, IMG_SIZE, 1)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    jx, jy = rng.randint(-3, 3), rng.randint(-3, 3)
    x0, y0 = int(IMG_SIZE * 0.32) + jx, int(IMG_SIZE * 0.34) + jy
    x1, y1 = int(IMG_SIZE * 0.68) + jx, int(IMG_SIZE * 0.66) + jy

    cv2.rectangle(img, (x0, y0), (x1, y1), PRODUCT_COLOR, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (100, 99, 94), 2)

    return img


def _draw_indicator(img: np.ndarray, order: list[int], rng: random.Random) -> None:
    """Three coloured segments below the product; order is part of the spec."""
    y0 = int(IMG_SIZE * 0.70)
    y1 = y0 + int(IMG_SIZE * 0.05)
    x0 = int(IMG_SIZE * 0.34)
    seg = int(IMG_SIZE * 0.32 / 3)

    for slot, colour_id in enumerate(order):
        sx = x0 + slot * seg
        cv2.rectangle(img, (sx, y0), (sx + seg - 3, y1), INDICATOR_COLORS[colour_id], -1)


def _mount_pixels(rng: random.Random, jitter: int = 3) -> list[tuple[int, int]]:
    """Mounting points in pixels, with a couple of pixels of placement noise."""
    return [
        (int(fx * IMG_SIZE) + rng.randint(-jitter, jitter),
         int(fy * IMG_SIZE) + rng.randint(-jitter, jitter))
        for fx, fy in MOUNTS
    ]


# --------------------------------------------------------------------------- #
# Image generators
# --------------------------------------------------------------------------- #
def generate_normal(rng: random.Random) -> np.ndarray:
    """A correct assembly: four screws, indicator in canonical order."""
    img = _draw_base(rng)
    _draw_indicator(img, [0, 1, 2], rng)
    for cx, cy in _mount_pixels(rng):
        _draw_screw(img, cx, cy, rng.randint(13, 15), rng)
    return _noise(img, rng)


def generate_logical(rng: random.Random, defect: str) -> tuple[np.ndarray, np.ndarray]:
    """A logical anomaly plus its ground-truth mask.

    The mask marks the region whose *content is wrong* -- an emptied mounting
    point, the location of a spurious screw, or the indicator strip. Every
    individual pixel remains a plausible board/screw pixel; only the
    arrangement is invalid.
    """
    img = _draw_base(rng)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    mounts = _mount_pixels(rng)
    order = [0, 1, 2]

    if defect == "missing_screw":
        drop = rng.randrange(4)
        mounts.pop(drop)
        cx, cy = _mount_pixels(random.Random(0))[drop]
        cv2.circle(mask, (cx, cy), 22, 255, -1)

    elif defect == "wrong_count":
        for drop in sorted(rng.sample(range(4), 2), reverse=True):
            cx, cy = mounts[drop]
            cv2.circle(mask, (cx, cy), 22, 255, -1)
            mounts.pop(drop)

    elif defect == "extra_screw":
        # A perfectly normal screw, in a place the specification has no hole.
        ex = rng.randint(int(IMG_SIZE * 0.42), int(IMG_SIZE * 0.58))
        ey = rng.randint(int(IMG_SIZE * 0.10), int(IMG_SIZE * 0.24))
        mounts.append((ex, ey))
        cv2.circle(mask, (ex, ey), 22, 255, -1)

    elif defect == "misplaced_screw":
        move = rng.randrange(4)
        ox, oy = mounts[move]
        nx = ox + rng.choice([-1, 1]) * rng.randint(38, 60)
        ny = oy + rng.choice([-1, 1]) * rng.randint(20, 40)
        nx = int(np.clip(nx, 25, IMG_SIZE - 25))
        ny = int(np.clip(ny, 25, IMG_SIZE - 25))
        mounts[move] = (nx, ny)
        cv2.circle(mask, (ox, oy), 22, 255, -1)   # where it should have been
        cv2.circle(mask, (nx, ny), 22, 255, -1)   # where it wrongly is

    elif defect == "swapped_indicator":
        while order == [0, 1, 2]:
            rng.shuffle(order)
        y0, y1 = int(IMG_SIZE * 0.70), int(IMG_SIZE * 0.75)
        cv2.rectangle(mask, (int(IMG_SIZE * 0.34), y0), (int(IMG_SIZE * 0.66), y1), 255, -1)

    else:
        raise ValueError(f"Unknown logical defect: {defect}")

    _draw_indicator(img, order, rng)
    for cx, cy in mounts:
        _draw_screw(img, cx, cy, rng.randint(13, 15), rng)

    return _noise(img, rng), mask


def generate_structural(rng: random.Random, defect: str) -> tuple[np.ndarray, np.ndarray]:
    """A structural anomaly: locally invalid texture, globally correct layout."""
    img = generate_normal(rng)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    if defect == "scratch":
        x, y = rng.randint(90, 230), rng.randint(110, 210)
        pts = [(x, y)]
        for _ in range(rng.randint(3, 6)):
            x += rng.randint(-28, 28)
            y += rng.randint(-18, 18)
            pts.append((int(np.clip(x, 5, IMG_SIZE - 5)), int(np.clip(y, 5, IMG_SIZE - 5))))
        arr = np.array(pts, np.int32)
        cv2.polylines(img, [arr], False, (58, 56, 54), rng.randint(2, 4), cv2.LINE_AA)
        cv2.polylines(mask, [arr], False, 255, 6)

    elif defect == "contamination":
        cx, cy = rng.randint(60, 260), rng.randint(60, 260)
        rx, ry = rng.randint(9, 18), rng.randint(7, 15)
        colour = (rng.randint(20, 70),) * 3
        cv2.ellipse(img, (cx, cy), (rx, ry), rng.randint(0, 180), 0, 360, colour, -1, cv2.LINE_AA)
        cv2.ellipse(mask, (cx, cy), (rx + 2, ry + 2), 0, 0, 360, 255, -1)

    elif defect == "chip":
        # A bite taken out of the product edge, filled with board colour.
        ex = rng.choice([int(IMG_SIZE * 0.32), int(IMG_SIZE * 0.68)])
        ey = rng.randint(int(IMG_SIZE * 0.40), int(IMG_SIZE * 0.60))
        size = rng.randint(10, 20)
        cv2.circle(img, (ex, ey), size, BOARD_COLOR, -1, cv2.LINE_AA)
        cv2.circle(mask, (ex, ey), size + 2, 255, -1)

    else:
        raise ValueError(f"Unknown structural defect: {defect}")

    return img, mask


# --------------------------------------------------------------------------- #
# Dataset writer
# --------------------------------------------------------------------------- #
def _save(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def generate_dataset(
    root: str = "data/mvtec_loco",
    category: str = CATEGORY,
    n_train: int = 200,
    n_val: int = 20,
    n_test_good: int = 25,
    n_test_logical: int = 30,
    n_test_structural: int = 30,
    seed: int = 0,
    verbose: bool = True,
) -> str:
    """Write a full MVTec-LOCO-shaped dataset to disk.

    Returns:
        The category directory that was written.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    base = os.path.join(root, category)

    for i in range(n_train):
        _save(os.path.join(base, "train", "good", f"{i:03d}.png"), generate_normal(rng))
    for i in range(n_val):
        _save(os.path.join(base, "validation", "good", f"{i:03d}.png"), generate_normal(rng))
    for i in range(n_test_good):
        _save(os.path.join(base, "test", "good", f"{i:03d}.png"), generate_normal(rng))

    manifest: dict[str, dict[str, str]] = {}
    for family, types, count in (
        ("logical_anomalies", LOGICAL_TYPES, n_test_logical),
        ("structural_anomalies", STRUCTURAL_TYPES, n_test_structural),
    ):
        maker = generate_logical if family == "logical_anomalies" else generate_structural
        for i in range(count):
            defect = types[i % len(types)]
            img, mask = maker(rng, defect)
            _save(os.path.join(base, "test", family, f"{i:03d}.png"), img)
            # Ground truth is a directory of region masks, matching the real
            # benchmark's format even though we only emit one region here.
            _save(os.path.join(base, "ground_truth", family, f"{i:03d}", "000.png"), mask)
            manifest[f"{family}/{i:03d}.png"] = {
                "subtype": defect, "direction": DEFECT_DIRECTION[defect],
            }

    with open(os.path.join(base, SUBTYPE_MANIFEST), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)

    if verbose:
        print(f"Synthetic dataset written to {base}")
        print(f"  train/good           {n_train}")
        print(f"  validation/good      {n_val}")
        print(f"  test/good            {n_test_good}")
        print(f"  test/logical         {n_test_logical}  ({', '.join(LOGICAL_TYPES)})")
        print(f"  test/structural      {n_test_structural}  ({', '.join(STRUCTURAL_TYPES)})")

    return base


if __name__ == "__main__":
    generate_dataset()
