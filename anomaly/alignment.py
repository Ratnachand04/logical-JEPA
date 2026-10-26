"""Align a current photo to a reference photo before comparing them.

A user's two photos are never taken from exactly the same place. The model
tolerates small offsets (it is trained on independently jittered pairs), but a
shift of a few percent would otherwise read as "everything moved". ECC image
alignment (Evangelidis & Psarakis, 2008) estimates the affine warp that best
maps the current image onto the reference, and is cheap at 256 px.

The warp is accepted only when it is plausible -- moderate scale, shift and
rotation -- because ECC can converge to a wild warp on low-texture images, and a
wrong alignment is worse than none. Pixels that fall outside the warped current
image are filled from the reference, so the empty border is never reported as
a defect.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

MAX_SHIFT = 0.25        # fraction of the image side
SCALE_RANGE = (0.8, 1.25)
MAX_ROTATION_DEG = 20.0


def _gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.GaussianBlur(gray, (5, 5), 0)


def plausible(warp: np.ndarray, size: int) -> bool:
    """True if a 2x3 affine warp is a moderate camera change."""
    a = warp[:, :2]
    scale = math.sqrt(abs(np.linalg.det(a)))
    rotation = math.degrees(math.atan2(a[1, 0], a[0, 0]))
    shift = float(np.hypot(warp[0, 2], warp[1, 2])) / size
    return (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]
            and abs(rotation) <= MAX_ROTATION_DEG and shift <= MAX_SHIFT)


def align_to_reference(current: np.ndarray, reference: np.ndarray,
                       iterations: int = 200) -> tuple[np.ndarray, np.ndarray, dict]:
    """Warp ``current`` onto ``reference`` (both HxWx3 uint8, same size).

    Returns:
        ``(aligned, valid, info)``: the aligned image with out-of-view pixels
        taken from the reference, a boolean map of pixels that came from the
        current image, and ``{"aligned": bool, "warp": 2x3 array, "cc": float}``.
    """
    h, w = reference.shape[:2]
    identity = np.eye(2, 3, dtype=np.float32)
    info = {"aligned": False, "warp": identity, "cc": float("nan")}
    try:
        cc, warp = cv2.findTransformECC(
            _gray(reference), _gray(current), identity.copy(), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6), None, 5)
    except cv2.error:
        return current.copy(), np.ones((h, w), bool), info
    if not plausible(warp, max(h, w)):
        return current.copy(), np.ones((h, w), bool), info

    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(current, warp, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpAffine(np.ones((h, w), np.float32), warp, (w, h), flags=flags,
                           borderMode=cv2.BORDER_CONSTANT) > 0.99
    aligned[~valid] = reference[~valid]
    info.update(aligned=True, warp=warp, cc=float(cc))
    return aligned, valid, info


def warp_mask(mask: np.ndarray, info: dict) -> np.ndarray:
    """Apply the alignment found by :func:`align_to_reference` to a label mask."""
    if not info.get("aligned"):
        return mask
    h, w = mask.shape[:2]
    return cv2.warpAffine(mask, info["warp"], (w, h),
                          flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT)
