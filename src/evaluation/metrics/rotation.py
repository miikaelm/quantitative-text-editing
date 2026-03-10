"""
metrics/rotation.py — Rotation edit measurement.

Estimates the rotation angle applied to the text in a model output image
by sweeping candidate angles and finding which rotation of the source image
best matches the output (normalized cross-correlation template search).

This avoids relying on OCR — which fails on heavily rotated text — and
instead treats the problem as a 1-D angle search over a known signal.

Primary metrics:
    measured_angle_deg  — estimated CSS-convention clockwise rotation of output text
    angle_error_deg     — |measured_angle_deg − target_angle_deg|
    ecr                 — Edit Completion Ratio: fraction of planned rotation achieved
                          ECR = 1 − angle_error / |rotation_delta_deg|
                          1.0 = perfect, 0.0 = no change, negative = overshot or wrong direction

Inputs (all from metadata):
    old_angle_deg       — source rotation (degrees, CSS clockwise convention)
    new_angle_deg       — target rotation
    rotation_delta_deg  — new_angle_deg − old_angle_deg (signed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RotationMeasurement:
    """Measurement result for a single rotation evaluation."""

    measured_angle_deg: float       # estimated rotation angle in output image (CSS clockwise)
    source_angle_deg: float         # rotation in source (old_angle_deg)
    target_angle_deg: float         # intended rotation (new_angle_deg)
    angle_error_deg: float          # |measured − target| in degrees
    rotation_delta_deg: float       # planned total rotation (target − source)
    ecr: float | None               # Edit Completion Ratio (None if delta ≈ 0)
    search_score: float             # peak NCC score from the angle sweep (quality indicator)


# ---------------------------------------------------------------------------
# Angle sweep — primary measurement strategy
# ---------------------------------------------------------------------------

def _estimate_bg_color(img: np.ndarray) -> tuple[int, int, int]:
    """
    Estimate the background fill colour from the four corner pixels.
    Returns an (R, G, B) integer tuple suitable for PIL fillcolor.
    """
    corners = np.array([
        img[0, 0], img[0, -1], img[-1, 0], img[-1, -1],
    ], dtype=np.float64)
    median_bg = np.median(corners, axis=0)
    return tuple(int(round(v)) for v in median_bg)


def _fg_signal(img: np.ndarray, bg: tuple[int, int, int], tol: float = 12.0) -> np.ndarray:
    """
    Return a float array with background pixels zeroed out.

    Pixels whose max channel distance from bg exceeds tol are kept; the rest
    are set to zero.  This focuses the NCC on the text signal only, which
    is much smaller than the background and would otherwise be swamped by it.
    """
    bg_arr = np.array(bg, dtype=np.float64)
    diff = np.abs(img.astype(np.float64) - bg_arr)         # shape H×W×3
    mask = diff.max(axis=2) > tol                           # True = foreground
    result = img.astype(np.float64).copy()
    result[~mask] = 0.0
    return result


def estimate_rotation_angle(
    source_img: np.ndarray,
    output_img: np.ndarray,
    old_angle_deg: float,
    new_angle_deg: float,
    coarse_step_deg: float = 2.0,
    fine_step_deg: float = 0.25,
    search_padding_deg: float = 20.0,
) -> tuple[float, float]:
    """
    Estimate the rotation angle applied to the source image to produce the output,
    using a two-pass (coarse then fine) foreground-focused NCC sweep.

    Two design decisions vs a naive full-image NCC:
    1. PIL rotation is performed with fillcolor=bg so corner fill pixels do NOT
       create a large mismatch against the output background.
    2. NCC is computed on foreground-only (text) pixels; background pixels are
       zeroed out.  Without this, a uniform background dominates the dot product
       and makes every angle look equally good (NCC ≈ 1.0 everywhere).

    Searches a window of [old_angle − padding, new_angle + padding].

    Returns:
        (best_angle_deg, best_ncc_score)
    """
    from PIL import Image as PILImage

    bg = _estimate_bg_color(source_img)
    src_pil = PILImage.fromarray(source_img)

    # Foreground signal of the output (fixed reference throughout the sweep)
    out_fg = _fg_signal(output_img, bg).ravel()
    out_norm = np.linalg.norm(out_fg)

    lo = min(old_angle_deg, new_angle_deg) - search_padding_deg
    hi = max(old_angle_deg, new_angle_deg) + search_padding_deg

    def _ncc_at(angle: float) -> float:
        # PIL rotates counterclockwise; negate for CSS clockwise convention.
        # fillcolor=bg prevents black corner bleed that would kill the NCC.
        rotated = src_pil.rotate(-angle, resample=PILImage.BICUBIC,
                                  expand=False, fillcolor=bg)
        rot_arr = np.array(rotated)
        rot_fg = _fg_signal(rot_arr, bg).ravel()
        rot_norm = np.linalg.norm(rot_fg)
        if rot_norm < 1e-8 or out_norm < 1e-8:
            return 0.0
        return float(np.dot(rot_fg, out_fg) / (rot_norm * out_norm))

    # --- Coarse pass ---
    coarse_angles = np.arange(lo, hi + coarse_step_deg, coarse_step_deg)
    coarse_scores = [(_ncc_at(a), a) for a in coarse_angles]
    best_score, best_angle = max(coarse_scores)

    # --- Fine pass: ±coarse_step around the coarse best ---
    fine_lo = best_angle - coarse_step_deg
    fine_hi = best_angle + coarse_step_deg
    fine_angles = np.arange(fine_lo, fine_hi + fine_step_deg, fine_step_deg)
    for a in fine_angles:
        s = _ncc_at(a)
        if s > best_score:
            best_score = s
            best_angle = a

    return round(float(best_angle), 2), round(float(best_score), 4)


# ---------------------------------------------------------------------------
# ECR computation
# ---------------------------------------------------------------------------

def _compute_ecr(
    measured_angle: float,
    target_angle: float,
    source_angle: float,
    floor_deg: float = 1.0,
) -> float | None:
    """
    Edit Completion Ratio for rotation:
        ECR = 1 − |measured − target| / |target − source|

    Returns None if the planned delta is below floor_deg (near-identity edit).
    """
    planned = abs(target_angle - source_angle)
    if planned < floor_deg:
        return None
    residual = abs(measured_angle - target_angle)
    return round(1.0 - residual / planned, 4)


# ---------------------------------------------------------------------------
# Primary metric function
# ---------------------------------------------------------------------------

def evaluate_rotation_edit(
    source_img: np.ndarray,
    output_img: np.ndarray,
    metadata: dict,
    coarse_step_deg: float = 2.0,
    fine_step_deg: float = 0.25,
    search_padding_deg: float = 20.0,
) -> RotationMeasurement:
    """
    Evaluate a rotation edit by estimating the rotation angle of the output image.

    Args:
        source_img:          RGB numpy array — the pre-edit source image.
        output_img:          RGB numpy array — the model's output image.
        metadata:            Record metadata dict; must contain old_angle_deg and new_angle_deg.
        coarse_step_deg:     Coarse sweep step size in degrees.
        fine_step_deg:       Fine sweep step size in degrees.
        search_padding_deg:  Extra search window padding beyond [old, new] range.

    Returns:
        RotationMeasurement with measured_angle_deg, angle_error_deg, and ecr.
    """
    old_angle = float(metadata.get("old_angle_deg", 0.0))
    new_angle = float(metadata.get("new_angle_deg", 0.0))
    delta = round(new_angle - old_angle, 4)

    measured_angle, search_score = estimate_rotation_angle(
        source_img=source_img,
        output_img=output_img,
        old_angle_deg=old_angle,
        new_angle_deg=new_angle,
        coarse_step_deg=coarse_step_deg,
        fine_step_deg=fine_step_deg,
        search_padding_deg=search_padding_deg,
    )

    angle_error = round(abs(measured_angle - new_angle), 2)
    ecr = _compute_ecr(measured_angle, new_angle, old_angle)

    return RotationMeasurement(
        measured_angle_deg=measured_angle,
        source_angle_deg=old_angle,
        target_angle_deg=new_angle,
        angle_error_deg=angle_error,
        rotation_delta_deg=delta,
        ecr=ecr,
        search_score=search_score,
    )
