"""
metrics/rotation.py — Rotation edit measurement.

Two measurement strategies are available, selected automatically:

  1. Image-moments (preferred, used when source_bbox is in metadata):
       Crops both images to the text element region (expanded to contain the element
       at any angle), then estimates the absolute orientation of the target crop
       via PCA on foreground pixels.  angle_error = |measured_target − new_angle_deg|.

       Why not NCC for element-level rotation: the NCC full-image sweep rotates the
       entire source image, so the background (dominant by area) always matches best
       at 0° regardless of element rotation.  Image moments work on the crop and
       directly read the absolute text orientation.

  2. NCC sweep (fallback, used when no bbox is available):
       Sweeps candidate angles, rotating the full source image and measuring NCC
       against the full output.  Designed for global image rotation or when no bbox
       can be supplied.  Note: the sweep measures the *delta* rotation (angle applied
       to source), but angle_error is computed against new_angle_deg — correct only
       when old_angle_deg ≈ 0.

Primary metrics:
    measured_angle_deg  — estimated CSS-convention clockwise rotation of output text
    angle_error_deg     — |measured_angle_deg − target_angle_deg|
    ecr                 — Edit Completion Ratio: 1 − angle_error / |rotation_delta_deg|
    fg_pixel_count      — foreground pixels in element crop (moments path); None for NCC
    search_score        — peak NCC score (NCC path); None for moments

Inputs (all from metadata):
    old_angle_deg       — source rotation (degrees, CSS clockwise convention)
    new_angle_deg       — target rotation
    source_bbox         — {x, y, width, height} element bounding box (enables moments path)
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

    measured_angle_deg: float           # estimated rotation angle in output image (CSS clockwise)
    source_angle_deg: float             # rotation in source (old_angle_deg)
    target_angle_deg: float             # intended rotation (new_angle_deg)
    angle_error_deg: float              # |measured − target| in degrees
    rotation_delta_deg: float           # planned total rotation (target − source)
    ecr: float | None                   # Edit Completion Ratio (None if delta ≈ 0)
    fg_pixel_count: int | None          # foreground pixels in element crop (moments path)
    search_score: float | None          # peak NCC score (NCC path)


# ---------------------------------------------------------------------------
# Image-moments strategy (element crop)
# ---------------------------------------------------------------------------

def _crop_to_element(img: np.ndarray, bbox: dict) -> np.ndarray:
    """
    Crop image to a square centred on the element, large enough to contain the
    element at any rotation angle (half-size = bbox diagonal × 0.7 + pad).
    """
    cx = bbox["x"] + bbox["width"] / 2.0
    cy = bbox["y"] + bbox["height"] / 2.0
    half = int(math.hypot(bbox["width"], bbox["height"]) * 0.7) + 4
    H, W = img.shape[:2]
    x0 = max(0, int(cx - half))
    y0 = max(0, int(cy - half))
    x1 = min(W, int(cx + half))
    y1 = min(H, int(cy + half))
    return img[y0:y1, x0:x1]


def _measure_angle_by_moments(crop: np.ndarray) -> tuple[float, int]:
    """
    Estimate the dominant orientation angle (degrees, CSS clockwise) of text in a crop
    using image moments (PCA on foreground pixels).

    Assumes the text mass forms an elongated blob whose principal axis aligns with the
    text line direction.  Reliable when the text is wider than it is tall, which holds
    for the rotation range used in generation (±30°).

    Returns:
        (angle_deg, fg_pixel_count)
        angle_deg is in approximately [−90, 90].
    """
    gray = np.mean(crop, axis=2).astype(np.float64) if crop.ndim == 3 else crop.astype(np.float64)
    if np.mean(gray) > 128:
        gray = 255.0 - gray          # invert: make text pixels bright
    thresh = np.max(gray) * 0.3
    if thresh < 1.0:
        return 0.0, 0
    ys, xs = np.where(gray > thresh)
    n = len(xs)
    if n < 50:
        return 0.0, n
    cx_f = float(xs.mean())
    cy_f = float(ys.mean())
    dx = xs - cx_f
    dy = ys - cy_f
    cxx = float(np.mean(dx * dx))
    cxy = float(np.mean(dx * dy))
    cyy = float(np.mean(dy * dy))
    angle_rad = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    return math.degrees(angle_rad), n


def _evaluate_by_moments(
    source_img: np.ndarray,
    output_img: np.ndarray,
    old_angle: float,
    new_angle: float,
    bbox: dict,
) -> RotationMeasurement:
    """Measure rotation using image moments on the element crop."""
    src_crop = _crop_to_element(source_img, bbox)
    tgt_crop = _crop_to_element(output_img, bbox)

    measured_angle, tgt_fg = _measure_angle_by_moments(tgt_crop)
    _, src_fg = _measure_angle_by_moments(src_crop)
    fg_pixel_count = min(src_fg, tgt_fg)

    delta = round(new_angle - old_angle, 4)
    angle_error = round(abs(measured_angle - new_angle), 2)
    ecr = _compute_ecr(measured_angle, new_angle, old_angle)

    return RotationMeasurement(
        measured_angle_deg=round(measured_angle, 2),
        source_angle_deg=old_angle,
        target_angle_deg=new_angle,
        angle_error_deg=angle_error,
        rotation_delta_deg=delta,
        ecr=ecr,
        fg_pixel_count=fg_pixel_count,
        search_score=None,
    )


# ---------------------------------------------------------------------------
# NCC sweep strategy (full-image fallback)
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

    NOTE: This function rotates the entire source image and compares to the entire
    output image.  It works correctly for global image rotation.  For layouts where
    only one element is rotated, use evaluate_rotation_edit with source_bbox in
    metadata instead — it will use the image-moments path on the element crop.

    The sweep measures the *delta* rotation applied to the source.  The returned
    angle is compared against new_angle_deg in evaluate_rotation_edit, so results
    are only directly interpretable when old_angle_deg ≈ 0.

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


def _evaluate_by_ncc(
    source_img: np.ndarray,
    output_img: np.ndarray,
    old_angle: float,
    new_angle: float,
    coarse_step_deg: float,
    fine_step_deg: float,
    search_padding_deg: float,
) -> RotationMeasurement:
    """Measure rotation using the full-image NCC sweep."""
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
        fg_pixel_count=None,
        search_score=search_score,
    )


# ---------------------------------------------------------------------------
# ECR computation (shared)
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
    Evaluate a rotation edit.

    Automatically selects the measurement strategy:
      - Image-moments on element crop  when source_bbox is present in metadata.
      - Full-image NCC sweep           otherwise (fallback).

    Args:
        source_img:          RGB numpy array — the pre-edit source image.
        output_img:          RGB numpy array — the model output (or ground-truth target).
        metadata:            Record metadata dict; must contain old_angle_deg and new_angle_deg.
                             If source_bbox is present, the moments path is used.
        coarse_step_deg:     NCC path only — coarse sweep step size.
        fine_step_deg:       NCC path only — fine sweep step size.
        search_padding_deg:  NCC path only — search window padding beyond [old, new].

    Returns:
        RotationMeasurement.  fg_pixel_count is set for moments path, search_score for NCC.
    """
    old_angle = float(metadata.get("old_angle_deg", 0.0))
    new_angle = float(metadata.get("new_angle_deg", 0.0))

    if "source_bbox" in metadata:
        return _evaluate_by_moments(source_img, output_img, old_angle, new_angle,
                                    metadata["source_bbox"])

    return _evaluate_by_ncc(source_img, output_img, old_angle, new_angle,
                             coarse_step_deg, fine_step_deg, search_padding_deg)
