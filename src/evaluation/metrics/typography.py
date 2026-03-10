"""
metrics/typography.py — Typography edit measurement.

Covers four subcategories, each with its own primary metric and ECR:

    font_weight    — stroke width change, measured via skeletonization.
                     Primary metric: stroke_width_px (mean stroke thickness).
                     ECR = 1 - |measured_sw - target_sw| / |target_sw - source_sw|

    font_style     — italic shear angle, measured via horizontal centroid profile.
                     Primary metric: shear_angle_deg (degrees of lean from vertical).
                     ECR = 1 - |measured_angle - target_angle| / |target_angle - source_angle|

    font_family    — glyph aspect ratio (bbox width / bbox height).
                     Primary metric: aspect_ratio.
                     ECR = 1 - |measured_ar - target_ar| / |target_ar - source_ar|

    letter_spacing — text bbox width (relative to bbox height, a spacing ratio).
                     Primary metric: spacing_ratio (bbox_width / bbox_height).
                     ECR = 1 - |measured_sr - target_sr| / |target_sr - source_sr|

For font_weight and font_style, this module computes pixel-based measurements by
loading a crop of the text region, binarizing it, and then either skeletonizing
(stroke width) or fitting a centroid profile (shear angle).

Metadata expected by evaluate_typography_edit:
    source_bbox             {x, y, width, height}  — always required
    target_bbox             {x, y, width, height}  — for aspect ratio / spacing subcategories
    target_stroke_width     float                   — required for font_weight ECR
    target_shear_angle      float                   — required for font_style ECR
    typography_subcategory  str

scikit-image is required for font_weight and font_style metrics.
If scikit-image is not installed, those metrics degrade gracefully to ECR=None.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TypographyMeasurement:
    """Measurement result for a single typography evaluation."""

    typography_subcategory: str      # "font_weight" | "font_style" | "font_family" | "letter_spacing"
    primary_metric: str              # name of the key measurement
    source_value: float              # measured value in the source image / metadata
    target_value: float              # expected value from the target / metadata
    measured_value: float            # measured value in the output image
    absolute_error: float            # |measured_value - target_value|
    planned_delta: float | None      # |target_value - source_value|
    ecr: float | None                # Edit Completion Ratio (None if underdetermined)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def binarize_text_region(image: np.ndarray, bbox: dict, pad: int = 4) -> np.ndarray:
    """
    Crop the image to the text bounding box (with padding) and binarize.

    Returns a boolean array where True = text pixel, False = background.
    Assumes text is darker than background (works for both light-on-dark and
    dark-on-light after inverting if needed).

    Args:
        image: RGB or grayscale numpy array.
        bbox:  {x, y, width, height} dict.
        pad:   Extra pixels to add around the crop on each side.
    """
    x = int(bbox["x"])
    y = int(bbox["y"])
    w = int(bbox["width"])
    h = int(bbox["height"])

    if w <= 0 or h <= 0:
        return np.zeros((1, 1), dtype=bool)

    img_h, img_w = image.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + w + pad)
    y1 = min(img_h, y + h + pad)

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((1, 1), dtype=bool)

    # Convert to grayscale
    if crop.ndim == 3:
        gray = np.mean(crop, axis=2)
    else:
        gray = crop.astype(float)

    # Simple Otsu-like threshold: midpoint between background (corner) and text
    corner_vals = [
        gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1],
    ]
    bg_level = float(np.median(corner_vals))
    text_level = float(np.percentile(gray, 5))   # darkest 5% ≈ text pixels

    if abs(bg_level - text_level) < 10:
        # Low contrast — use global Otsu threshold instead
        threshold = float(np.mean(gray))
    else:
        threshold = (bg_level + text_level) / 2.0

    # If background is dark (text is bright), invert the comparison
    if bg_level < text_level:
        return gray > threshold
    return gray < threshold


def compute_stroke_width(binary: np.ndarray) -> float:
    """
    Estimate mean stroke width of binarized text via skeletonization.

    Algorithm:
        1. Skeletonize the binary text mask (one-pixel-wide centerlines).
        2. Compute distance transform on the binary mask (distance to background).
        3. Average the distance-transform values at skeleton pixels × 2 (diameter).

    Requires scikit-image and scipy.  Returns 0.0 if unavailable or empty.
    """
    try:
        from skimage.morphology import skeletonize
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return 0.0

    if not binary.any():
        return 0.0

    binary_bool = binary.astype(bool)
    skeleton = skeletonize(binary_bool)
    if not skeleton.any():
        return 0.0

    dist = distance_transform_edt(binary_bool)
    stroke_widths = dist[skeleton] * 2.0
    return float(np.mean(stroke_widths))


def compute_shear_angle(binary: np.ndarray) -> float:
    """
    Estimate the shear (lean) angle of text from a binarized image.

    Algorithm (centroid-profile method):
        For each row y that contains text pixels, compute the horizontal
        centroid of text pixels x̄(y).  Fit a line x̄ = slope·y + c.
        The slope is tan(shear_angle); angle = arctan(slope) in degrees.

    Interpretation:
        0°   → upright (normal)
        < 0° → leans right (typical italic, e.g. −12°)
        > 0° → leans left (backslash italic)

    Returns 0.0 if there are too few rows to fit a line.
    """
    rows_with_text = np.where(binary.any(axis=1))[0]
    if len(rows_with_text) < 5:
        return 0.0

    h_centroids = []
    y_coords = []
    for y in rows_with_text:
        xs = np.where(binary[y])[0]
        if len(xs) > 0:
            h_centroids.append(float(np.mean(xs)))
            y_coords.append(float(y))

    if len(h_centroids) < 5:
        return 0.0

    # Fit: centroid_x = slope * y + intercept → slope = d(x)/d(y)
    coeffs = np.polyfit(y_coords, h_centroids, 1)
    slope = float(coeffs[0])
    return float(np.degrees(np.arctan(slope)))


# ---------------------------------------------------------------------------
# ECR helper
# ---------------------------------------------------------------------------

def _compute_ecr(
    measured: float,
    target: float,
    source: float,
    floor: float = 0.01,
) -> tuple[float | None, float | None]:
    """
    Edit Completion Ratio:
        ECR = 1 - |measured - target| / |target - source|

    Returns (ecr_or_none, planned_delta).
    """
    planned_delta = abs(target - source)
    if planned_delta < floor:
        return None, round(planned_delta, 4)
    residual = abs(measured - target)
    ecr = 1.0 - residual / planned_delta
    return round(ecr, 4), round(planned_delta, 4)


# ---------------------------------------------------------------------------
# Subcategory metric implementations
# ---------------------------------------------------------------------------

def _eval_font_weight(
    source_img: np.ndarray,
    output_img: np.ndarray,
    metadata: dict,
    measured_bbox: dict,
) -> TypographyMeasurement:
    source_bbox = metadata.get("source_bbox", measured_bbox)
    source_sw = metadata.get("source_stroke_width")
    target_sw = metadata.get("target_stroke_width")

    # Compute source stroke width on the fly if not pre-stored
    if source_sw is None and source_img is not None:
        src_bin = binarize_text_region(source_img, source_bbox)
        source_sw = compute_stroke_width(src_bin)

    out_bin = binarize_text_region(output_img, measured_bbox)
    measured_sw = compute_stroke_width(out_bin)

    source_sw = source_sw or 0.0
    target_sw = target_sw or 0.0

    ecr, planned_delta = _compute_ecr(measured_sw, target_sw, source_sw)
    error = abs(measured_sw - target_sw)

    return TypographyMeasurement(
        typography_subcategory="font_weight",
        primary_metric="stroke_width_px",
        source_value=round(source_sw, 3),
        target_value=round(target_sw, 3),
        measured_value=round(measured_sw, 3),
        absolute_error=round(error, 3),
        planned_delta=planned_delta,
        ecr=ecr,
    )


def _eval_font_style(
    source_img: np.ndarray,
    output_img: np.ndarray,
    metadata: dict,
    measured_bbox: dict,
) -> TypographyMeasurement:
    source_bbox = metadata.get("source_bbox", measured_bbox)
    source_angle = metadata.get("source_shear_angle")
    target_angle = metadata.get("target_shear_angle")

    if source_angle is None and source_img is not None:
        src_bin = binarize_text_region(source_img, source_bbox)
        source_angle = compute_shear_angle(src_bin)

    out_bin = binarize_text_region(output_img, measured_bbox)
    measured_angle = compute_shear_angle(out_bin)

    source_angle = source_angle or 0.0
    target_angle = target_angle or 0.0

    ecr, planned_delta = _compute_ecr(measured_angle, target_angle, source_angle)
    error = abs(measured_angle - target_angle)

    return TypographyMeasurement(
        typography_subcategory="font_style",
        primary_metric="shear_angle_deg",
        source_value=round(source_angle, 3),
        target_value=round(target_angle, 3),
        measured_value=round(measured_angle, 3),
        absolute_error=round(error, 3),
        planned_delta=planned_delta,
        ecr=ecr,
    )


def _bbox_aspect_ratio(bbox: dict) -> float:
    """width / height of a text bbox."""
    h = float(bbox.get("height", 1))
    w = float(bbox.get("width", 0))
    return w / max(h, 1.0)


def _eval_font_family(
    metadata: dict,
    measured_bbox: dict,
) -> TypographyMeasurement:
    source_bbox = metadata.get("source_bbox", measured_bbox)
    target_bbox = metadata.get("target_bbox", measured_bbox)

    source_ar = _bbox_aspect_ratio(source_bbox)
    target_ar = _bbox_aspect_ratio(target_bbox)
    measured_ar = _bbox_aspect_ratio(measured_bbox)

    ecr, planned_delta = _compute_ecr(measured_ar, target_ar, source_ar)
    error = abs(measured_ar - target_ar)

    return TypographyMeasurement(
        typography_subcategory="font_family",
        primary_metric="aspect_ratio",
        source_value=round(source_ar, 4),
        target_value=round(target_ar, 4),
        measured_value=round(measured_ar, 4),
        absolute_error=round(error, 4),
        planned_delta=planned_delta,
        ecr=ecr,
    )


def _eval_letter_spacing(
    metadata: dict,
    measured_bbox: dict,
) -> TypographyMeasurement:
    """Use width/height ratio (spacing_ratio) as the primary signal for letter spacing."""
    source_bbox = metadata.get("source_bbox", measured_bbox)
    target_bbox = metadata.get("target_bbox", measured_bbox)

    source_sr = _bbox_aspect_ratio(source_bbox)
    target_sr = _bbox_aspect_ratio(target_bbox)
    measured_sr = _bbox_aspect_ratio(measured_bbox)

    ecr, planned_delta = _compute_ecr(measured_sr, target_sr, source_sr)
    error = abs(measured_sr - target_sr)

    return TypographyMeasurement(
        typography_subcategory="letter_spacing",
        primary_metric="spacing_ratio",
        source_value=round(source_sr, 4),
        target_value=round(target_sr, 4),
        measured_value=round(measured_sr, 4),
        absolute_error=round(error, 4),
        planned_delta=planned_delta,
        ecr=ecr,
    )


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def evaluate_typography_edit(
    source_img: np.ndarray | None,
    output_img: np.ndarray,
    metadata: dict,
    measured_bbox: dict,
) -> TypographyMeasurement:
    """
    Evaluate a typography edit of any subcategory.

    Args:
        source_img:    Source image as RGB numpy array (may be None for bbox-only subcategories).
        output_img:    Model output image as RGB numpy array.
        metadata:      Pair metadata dict.  Must contain typography_subcategory and source_bbox.
                       For font_weight / font_style: also target_stroke_width / target_shear_angle.
                       For font_family / letter_spacing: also target_bbox.
        measured_bbox: OCR-detected text bbox in the output image {x, y, width, height}.

    Returns:
        TypographyMeasurement with primary metric, absolute_error, and ecr.
    """
    subcategory = metadata.get("typography_subcategory", "letter_spacing")

    if subcategory == "font_weight":
        return _eval_font_weight(source_img, output_img, metadata, measured_bbox)
    if subcategory == "font_style":
        return _eval_font_style(source_img, output_img, metadata, measured_bbox)
    if subcategory == "font_family":
        return _eval_font_family(metadata, measured_bbox)
    if subcategory == "letter_spacing":
        return _eval_letter_spacing(metadata, measured_bbox)

    raise ValueError(f"Unknown typography_subcategory: {subcategory!r}")
