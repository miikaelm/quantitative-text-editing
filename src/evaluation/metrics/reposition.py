"""
metrics/reposition.py — Reposition edit measurement.

Measures positional accuracy of a text bounding box relative to
a ground-truth target bounding box.

Three metrics:
    1. Centroid distance (pixels) — raw Euclidean error between
       measured and target bbox centers.
    2. Normalized distance (0–1) — centroid distance divided by
       image diagonal, resolution-independent.
    3. Edit Completion Ratio (ECR) — analogous to color ECR.
       Measures what fraction of the intended displacement was achieved.
           1.0  = perfect placement
           0.0  = text didn't move (still at original position)
           <0   = moved further away from target than it started
           >1   = overshot past the target

Inputs are simple {x, y, width, height} bbox dicts:
    - target_bbox:   from the ground-truth rendered target image
    - measured_bbox:  from OCR on model output (evaluation) or
                      from metadata (validation)
    - original_bbox:  from the source image (needed only for ECR)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RepositionMeasurement:
    """Measurement result for a single reposition evaluation."""

    measured_centroid: tuple[float, float]
    target_centroid: tuple[float, float]
    centroid_distance: float          # pixels
    normalized_distance: float        # 0–1, relative to image diagonal
    ecr: float | None                 # Edit Completion Ratio
    original_centroid: tuple[float, float] | None  # None if not provided
    planned_distance: float | None    # original→target distance, pixels
    img_width: int
    img_height: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centroid(bbox: dict) -> tuple[float, float]:
    """Return (cx, cy) of a {x, y, width, height} bbox dict."""
    return (
        bbox["x"] + bbox["width"] / 2.0,
        bbox["y"] + bbox["height"] / 2.0,
    )


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _compute_ecr(
    measured_centroid: tuple[float, float],
    target_centroid: tuple[float, float],
    original_centroid: tuple[float, float],
    floor: float = 5.0,
) -> tuple[float | None, float]:
    """
    Edit Completion Ratio.

        ECR = 1 - (dist(measured, target) / dist(original, target))

    Args:
        floor: minimum planned distance in pixels below which ECR
               is undefined (near-identity edits). Default 5px.

    Returns:
        (ecr_or_none, planned_distance)
    """
    planned = _euclidean(original_centroid, target_centroid)
    if planned < floor:
        return None, planned

    residual = _euclidean(measured_centroid, target_centroid)
    ecr = 1.0 - (residual / planned)
    return round(ecr, 4), round(planned, 2)


# ---------------------------------------------------------------------------
# Primary metric function
# ---------------------------------------------------------------------------

def evaluate_reposition_edit(
    target_bbox: dict,
    measured_bbox: dict,
    img_width: int = 512,
    img_height: int = 512,
    original_bbox: dict | None = None,
    ecr_floor: float = 5.0,
) -> RepositionMeasurement:
    """
    Evaluate positional accuracy of a repositioned text element.

    Args:
        target_bbox:   {x, y, width, height} — ground-truth target position.
        measured_bbox:  {x, y, width, height} — observed position (OCR or metadata).
        img_width:      Image width in pixels (default 512).
        img_height:     Image height in pixels.
        original_bbox:  {x, y, width, height} — pre-edit position (needed for ECR).
                        If None, ECR is not computed.
        ecr_floor:      Minimum planned displacement (px) for ECR to be defined.

    Returns:
        RepositionMeasurement with centroid_distance, normalized_distance, and ecr.
    """
    m_c = _centroid(measured_bbox)
    t_c = _centroid(target_bbox)

    dist = _euclidean(m_c, t_c)
    diagonal = math.hypot(img_width, img_height)
    norm_dist = dist / diagonal if diagonal > 0 else 0.0

    # ECR requires original bbox
    ecr = None
    o_c = None
    planned = None

    if original_bbox is not None:
        o_c = _centroid(original_bbox)
        ecr, planned = _compute_ecr(m_c, t_c, o_c, floor=ecr_floor)

    return RepositionMeasurement(
        measured_centroid=(round(m_c[0], 1), round(m_c[1], 1)),
        target_centroid=(round(t_c[0], 1), round(t_c[1], 1)),
        centroid_distance=round(dist, 2),
        normalized_distance=round(norm_dist, 4),
        ecr=ecr,
        original_centroid=(round(o_c[0], 1), round(o_c[1], 1)) if o_c else None,
        planned_distance=planned,
        img_width=img_width,
        img_height=img_height,
    )
