"""
metrics/scaling.py — Scaling edit measurement.

Measures font-size accuracy of a text element relative to a ground-truth
target bounding box, using bounding box height as the proxy for rendered
font size.

Three metrics:
    1. measured_scale — actual height ratio (measured_h / source_h).
       Represents the scale factor the model produced relative to the original.
    2. ratio_error — |measured_scale - target_scale|.
       How far the model's scale is from the intended target (lower is better).
    3. Edit Completion Ratio (ECR) — analogous to color/reposition ECR.
       Measures what fraction of the intended scale change was achieved.
           1.0  = perfect (model hit the target scale exactly)
           0.0  = no change (model left text at original size)
           <0   = moved further from target than it started
           >1   = overshot past the target scale

Inputs are {x, y, width, height} bbox dicts from OCR or metadata.
Height is used as the primary proxy for font size, since it is more
robust than width (which varies with text content length).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScalingMeasurement:
    """Measurement result for a single scaling evaluation."""

    source_height: float             # text bbox height in source image (pixels)
    target_height: float             # text bbox height in target/ground-truth image (pixels)
    measured_height: float           # text bbox height in output image (pixels)
    target_scale: float              # target_height / source_height
    measured_scale: float            # measured_height / source_height
    ratio_error: float               # |measured_scale - target_scale|
    ecr: float | None                # Edit Completion Ratio (None if source unavailable)
    planned_scale_delta: float | None  # |target_scale - 1.0| — magnitude of intended change


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _height(bbox: dict) -> float:
    return float(bbox["height"])


def _compute_ecr(
    measured_scale: float,
    target_scale: float,
    floor: float = 0.05,
) -> tuple[float | None, float | None]:
    """
    Edit Completion Ratio for scaling.

        ECR = 1 - |measured_scale - target_scale| / |target_scale - 1.0|

    Args:
        floor: minimum |target_scale - 1.0| below which ECR is undefined
               (near-identity edits). Default 0.05 (5% change).

    Returns:
        (ecr_or_none, planned_scale_delta)
    """
    planned_delta = abs(target_scale - 1.0)
    if planned_delta < floor:
        return None, planned_delta

    residual = abs(measured_scale - target_scale)
    ecr = 1.0 - (residual / planned_delta)
    return round(ecr, 4), round(planned_delta, 4)


# ---------------------------------------------------------------------------
# Primary metric function
# ---------------------------------------------------------------------------

def evaluate_scaling_edit(
    source_bbox: dict,
    target_bbox: dict,
    measured_bbox: dict,
) -> ScalingMeasurement:
    """
    Evaluate font-size accuracy of a scaled text element.

    Args:
        source_bbox:   {x, y, width, height} — text bbox in the source image.
        target_bbox:   {x, y, width, height} — ground-truth text bbox in the target image.
        measured_bbox: {x, y, width, height} — text bbox found in the model output image.

    Returns:
        ScalingMeasurement with ratio_error and ecr as the primary signals.
    """
    src_h = _height(source_bbox)
    tgt_h = _height(target_bbox)
    meas_h = _height(measured_bbox)

    # Guard against zero-height bboxes (degenerate OCR result)
    if src_h <= 0:
        src_h = 1.0

    target_scale = tgt_h / src_h
    measured_scale = meas_h / src_h
    ratio_error = abs(measured_scale - target_scale)

    ecr, planned_delta = _compute_ecr(measured_scale, target_scale)

    return ScalingMeasurement(
        source_height=round(src_h, 1),
        target_height=round(tgt_h, 1),
        measured_height=round(meas_h, 1),
        target_scale=round(target_scale, 4),
        measured_scale=round(measured_scale, 4),
        ratio_error=round(ratio_error, 4),
        ecr=ecr,
        planned_scale_delta=planned_delta,
    )
