"""
preservation.py — Edit-type-agnostic unintended modification detection.

Checks whether pixels outside the edit bounding box changed between
source and output, flagging unintended side effects of model edits.

Used by both evaluate.py (post-inference) and validate.py (pipeline QC).
"""

from typing import Any

import numpy as np


def check_unintended_modifications(
    source_img: np.ndarray,
    output_img: np.ndarray,
    metadata: dict,
) -> dict[str, Any]:
    """
    Check whether pixels outside the edit bounding box changed.

    Args:
        source_img: H×W×3 uint8 source (before edit).
        output_img: H×W×3 uint8 output (model output or ground truth target).
        metadata:   Pair metadata dict; must contain "source_bbox" to run.

    Returns:
        Dict with keys:
            outside_mse            — mean squared error of outside pixels
            outside_psnr           — PSNR of the non-edited region
            outside_changed_ratio  — fraction of outside pixels that changed
        Or a note explaining why the check was skipped.
    """
    result: dict[str, Any] = {}

    if "source_bbox" not in metadata:
        result["note"] = "no bbox available for unintended modification check"
        return result

    sb = metadata["source_bbox"]
    x, y, w, h = sb["x"], sb["y"], sb["width"], sb["height"]

    # Handle shape mismatch (model may output different resolution)
    if source_img.shape != output_img.shape:
        result["shape_mismatch"] = True
        result["source_shape"] = list(source_img.shape)
        result["output_shape"] = list(output_img.shape)
        return result

    # Mask: True = outside edit region
    mask = np.ones(source_img.shape[:2], dtype=bool)
    y_end = min(y + h, source_img.shape[0])
    x_end = min(x + w, source_img.shape[1])
    mask[y:y_end, x:x_end] = False

    outside_source = source_img[mask].astype(np.float64)
    outside_output = output_img[mask].astype(np.float64)

    if outside_source.size == 0:
        result["note"] = "bbox covers entire image"
        return result

    diff = outside_source - outside_output
    mse = float(np.mean(diff ** 2))
    result["outside_mse"] = round(mse, 4)

    if mse > 0:
        result["outside_psnr"] = round(10 * np.log10(255.0 ** 2 / mse), 2)
    else:
        result["outside_psnr"] = float("inf")

    # Fraction of pixels that changed (with small tolerance for JPEG artefacts)
    pixel_diff = np.abs(diff).max(axis=-1) if diff.ndim > 1 else np.abs(diff)
    changed = pixel_diff > 5
    result["outside_changed_ratio"] = round(float(changed.mean()), 6)

    return result
