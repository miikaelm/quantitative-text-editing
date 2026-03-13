"""
validate.py — Post-rendering validation for generated pairs.

Calls evaluation metric primitives with pass/fail thresholds.
Runs after render.py, before train/test splitting.

Currently implements: color, scaling, typography, reposition (alignment), rotation edit validation.
TODO: content edit validation.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import math

import numpy as np
from PIL import Image

from evaluation.metrics.color import evaluate_color_edit, ColorMeasurement
from evaluation.metrics.rotation import evaluate_rotation_edit, RotationMeasurement
from evaluation.metrics.scaling import evaluate_scaling_edit, ScalingMeasurement
from evaluation.metrics.typography import (
    binarize_text_region,
    compute_stroke_width,
    compute_shear_angle,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """Thresholds for pass/fail decisions."""
    # Color edit thresholds
    max_color_delta_e: float = 8.0          # target color must be near-exact in ground truth
    min_ecr: float = 0.75
    max_ecr: float = 1.1

    # Scaling edit thresholds
    max_scaling_ratio_error: float = 0.05   # rendered size must be within 5% of target scale

    # Typography edit thresholds
    max_typography_absolute_error: float = 1.5   # stroke width or shear angle tolerance
    max_typography_aspect_ratio_error: float = 0.15  # aspect ratio tolerance for font_family
    max_typography_spacing_ratio_error: float = 0.10  # spacing ratio tolerance for letter_spacing
    min_typography_ecr: float = 0.70
    max_typography_ecr: float = 1.15

    # Rotation edit thresholds
    max_rotation_angle_error: float = 3.0        # measured angle must be within 3° of target
    min_rotation_ecr: float = 0.80               # at least 80% of planned rotation achieved
    min_rotation_search_score: float = 0.80      # NCC score must be high enough to trust measurement


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    pair_id: str
    edit_type: str
    passed: bool
    checks: dict[str, bool]
    details: dict[str, float | str]
    metadata: dict

    @property
    def failure_reasons(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


# ---------------------------------------------------------------------------
# Metadata schema (expected fields per edit type)
# ---------------------------------------------------------------------------

# Color edit metadata:
# {
#     "edit_type": "color",
#     "target_element": "title",              # semantic label of edited element
#     "text_content": "Welcome to My Site",   # actual text (for OCR-based ROI in model eval)
#     "old_value": "#1A1A1A",                 # original color hex
#     "new_value": "#3B82F6",                 # target color hex
#     "bbox": [x, y, w, h],                   # rendered bounding box (pixels, from browser)
# }


# ---------------------------------------------------------------------------
# Single pair validation
# ---------------------------------------------------------------------------

def validate_pair(
    pair: dict,
    image_dir: Path,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """
    Validate a single generated pair.

    For color edits:
        1. images_exist: both source and target images present
        2. edit_applied: histogram diff detects correct color change (ΔE ≤ threshold)
        3. measurement_confident: enough text pixels and clear signal
    """
    config = config or ValidationConfig()
    checks = {}
    details = {}

    source_path = image_dir / f"{pair['pair_id']}_source.png"
    target_path = image_dir / f"{pair['pair_id']}_target.png"

    # --- Check 1: Images exist ---
    if not source_path.exists() or not target_path.exists():
        return ValidationResult(
            pair_id=pair["pair_id"],
            edit_type=pair.get("edit_type", "unknown"),
            passed=False,
            checks={"images_exist": False},
            details={"error": "missing image files"},
            metadata=pair["metadata"]
        )
    checks["images_exist"] = True

    source_img = np.array(Image.open(source_path).convert("RGB"))
    target_img = np.array(Image.open(target_path).convert("RGB"))

    edit_type = pair["edit_type"]
    metadata = pair["metadata"]

    validator = _VALIDATORS.get(edit_type)
    if validator is not None:
        validator(source_img, target_img, metadata, config, checks, details)
    else:
        details["note"] = f"{edit_type} validation not yet implemented"

    passed = all(checks.values())
    return ValidationResult(
        pair_id=pair["pair_id"],
        edit_type=edit_type,
        passed=passed,
        checks=checks,
        details=details,
        metadata=metadata
    )


def _validate_color(
    source_img: np.ndarray,
    target_img: np.ndarray,
    metadata: dict,
    config: ValidationConfig,
    checks: dict,
    details: dict,
) -> None:
    """
    Validate a color edit using histogram differencing.

    Compares ground truth source vs ground truth target.
    For pipeline validation, the target IS the ground truth,
    so ΔE should be ≈ 0. Any deviation indicates a pipeline
    or measurement issue.
    """
    # source_bbox may be absent if OCR failed during generation — fail fast.
    if "source_bbox" not in metadata:
        checks["bbox_available"] = False
        details["error"] = "source_bbox missing (OCR failed during generation)"
        return
    checks["bbox_available"] = True

    # source_bbox is stored as {"x", "y", "width", "height"} by generate.py
    sb = metadata["source_bbox"]
    bbox = (sb["x"], sb["y"], sb["width"], sb["height"])
    target_hex = metadata["new_value"]
    planned_delta_e = metadata["planned_delta_e"]

    measurement: ColorMeasurement = evaluate_color_edit(
        source_image=source_img,
        output_image=target_img,
        bbox=bbox,
        target_color_hex=target_hex,
        planned_delta_e=planned_delta_e
    )

    # Check: color is correct
    checks["edit_applied"] = measurement.delta_e <= config.max_color_delta_e and (measurement.edit_completion_ratio >= config.min_ecr and measurement.edit_completion_ratio <= config.max_ecr)

    # Check: measurement is reliable (enough text pixels to trust the color sample)
    checks["measurement_confident"] = measurement.text_pixel_count >= 50

    # Record details for debugging
    details["measured_color"] = measurement.measured_hex
    details["target_color"] = measurement.target_hex
    details["delta_e"] = round(measurement.delta_e, 4)
    details["edit_completion_ratio"] = measurement.edit_completion_ratio
    details["exact_match"] = measurement.exact_match
    details["old_color"] = metadata.get("old_value", "unknown")


def _validate_reposition(
    source_img: np.ndarray,
    target_img: np.ndarray,
    metadata: dict,
    config: ValidationConfig,
    checks: dict,
    details: dict,
) -> None:
    """
    Validate a reposition edit by verifying text moved from source to target.

    Uses OCR bboxes stored in metadata by generate.py, so no OCR is re-run here.
    Checks that the target_bbox centroid is sufficiently displaced from source_bbox.
    """
    if "target_bbox" not in metadata:
        checks["bbox_available"] = False
        details["error"] = "target_bbox missing (OCR failed during generation)"
        return
    checks["bbox_available"] = True

    tb = metadata["target_bbox"]
    tgt_cx = tb["x"] + tb["width"] / 2.0
    tgt_cy = tb["y"] + tb["height"] / 2.0
    details["target_centroid"] = (round(tgt_cx, 1), round(tgt_cy, 1))

    # Verify text actually moved from its original position
    if "source_bbox" in metadata:
        sb = metadata["source_bbox"]
        src_cx = sb["x"] + sb["width"] / 2.0
        src_cy = sb["y"] + sb["height"] / 2.0
        pixels_moved = math.hypot(tgt_cx - src_cx, tgt_cy - src_cy)
        checks["text_moved"] = pixels_moved > 20   # at least 20px displacement
        details["pixels_moved"] = round(pixels_moved, 1)
        details["source_centroid"] = (round(src_cx, 1), round(src_cy, 1))


def _validate_scaling(
    source_img: np.ndarray,
    target_img: np.ndarray,
    metadata: dict,
    config: ValidationConfig,
    checks: dict,
    details: dict,
) -> None:
    """
    Validate a scaling edit by comparing bounding box heights from metadata.

    Uses source_bbox and target_bbox stored in metadata by generate.py.
    Checks that the rendered target bbox height matches the intended scale factor
    within tolerance, confirming the HTML template rendered correctly.
    """
    if "source_bbox" not in metadata or "target_bbox" not in metadata:
        missing = [k for k in ("source_bbox", "target_bbox") if k not in metadata]
        checks["bbox_available"] = False
        details["error"] = f"missing bboxes: {missing} (OCR failed during generation)"
        return
    checks["bbox_available"] = True

    measurement: ScalingMeasurement = evaluate_scaling_edit(
        source_bbox=metadata["source_bbox"],
        target_bbox=metadata["target_bbox"],
        measured_bbox=metadata["target_bbox"],   # ground truth: target IS the measurement
    )

    checks["edit_applied"] = measurement.ratio_error <= config.max_scaling_ratio_error

    details["source_height_px"] = measurement.source_height
    details["target_height_px"] = measurement.target_height
    details["target_scale"] = measurement.target_scale
    details["ratio_error"] = measurement.ratio_error
    details["old_value"] = metadata.get("old_value", "unknown")
    details["new_value"] = metadata.get("new_value", "unknown")
    details["scale_factor"] = metadata.get("scale_factor", "unknown")


def _validate_typography(
    source_img: np.ndarray,
    target_img: np.ndarray,
    metadata: dict,
    config: ValidationConfig,
    checks: dict,
    details: dict,
) -> None:
    """
    Validate a typography edit by comparing the ground-truth target to the source.

    For font_weight and font_style: computes pixel-based measurements from images.
    For font_family and letter_spacing: uses OCR bounding boxes from metadata.

    In the pipeline validation context, target IS the ground truth, so the edit
    should be fully applied (ECR ≈ 1.0) and within tolerance.
    """
    subcategory = metadata.get("typography_subcategory", "unknown")
    details["typography_subcategory"] = subcategory

    if "source_bbox" not in metadata:
        checks["bbox_available"] = False
        details["error"] = "source_bbox missing (OCR failed during generation)"
        return
    checks["bbox_available"] = True

    source_bbox = metadata["source_bbox"]

    if subcategory in ("font_weight", "font_style"):
        # Pixel-based validation using both images
        src_bin = binarize_text_region(source_img, source_bbox)

        if subcategory == "font_weight":
            tgt_bbox = metadata.get("target_bbox", source_bbox)
            tgt_bin = binarize_text_region(target_img, tgt_bbox)
            source_sw = compute_stroke_width(src_bin)
            target_sw = compute_stroke_width(tgt_bin)
            planned_delta = abs(target_sw - source_sw)
            details["source_stroke_width"] = round(source_sw, 3)
            details["target_stroke_width"] = round(target_sw, 3)
            details["planned_stroke_delta"] = round(planned_delta, 3)
            checks["edit_applied"] = planned_delta >= 0.5   # at least 0.5px stroke change

        else:  # font_style
            tgt_bbox = metadata.get("target_bbox", source_bbox)
            tgt_bin = binarize_text_region(target_img, tgt_bbox)
            source_angle = compute_shear_angle(src_bin)
            target_angle = compute_shear_angle(tgt_bin)
            planned_delta = abs(target_angle - source_angle)
            details["source_shear_angle"] = round(source_angle, 3)
            details["target_shear_angle"] = round(target_angle, 3)
            details["planned_angle_delta"] = round(planned_delta, 3)
            checks["edit_applied"] = planned_delta >= 3.0   # at least 3° of shear change

    elif subcategory in ("font_family", "letter_spacing"):
        if "target_bbox" not in metadata:
            checks["bbox_available"] = False
            details["error"] = "target_bbox missing (OCR failed during generation)"
            return

        tb = metadata["target_bbox"]
        sb = metadata["source_bbox"]

        def _ar(bbox):
            return bbox["width"] / max(bbox["height"], 1.0)

        source_ar = _ar(sb)
        target_ar = _ar(tb)
        planned_delta = abs(target_ar - source_ar)

        label = "aspect_ratio" if subcategory == "font_family" else "spacing_ratio"
        thresh = (config.max_typography_aspect_ratio_error
                  if subcategory == "font_family"
                  else config.max_typography_spacing_ratio_error)

        details[f"source_{label}"] = round(source_ar, 4)
        details[f"target_{label}"] = round(target_ar, 4)
        details[f"planned_{label}_delta"] = round(planned_delta, 4)
        checks["edit_applied"] = planned_delta >= thresh * 0.5  # minimum detectable change

    else:
        details["note"] = f"typography subcategory {subcategory!r} validation not yet implemented"


def _validate_rotation(
    source_img: np.ndarray,
    target_img: np.ndarray,
    metadata: dict,
    config: ValidationConfig,
    checks: dict,
    details: dict,
) -> None:
    """
    Validate a rotation edit.

    Delegates to evaluate_rotation_edit, which selects the measurement strategy
    automatically: image-moments on the element crop when source_bbox is present,
    full-image NCC sweep otherwise.

    Checks:
        angles_available      — old/new angle metadata present
        edit_applied          — angle_error_deg ≤ max_rotation_angle_error
        measurement_confident — fg_pixel_count ≥ 50 (moments) or search_score ≥ threshold (NCC)
    """
    if "old_angle_deg" not in metadata or "new_angle_deg" not in metadata:
        checks["angles_available"] = False
        details["error"] = "old_angle_deg / new_angle_deg missing from metadata"
        return
    checks["angles_available"] = True

    measurement: RotationMeasurement = evaluate_rotation_edit(
        source_img=source_img,
        output_img=target_img,
        metadata=metadata,
    )

    checks["edit_applied"] = measurement.angle_error_deg <= config.max_rotation_angle_error
    if measurement.fg_pixel_count is not None:
        checks["measurement_confident"] = measurement.fg_pixel_count >= 50
    else:
        checks["measurement_confident"] = (measurement.search_score or 0.0) >= config.min_rotation_search_score

    details["source_angle_deg"] = measurement.source_angle_deg
    details["target_angle_deg"] = measurement.target_angle_deg
    details["measured_angle_deg"] = measurement.measured_angle_deg
    details["angle_error_deg"] = measurement.angle_error_deg
    details["rotation_delta_deg"] = measurement.rotation_delta_deg
    details["ecr"] = measurement.ecr
    if measurement.fg_pixel_count is not None:
        details["fg_pixel_count"] = measurement.fg_pixel_count
    else:
        details["search_score"] = measurement.search_score
    details["old_value"] = metadata.get("old_value", "unknown")
    details["new_value"] = metadata.get("new_value", "unknown")


# ---------------------------------------------------------------------------
# Dispatch table — add new edit type validators here
# ---------------------------------------------------------------------------

_VALIDATORS = {
    "color":      _validate_color,
    "alignment":  _validate_reposition,   # same positional validation logic
    "scaling":    _validate_scaling,
    "typography": _validate_typography,
    "rotation":   _validate_rotation,
    # "content":  _validate_content,
}


# ---------------------------------------------------------------------------
# Dataset-level validation
# ---------------------------------------------------------------------------

def validate_dataset(
    jsonl_path: str,
    image_dir: str,
    config: ValidationConfig | None = None,
) -> tuple[list[dict], list[dict], list[ValidationResult]]:
    """
    Validate all pairs in a JSONL file.
    Returns (valid_pairs, invalid_pairs, all_results).
    """
    _image_dir = Path(image_dir)
    config = config or ValidationConfig()
    valid, invalid = [], []
    results = []

    with open(jsonl_path) as f:
        pairs = [json.loads(line) for line in f]

    for pair in pairs:
        result = validate_pair(pair, _image_dir, config)
        results.append(result)
        if result.passed:
            valid.append(pair)
        else:
            invalid.append(pair)
            print(f"FAIL {pair['pair_id']}: {result.failure_reasons} | {result.details}")

    print(f"\nValidation: {len(valid)}/{len(pairs)} passed "
          f"({len(invalid)} rejected)")
    return valid, invalid, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate generated edit pairs.")
    parser.add_argument(
        "--jsonl", type=Path, required=True,
        help="Path to pairs.jsonl produced by generate.py",
    )
    parser.add_argument(
        "--image-dir", type=Path, required=True,
        help="Directory containing the rendered images (pairs.jsonl's images/ folder)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write pairs_valid.jsonl and pairs_invalid.jsonl "
             "(defaults to the same directory as --jsonl)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or args.jsonl.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    valid, invalid, _ = validate_dataset(str(args.jsonl), str(args.image_dir))

    valid_path = out_dir / "pairs_valid.jsonl"
    invalid_path = out_dir / "pairs_invalid.jsonl"

    with open(valid_path, "w", encoding="utf-8") as f:
        for pair in valid:
            f.write(json.dumps(pair) + "\n")

    with open(invalid_path, "w", encoding="utf-8") as f:
        for pair in invalid:
            f.write(json.dumps(pair) + "\n")

    print(f"Valid   → {valid_path}")
    print(f"Invalid → {invalid_path}")