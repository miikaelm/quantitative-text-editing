"""
validate.py — Post-rendering validation for generated pairs.

Calls evaluation metric primitives with pass/fail thresholds.
Runs after render.py, before train/test splitting.

Currently implements: color edit validation.
TODO: reposition, scaling, content edit validation.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.metrics.color import evaluate_color_edit, ColorMeasurement


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """Thresholds for pass/fail decisions."""
    # Color edit thresholds
    max_color_delta_e: float = 2.0          # target color must be near-exact in ground truth
    min_color_confidence: float = 0.3       # histogram peak must be ≥30% of total positive shift
    min_peak_pixel_count: int = 20          # need enough text pixels for reliable measurement


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    pair_id: str
    passed: bool
    checks: dict[str, bool]
    details: dict[str, float | str]

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
            passed=False,
            checks={"images_exist": False},
            details={"error": "missing image files"},
        )
    checks["images_exist"] = True

    source_img = np.array(Image.open(source_path).convert("RGB"))
    target_img = np.array(Image.open(target_path).convert("RGB"))

    edit_type = pair["edit_type"]
    metadata = pair["metadata"]

    if edit_type == "color":
        _validate_color(source_img, target_img, metadata, config, checks, details)

    elif edit_type == "reposition":
        # TODO
        details["note"] = "reposition validation not yet implemented"

    elif edit_type == "scaling":
        # TODO
        details["note"] = "scaling validation not yet implemented"

    elif edit_type == "content":
        # TODO
        details["note"] = "content validation not yet implemented"

    passed = all(checks.values())
    return ValidationResult(
        pair_id=pair["pair_id"],
        passed=passed,
        checks=checks,
        details=details,
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

    measurement: ColorMeasurement = evaluate_color_edit(
        source_image=source_img,
        output_image=target_img,
        bbox=bbox,
        target_color_hex=target_hex,
    )

    # Check: color is correct
    checks["edit_applied"] = measurement.delta_e <= config.max_color_delta_e

    # Check: measurement is reliable
    checks["measurement_confident"] = (
        -1
    )

    # Record details for debugging
    details["measured_color"] = measurement.measured_hex
    details["target_color"] = measurement.target_hex
    details["delta_e"] = round(measurement.delta_e, 4)
    details["exact_match"] = measurement.exact_match
    details["old_color"] = metadata.get("old_value", "unknown")


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
    image_dir = Path(image_dir)
    config = config or ValidationConfig()
    valid, invalid = [], []
    results = []

    with open(jsonl_path) as f:
        pairs = [json.loads(line) for line in f]

    for pair in pairs:
        result = validate_pair(pair, image_dir, config)
        results.append(result)
        if result.passed:
            valid.append(pair)
        else:
            invalid.append(pair)
            print(f"FAIL {pair['pair_id']}: {result.failure_reasons} | {result.details}")

    print(f"\nValidation: {len(valid)}/{len(pairs)} passed "
          f"({len(invalid)} rejected)")
    return valid, invalid, results