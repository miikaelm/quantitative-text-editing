"""
evaluate.py — Post-inference evaluation for model-generated image edits.

Takes a manifest JSONL (produced by the inference runner) and evaluates
each generated output against ground truth using the same metric primitives
as validate.py, but with evaluation-appropriate thresholds and richer
reporting (per-pair scores, per-edit-type aggregates, failure mode breakdown).

Usage:
    python src/evaluation/evaluate.py --manifest results/run_01/manifest.jsonl

Manifest schema (one JSON object per line):
    {
        "index": 0,
        "pair_id": "color_001",
        "output_image": "results/run_01/images/color_001_out.png",
        "source_image": "data/color/test/images/color_001_source.png",
        "ground_truth": "data/color/test/images/color_001_target.png",   # optional
        "elapsed_seconds": 4.12,
        "edit_type": "color",
        "instruction": "Change the title color to #3B82F6",
        "metadata": { ... }                                               # same as generation metadata
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluation.metrics.color import evaluate_color_edit, ColorMeasurement
from evaluation.metrics.preservation import check_unintended_modifications
from evaluation.metrics.reposition import evaluate_reposition_edit, RepositionMeasurement
from utils.ocr import find_text_bbox

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Thresholds for evaluation grading (looser than pipeline validation)."""

    # Color edit thresholds — tiered grading
    color_delta_e_excellent: float = 3.0    # perceptually indistinguishable
    color_delta_e_good: float = 7.0         # noticeable but acceptable
    color_delta_e_poor: float = 15.0        # clearly wrong (above = fail)

    color_ecr_min: float = 0.6              # minimum edit completion ratio
    color_ecr_max: float = 1.4              # maximum (overshoot tolerance)

    # Reposition thresholds (pixel distance)
    reposition_px_excellent: float = 3.0
    reposition_px_good: float = 10.0
    reposition_px_poor: float = 25.0

    # Reposition ECR adjustments (applied when ECR is available):
    #   - ECR < ecr_poor AND pixel grade is "poor" → downgrade to "fail"
    #     (element didn't actually move enough; closeness was coincidental)
    #   - ECR >= ecr_good AND pixel grade is "fail" → upgrade to "good"
    #   - ECR >= ecr_poor AND pixel grade is "fail" → upgrade to "poor"
    reposition_ecr_good: float = 0.9
    reposition_ecr_poor: float = 0.5

    # Scaling thresholds (ratio error)
    scaling_ratio_excellent: float = 0.05   # within 5%
    scaling_ratio_good: float = 0.15
    scaling_ratio_poor: float = 0.30

    # Content thresholds (character error rate)
    content_cer_excellent: float = 0.0      # exact match
    content_cer_good: float = 0.1
    content_cer_poor: float = 0.3


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """Evaluation result for a single manifest entry."""
    pair_id: str
    edit_type: str
    grade: str                          # "excellent" | "good" | "poor" | "fail" | "skip"
    scores: dict[str, float | str]      # metric name → value
    checks: dict[str, bool]             # pass/fail per check
    details: dict[str, Any]             # extra info for debugging
    has_ground_truth: bool = True
    elapsed_seconds: float | None = None

    @property
    def passed(self) -> bool:
        return self.grade in ("excellent", "good", "poor")

    @property
    def failure_reasons(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


@dataclass
class AggregateResult:
    """Aggregated evaluation results across an entire manifest."""
    total: int = 0
    by_grade: dict[str, int] = field(default_factory=lambda: {
        "excellent": 0, "good": 0, "poor": 0, "fail": 0, "skip": 0
    })
    by_edit_type: dict[str, dict[str, int]] = field(default_factory=dict)
    score_distributions: dict[str, list[float]] = field(default_factory=dict)

    def add(self, result: PairResult) -> None:
        self.total += 1
        self.by_grade[result.grade] += 1

        if result.edit_type not in self.by_edit_type:
            self.by_edit_type[result.edit_type] = {
                "excellent": 0, "good": 0, "poor": 0, "fail": 0, "skip": 0
            }
        self.by_edit_type[result.edit_type][result.grade] += 1

        for metric_name, value in result.scores.items():
            if isinstance(value, (int, float)):
                self.score_distributions.setdefault(metric_name, []).append(value)

    def summary(self) -> dict:
        """Return a JSON-serialisable summary."""
        stats = {}
        for metric, values in self.score_distributions.items():
            arr = np.array(values)
            stats[metric] = {
                "mean": round(float(np.mean(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "n": len(values),
            }
        return {
            "total": self.total,
            "by_grade": dict(self.by_grade),
            "by_edit_type": {k: dict(v) for k, v in self.by_edit_type.items()},
            "score_statistics": stats,
        }


# ---------------------------------------------------------------------------
# Image loading helper
# ---------------------------------------------------------------------------

def _load_image(path_str: str, data_root: Path | None = None) -> np.ndarray | None:
    """Load an image as RGB numpy array, resolving relative paths."""

    p = Path(path_str)
    if not p.is_absolute() and data_root is not None:
        p = data_root / p
    if not p.exists():
        log.warning(f"Image not found: {p}")
        return None
    return np.array(Image.open(p).convert("RGB"))


# ---------------------------------------------------------------------------
# Grade helper
# ---------------------------------------------------------------------------

def _grade(value: float, excellent: float, good: float, poor: float, lower_is_better: bool = True) -> str:
    """Assign a grade based on thresholds."""
    if lower_is_better:
        if value <= excellent:
            return "excellent"
        if value <= good:
            return "good"
        if value <= poor:
            return "poor"
        return "fail"
    else:
        # higher is better (e.g. accuracy)
        if value >= excellent:
            return "excellent"
        if value >= good:
            return "good"
        if value >= poor:
            return "poor"
        return "fail"


# ---------------------------------------------------------------------------
# Per-edit-type evaluation
# ---------------------------------------------------------------------------

def _evaluate_color(
    source_img: np.ndarray,
    output_img: np.ndarray,
    metadata: dict,
    config: EvalConfig,
) -> PairResult:
    """Evaluate a color edit: model output vs expected target color."""
    checks: dict[str, bool] = {}
    scores: dict[str, float | str] = {}
    details: dict[str, Any] = {}

    # Require bbox
    if "source_bbox" not in metadata:
        return PairResult(
            pair_id="", edit_type="color", grade="skip",
            scores={}, checks={"bbox_available": False},
            details={"error": "source_bbox missing in metadata"},
        )
    checks["bbox_available"] = True

    sb = metadata["source_bbox"]
    bbox = (sb["x"], sb["y"], sb["width"], sb["height"])
    target_hex = metadata["new_value"]
    planned_delta_e = metadata.get("planned_delta_e", None)

    measurement: ColorMeasurement = evaluate_color_edit(
        source_image=source_img,
        output_image=output_img,
        bbox=bbox,
        target_color_hex=target_hex,
        planned_delta_e=planned_delta_e,
    )

    # Scores
    scores["delta_e"] = round(measurement.delta_e, 4)
    if measurement.edit_completion_ratio is not None:
        scores["edit_completion_ratio"] = round(measurement.edit_completion_ratio, 4)

    # Details
    details["measured_color"] = measurement.measured_hex
    details["target_color"] = measurement.target_hex
    details["old_color"] = metadata.get("old_value", "unknown")
    details["exact_match"] = measurement.exact_match

    # Grade on delta_e
    grade = _grade(
        measurement.delta_e,
        excellent=config.color_delta_e_excellent,
        good=config.color_delta_e_good,
        poor=config.color_delta_e_poor,
    )

    # Also check ECR bounds — downgrade if out of range
    if measurement.edit_completion_ratio is not None:
        ecr_ok = config.color_ecr_min <= measurement.edit_completion_ratio <= config.color_ecr_max
        checks["ecr_in_range"] = ecr_ok
        if not ecr_ok and grade != "fail":
            grade = "poor"

    checks["color_acceptable"] = grade != "fail"

    return PairResult(
        pair_id="",  # filled by caller
        edit_type="color",
        grade=grade,
        scores=scores,
        checks=checks,
        details=details,
    )


def _evaluate_reposition(
    source_img: np.ndarray,
    output_img: np.ndarray,
    metadata: dict,
    config: EvalConfig,
) -> PairResult:
    """
    Evaluate a reposition edit.

    Uses OCR (via metadata["_output_image_path"], injected by evaluate_entry)
    to locate the text in the model output, then measures centroid distance
    to the ground-truth target_bbox and computes ECR if source_bbox is available.
    """
    checks: dict[str, bool] = {}
    scores: dict[str, float | str] = {}
    details: dict[str, Any] = {}

    if "target_bbox" not in metadata:
        return PairResult(
            pair_id="", edit_type="reposition", grade="skip",
            scores={}, checks={},
            details={"error": "target_bbox missing from metadata"},
        )

    text_content = metadata.get("text_content")
    img_h, img_w = output_img.shape[:2]

    # Try OCR on the output image to find where text actually ended up
    output_bbox: dict | None = None
    output_path = metadata.get("_output_image_path")
    if output_path and text_content:
        output_bbox = find_text_bbox(output_path, text_content)

    if output_bbox is None:
        checks["ocr_found"] = False
        details["note"] = "OCR could not locate text in output image; reposition grade skipped"
        return PairResult(
            pair_id="", edit_type="reposition", grade="skip",
            scores=scores, checks=checks, details=details,
        )
    checks["ocr_found"] = True

    measurement: RepositionMeasurement = evaluate_reposition_edit(
        target_bbox=metadata["target_bbox"],
        measured_bbox=output_bbox,
        img_width=img_w,
        img_height=img_h,
        original_bbox=metadata.get("source_bbox"),
    )

    scores["centroid_distance"] = measurement.centroid_distance
    scores["normalized_distance"] = measurement.normalized_distance
    if measurement.ecr is not None:
        scores["ecr"] = measurement.ecr

    grade = _grade(
        measurement.centroid_distance,
        excellent=config.reposition_px_excellent,
        good=config.reposition_px_good,
        poor=config.reposition_px_poor,
    )

    # ECR-based grade adjustments (when available):
    #   - pixel grade "fail" but ECR is high → element moved most of the way
    #   - pixel grade "poor"/"good" but ECR is low → coincidental proximity, not a real edit
    if measurement.ecr is not None:
        ecr = measurement.ecr
        checks["ecr_sufficient"] = ecr >= config.reposition_ecr_poor
        if grade == "fail":
            if ecr >= config.reposition_ecr_good:
                grade = "good"
            elif ecr >= config.reposition_ecr_poor:
                grade = "poor"
        elif grade in ("poor", "good") and ecr < config.reposition_ecr_poor:
            grade = "fail"

    details["measured_centroid"] = measurement.measured_centroid
    details["target_centroid"] = measurement.target_centroid
    if measurement.original_centroid is not None:
        details["original_centroid"] = measurement.original_centroid
    if measurement.planned_distance is not None:
        details["planned_distance"] = measurement.planned_distance

    checks["position_acceptable"] = grade != "fail"

    return PairResult(
        pair_id="", edit_type="reposition",
        grade=grade, scores=scores, checks=checks, details=details,
    )


def _evaluate_scaling(
    source_img: np.ndarray,
    output_img: np.ndarray,
    metadata: dict,
    config: EvalConfig,
) -> PairResult:
    """Evaluate a scaling edit. TODO: implement with bounding box area ratio."""
    return PairResult(
        pair_id="", edit_type="scaling", grade="skip",
        scores={}, checks={},
        details={"note": "scaling evaluation not yet implemented"},
    )


def _evaluate_content(
    source_img: np.ndarray,
    output_img: np.ndarray,
    metadata: dict,
    config: EvalConfig,
) -> PairResult:
    """Evaluate a content replacement edit. TODO: implement with OCR + CER."""
    return PairResult(
        pair_id="", edit_type="content", grade="skip",
        scores={}, checks={},
        details={"note": "content evaluation not yet implemented"},
    )


_EVALUATORS = {
    "color": _evaluate_color,
    "reposition": _evaluate_reposition,
    "scaling": _evaluate_scaling,
    "content": _evaluate_content,
}


# ---------------------------------------------------------------------------
# Single entry evaluation
# ---------------------------------------------------------------------------

def evaluate_entry(
    entry: dict,
    config: EvalConfig,
    data_root: Path | None = None,
    check_unintended: bool = True,
) -> PairResult:
    """Evaluate a single manifest entry."""
    pair_id = entry["pair_id"]
    edit_type = entry.get("edit_type", "unknown")
    metadata = entry.get("metadata", {})

    # Load images
    source_img = _load_image(entry["source_image"], data_root)
    output_img = _load_image(entry["output_image"], data_root)

    if source_img is None or output_img is None:
        missing = []
        if source_img is None:
            missing.append("source")
        if output_img is None:
            missing.append("output")
        return PairResult(
            pair_id=pair_id, edit_type=edit_type, grade="skip",
            scores={}, checks={"images_exist": False},
            details={"error": f"missing images: {', '.join(missing)}"},
        )

    # Dispatch to edit-type-specific evaluator
    evaluator = _EVALUATORS.get(edit_type)
    if evaluator is None:
        return PairResult(
            pair_id=pair_id, edit_type=edit_type, grade="skip",
            scores={}, checks={},
            details={"error": f"unknown edit_type: {edit_type}"},
        )

    # Inject image paths so evaluators that need OCR can access them
    eval_metadata = {
        **metadata,
        "_output_image_path": entry.get("output_image"),
        "_source_image_path": entry.get("source_image"),
    }

    result = evaluator(source_img, output_img, eval_metadata, config)
    result.pair_id = pair_id
    result.elapsed_seconds = entry.get("elapsed_seconds")

    # Ground truth comparison (if available)
    if "ground_truth" in entry:
        gt_img = _load_image(entry["ground_truth"], data_root)
        if gt_img is not None and gt_img.shape == output_img.shape:
            gt_mse = float(np.mean((gt_img.astype(np.float64) - output_img.astype(np.float64)) ** 2))
            result.scores["gt_mse"] = round(gt_mse, 4)
            if gt_mse > 0:
                result.scores["gt_psnr"] = round(10 * np.log10(255.0 ** 2 / gt_mse), 2)
            else:
                result.scores["gt_psnr"] = float("inf")
            result.has_ground_truth = True
        else:
            result.has_ground_truth = False
    else:
        result.has_ground_truth = False

    # Unintended modification check
    if check_unintended:
        unintended = check_unintended_modifications(source_img, output_img, metadata)
        result.details["unintended_modifications"] = unintended

    return result


# ---------------------------------------------------------------------------
# Full manifest evaluation
# ---------------------------------------------------------------------------

def evaluate_manifest(
    manifest_path: str,
    config: EvalConfig | None = None,
    data_root: Path | None = None,
    check_unintended: bool = True,
) -> tuple[list[PairResult], AggregateResult]:
    """
    Evaluate all entries in a manifest JSONL.

    Returns:
        results: list of PairResult (one per manifest line)
        aggregate: AggregateResult with summary statistics
    """
    config = config or EvalConfig()
    _manifest_path = Path(manifest_path)

    if data_root is None:
        data_root = _manifest_path.parent

    with open(_manifest_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    results: list[PairResult] = []
    aggregate = AggregateResult()

    for entry in entries:
        result = evaluate_entry(entry, config, data_root, check_unintended)
        results.append(result)
        aggregate.add(result)

    return results, aggregate


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    results: list[PairResult],
    aggregate: AggregateResult,
    output_path: Path,
) -> None:
    """Write detailed evaluation results and summary to a JSON file."""
    report = {
        "summary": aggregate.summary(),
        "pairs": [
            {
                "pair_id": r.pair_id,
                "edit_type": r.edit_type,
                "grade": r.grade,
                "passed": r.passed,
                "scores": r.scores,
                "checks": r.checks,
                "details": r.details,
                "has_ground_truth": r.has_ground_truth,
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(results: list[PairResult], aggregate: AggregateResult) -> None:
    """Pretty-print evaluation results to stdout."""

    def hex_to_ansi(hex_str: str, text: str) -> str:
        if not isinstance(hex_str, str) or not hex_str.startswith("#"):
            return str(hex_str)
        h = hex_str.lstrip("#")
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        return f"\033[48;2;{r};{g};{b}m {text} \033[0m"

    GRADE_COLORS = {
        "excellent": "\033[92m",  # green
        "good":      "\033[93m",  # yellow
        "poor":      "\033[91m",  # red
        "fail":      "\033[31m",  # dark red
        "skip":      "\033[90m",  # gray
    }
    RESET = "\033[0m"

    # Per-pair table
    print("\n--- Per-pair evaluation results ---")
    ID_W = 20
    header = f"{'pair_id':<{ID_W}} {'type':<12} {'grade':<10} {'dE':>8} {'ECR':>8} {'GT PSNR':>9}  extra"
    print(header)
    print("-" * 80)

    for r in results:
        # Truncate pair_id to last ID_W chars so long ids don't break the table
        pid = r.pair_id[-ID_W:] if len(r.pair_id) > ID_W else r.pair_id

        de = r.scores.get("delta_e", "—")
        # reposition stores ECR as "ecr", color stores it as "edit_completion_ratio"
        ecr_val = r.scores.get("edit_completion_ratio", r.scores.get("ecr", None))
        psnr = r.scores.get("gt_psnr", "—")

        de_s   = f"{de:.4f}"    if isinstance(de, float)      else str(de)
        ecr_s  = f"{ecr_val:.4f}" if isinstance(ecr_val, float) else "—"
        psnr_s = f"{psnr:.2f}"  if isinstance(psnr, float)    else str(psnr)

        # Extra info column: color hex swatches or reposition centroids
        if r.edit_type == "color":
            meas = r.details.get("measured_color", "—")
            tgt  = r.details.get("target_color",   "—")
            meas_s = hex_to_ansi(meas, meas) if isinstance(meas, str) else str(meas)
            tgt_s  = hex_to_ansi(tgt,  tgt)  if isinstance(tgt,  str) else str(tgt)
            extra = f"measured={meas_s} target={tgt_s}"
        elif r.edit_type == "reposition":
            mc = r.details.get("measured_centroid", "—")
            tc = r.details.get("target_centroid",   "—")
            mc_s = f"({mc[0]:.0f},{mc[1]:.0f})" if isinstance(mc, (tuple, list)) else str(mc)
            tc_s = f"({tc[0]:.0f},{tc[1]:.0f})" if isinstance(tc, (tuple, list)) else str(tc)
            dist = r.scores.get("centroid_distance", "—")
            dist_s = f"{dist:.1f}px" if isinstance(dist, float) else str(dist)
            extra = f"measured={mc_s} target={tc_s} dist={dist_s}"
        else:
            extra = ""

        gc = GRADE_COLORS.get(r.grade, "")
        grade_s = f"{gc}{r.grade:<10}{RESET}"

        print(f"{pid:<{ID_W}} {r.edit_type:<12} {grade_s} {de_s:>8} {ecr_s:>8} {psnr_s:>9}  {extra}")

        if r.failure_reasons:
            print(f"  ^ failed: {r.failure_reasons}")

        um = r.details.get("unintended_modifications", {})
        if um.get("outside_changed_ratio", 0) > 0.01:
            print(f"  ^ unintended changes: {um['outside_changed_ratio']:.4f} of outside pixels, "
                  f"PSNR={um.get('outside_psnr', '?')}")

    # Summary
    summary = aggregate.summary()
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({aggregate.total} pairs)")
    print(f"{'='*60}")

    for grade in ("excellent", "good", "poor", "fail", "skip"):
        count = summary["by_grade"][grade]
        gc = GRADE_COLORS.get(grade, "")
        pct = count / aggregate.total * 100 if aggregate.total else 0
        bar = "█" * int(pct / 2)
        print(f"  {gc}{grade:<10}{RESET} {count:>4}  ({pct:5.1f}%)  {bar}")

    if summary["by_edit_type"]:
        print("\nBy edit type:")
        for et, grades in summary["by_edit_type"].items():
            et_total = sum(grades.values())
            et_pass = grades["excellent"] + grades["good"] + grades["poor"]
            print(f"  {et:<12} {et_pass}/{et_total} passed  "
                  f"(E={grades['excellent']} G={grades['good']} P={grades['poor']} F={grades['fail']})")

    if summary["score_statistics"]:
        print("\nScore statistics:")
        for metric, stats in summary["score_statistics"].items():
            print(f"  {metric:<24} mean={stats['mean']:.4f}  "
                  f"median={stats['median']:.4f}  std={stats['std']:.4f}  "
                  f"[{stats['min']:.4f}, {stats['max']:.4f}]  n={stats['n']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model-generated image edits against ground truth.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL from inference run")
    parser.add_argument("--data-root", default=None, help="Root dir for resolving relative image paths (defaults to manifest dir)")
    parser.add_argument("--output", default=None, help="Path to write JSON report (optional)")
    parser.add_argument("--no-unintended", action="store_true", help="Skip unintended modification checks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_root = Path(args.data_root) if args.data_root else None

    results, aggregate = evaluate_manifest(
        manifest_path=args.manifest,
        data_root=data_root,
        check_unintended=not args.no_unintended,
    )

    _print_results(results, aggregate)

    if args.output:
        output_path = Path(args.output)
        write_report(results, aggregate, output_path)
        print(f"\nReport saved to {output_path}")

    # Exit with error if any failures
    fail_count = aggregate.by_grade["fail"]
    if fail_count > 0:
        print(f"\n{fail_count} pair(s) failed evaluation.")
        sys.exit(1)


if __name__ == "__main__":
    main()