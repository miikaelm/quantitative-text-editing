"""
generate.py — Orchestrator: pairs -> render -> write JSONL.

Thin pipeline driver. Pair generation logic lives in pair_builder.py;
this file only wires rendering and JSONL writing.

To add a new edit type:
    1. Add the edit type to gen_pipeline/layouts.py (supported_edits)
    2. Add a builder function in gen_pipeline/pair_builder.py
"""

import argparse
import json
import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass, field

# Make src/ importable so sibling packages resolve regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from gen_pipeline.render import Renderer, RenderConfig
from gen_pipeline.build_pairs import EditPair
from gen_pipeline.pair_builder import build_edit_pairs, build_layout_pairs
from utils.ocr import find_text_bbox


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GenerateConfig:
    edit_type: str = "color"
    # Explicit output directory. When set, images go to output_dir/images/ and
    # JSONL to output_dir/pairs.jsonl. Overrides output_root + edit_type construction.
    output_dir: Path | None = None
    # Fallback root when output_dir is not set.
    output_root: Path = Path("data")
    render: RenderConfig = field(default_factory=RenderConfig)


# ---------------------------------------------------------------------------
# Rendering + JSONL writing
# ---------------------------------------------------------------------------
# Typography post-render reference measurement
# ---------------------------------------------------------------------------

def _add_typography_reference_measurements(record: dict, src_path, tgt_path) -> None:
    """
    Compute pixel-based reference measurements for font_weight and font_style
    subcategories and store them in record["metadata"].

    Called after rendering so that the evaluator can compare source/target
    reference values against the model output without needing the target image.
    """
    subcategory = record["metadata"].get("typography_subcategory")
    if subcategory not in ("font_weight", "font_style"):
        return

    try:
        import numpy as np
        from PIL import Image
        from evaluation.metrics.typography import (
            binarize_text_region,
            compute_stroke_width,
            compute_shear_angle,
        )
    except ImportError:
        return  # silently skip if dependencies are missing

    src_bbox = record["metadata"].get("source_bbox")
    tgt_bbox = record["metadata"].get("target_bbox")
    if src_bbox is None or tgt_bbox is None:
        return

    src_img = np.array(Image.open(src_path).convert("RGB"))
    tgt_img = np.array(Image.open(tgt_path).convert("RGB"))

    if subcategory == "font_weight":
        src_bin = binarize_text_region(src_img, src_bbox)
        tgt_bin = binarize_text_region(tgt_img, tgt_bbox)
        record["metadata"]["source_stroke_width"] = round(compute_stroke_width(src_bin), 3)
        record["metadata"]["target_stroke_width"] = round(compute_stroke_width(tgt_bin), 3)
    elif subcategory == "font_style":
        src_bin = binarize_text_region(src_img, src_bbox)
        tgt_bin = binarize_text_region(tgt_img, tgt_bbox)
        record["metadata"]["source_shear_angle"] = round(compute_shear_angle(src_bin), 3)
        record["metadata"]["target_shear_angle"] = round(compute_shear_angle(tgt_bin), 3)


# ---------------------------------------------------------------------------

async def generate_pairs(
    pairs: list[EditPair],
    config: GenerateConfig,
) -> Path:
    """Render all pairs and write a JSONL file. Returns the JSONL path."""
    if config.output_dir is not None:
        base_dir = config.output_dir
    else:
        base_dir = config.output_root / config.edit_type

    image_dir = base_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = base_dir / "pairs.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    async with Renderer(config.render) as renderer:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                src_result, tgt_result = await renderer.render_pair(
                    source_html=pair.source_html,
                    target_html=pair.target_html,
                    output_dir=image_dir,
                    pair_id=pair.pair_id,
                )

                record = pair.to_record()
                record["source_image"] = str(src_result.image_path)
                record["target_image"] = str(tgt_result.image_path)

                text_content = pair.metadata.get("text_content")
                if text_content:
                    src_bbox = find_text_bbox(src_result.image_path, text_content)
                    tgt_bbox = find_text_bbox(tgt_result.image_path, text_content)
                    if src_bbox:
                        record["metadata"]["source_bbox"] = src_bbox
                    else:
                        print(f"  [{pair.pair_id}] WARNING: OCR could not locate '{text_content}' in source image")
                    if tgt_bbox:
                        record["metadata"]["target_bbox"] = tgt_bbox
                    else:
                        print(f"  [{pair.pair_id}] WARNING: OCR could not locate '{text_content}' in target image")

                # For typography edits that need pixel-based reference measurements,
                # compute and store them now while we have access to both rendered images.
                if record.get("edit_type") == "typography":
                    _add_typography_reference_measurements(record, src_result.image_path, tgt_result.image_path)

                if src_result.errors or tgt_result.errors:
                    record["render_errors"] = src_result.errors + tgt_result.errors

                f.write(json.dumps(record) + "\n")
                print(f"  [{pair.pair_id}] {src_result.image_path.name}  {tgt_result.image_path.name}")

    return jsonl_path


def generate_pairs_sync(
    pairs: list[EditPair],
    config: GenerateConfig,
) -> Path:
    """Synchronous wrapper for generate_pairs."""
    return asyncio.run(generate_pairs(pairs, config))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edit pairs and render images.")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write pairs.jsonl and images/ into (e.g. data/color/batch_01)",
    )
    parser.add_argument(
        "--edit-type", default="color",
        help=(
            "Edit type to generate (default: color). "
            "Supported: color, alignment, scaling, typography, rotation."
        ),
    )
    parser.add_argument(
        "--count", type=int, required=True,
        help="Number of pairs to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducible generation.",
    )
    parser.add_argument(
        "--layout", default=None,
        help=(
            "Layout name (e.g. title_byline, split_panel). "
            "When set, generates pairs for that layout sampling edit types randomly. "
            "Mutually exclusive with --edit-type."
        ),
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Layout-first mode: --layout overrides --edit-type
    # ---------------------------------------------------------------------------
    if args.layout is not None:
        config = GenerateConfig(edit_type="mixed", output_dir=args.output_dir)
        pairs = build_layout_pairs(args.layout, count=args.count, seed=args.seed)
        print(f"Generating {len(pairs)} pairs for layout '{args.layout}' -> {args.output_dir}")
        jsonl_path = generate_pairs_sync(pairs, config)
        print(f"\nDone. JSONL written to: {jsonl_path}")
        sys.exit(0)

    supported = "color, alignment, scaling, typography, rotation"
    if args.edit_type not in ("color", "alignment", "scaling", "typography", "rotation"):
        print(f"Unknown edit type: {args.edit_type!r}. Supported: {supported}")
        sys.exit(1)

    config = GenerateConfig(edit_type=args.edit_type, output_dir=args.output_dir)
    pairs = build_edit_pairs(args.edit_type, count=args.count, seed=args.seed)

    print(f"Generating {len(pairs)} {args.edit_type} pairs -> {args.output_dir}")
    jsonl_path = generate_pairs_sync(pairs, config)
    print(f"\nDone. JSONL written to: {jsonl_path}")
