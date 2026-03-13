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

SUPPORTED_EDIT_TYPES = ("color", "alignment", "scaling", "typography", "rotation")

@dataclass
class GenerateConfig:
    edit_type: str = "color"
    # Explicit output directory. When set, images go to output_dir/images/ and
    # JSONL to output_dir/pairs.jsonl. Overrides output_root + edit_type construction.
    output_dir: Path | None = None
    # Fallback root when output_dir is not set.
    output_root: Path = Path("data")
    render: RenderConfig = field(default_factory=RenderConfig)


@dataclass
class PipelineConfig:
    """
    Top-level config for a multi-type generation run.

    YAML layout::

        output_dir: data/batch_01
        seed: 42
        counts:
            color: 50
            scaling: 20
            typography: 30
        render:
            width: 1024
            height: 1024
            downscale_to: 512
    """
    output_dir: Path = Path("data/batch_01")
    seed: int | None = None
    counts: dict = field(default_factory=lambda: {"color": 20})
    render: RenderConfig = field(default_factory=RenderConfig)


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    render_cfg = RenderConfig()
    if "render" in data:
        r = data["render"]
        render_cfg = RenderConfig(
            width=r.get("width", render_cfg.width),
            height=r.get("height", render_cfg.height),
            device_scale_factor=r.get("device_scale_factor", render_cfg.device_scale_factor),
            downscale_to=r.get("downscale_to", render_cfg.downscale_to),
            disable_animations=r.get("disable_animations", render_cfg.disable_animations),
            default_font=r.get("default_font", render_cfg.default_font),
        )

    return PipelineConfig(
        output_dir=Path(data["output_dir"]),
        seed=data.get("seed"),
        counts=data.get("counts", {"color": 20}),
        render=render_cfg,
    )


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
    """Render all pairs and append to a JSONL file. Returns the JSONL path.

    If the JSONL file already exists the new records are appended; existing
    records are never overwritten. The images/ sub-directory works the same
    way — new images are added alongside any that already exist.
    """
    if config.output_dir is not None:
        base_dir = config.output_dir
    else:
        base_dir = config.output_root / config.edit_type

    image_dir = base_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = base_dir / "pairs.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    async with Renderer(config.render) as renderer:
        with jsonl_path.open("a", encoding="utf-8") as f:
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
# Pipeline runner (multi-type)
# ---------------------------------------------------------------------------

def _count_existing_records(jsonl_path: Path, edit_type: str | None = None) -> int:
    """Count records in a JSONL file, optionally filtered to a specific edit_type."""
    if not jsonl_path.exists():
        return 0
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if edit_type is None:
                count += 1
            else:
                try:
                    record = json.loads(line)
                    if record.get("edit_type") == edit_type:
                        count += 1
                except json.JSONDecodeError:
                    pass
    return count


async def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    """
    Run generation for every edit type listed in config.counts.

    All edit types write into the same output_dir — one shared pairs.jsonl
    and one shared images/ folder. Returns a mapping of edit_type -> jsonl_path.
    """
    results: dict[str, Path] = {}

    jsonl_path = config.output_dir / "pairs.jsonl"

    for edit_type, count in config.counts.items():
        if edit_type not in SUPPORTED_EDIT_TYPES:
            print(f"  [pipeline] WARNING: skipping unknown edit type '{edit_type}'")
            continue

        id_offset = _count_existing_records(jsonl_path, edit_type=edit_type)
        if id_offset:
            print(f"\n[pipeline] Found {id_offset} existing '{edit_type}' records — resuming from id {id_offset + 1:03d}")

        print(f"\n[pipeline] Generating {count} '{edit_type}' pairs ...")
        gen_cfg = GenerateConfig(
            edit_type=edit_type,
            output_dir=config.output_dir,
            render=config.render,
        )
        pairs = build_edit_pairs(edit_type, count=count, seed=config.seed, id_offset=id_offset)
        jsonl_path_out = await generate_pairs(pairs, gen_cfg)
        results[edit_type] = jsonl_path_out
        print(f"[pipeline] '{edit_type}' done -> {jsonl_path_out}")

    return results


def run_pipeline_sync(config: PipelineConfig) -> dict[str, Path]:
    """Synchronous wrapper for run_pipeline."""
    return asyncio.run(run_pipeline(config))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate edit pairs and render images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes (mutually exclusive):\n"
            "  --config FILE          Run a multi-type pipeline from a YAML config.\n"
            "  --edit-type / --count  Generate a single edit type (original behaviour).\n"
            "  --layout               Generate pairs for one layout (original behaviour).\n"
            "\n"
            "Output is always appended — existing pairs.jsonl and images/ are preserved."
        ),
    )

    # --- config-file mode ---
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a YAML PipelineConfig file. Mutually exclusive with --edit-type/--layout.",
    )

    # --- single-type / layout modes (original) ---
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write pairs.jsonl and images/ into (e.g. data/color/batch_01). "
             "Required for --edit-type and --layout modes.",
    )
    parser.add_argument(
        "--edit-type", default="color",
        help=(
            "Edit type to generate (default: color). "
            f"Supported: {', '.join(SUPPORTED_EDIT_TYPES)}."
        ),
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Number of pairs to generate. Required for --edit-type and --layout modes.",
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
    # Config-file mode
    # ---------------------------------------------------------------------------
    if args.config is not None:
        if args.layout is not None:
            parser.error("--config and --layout are mutually exclusive")
        pipeline_cfg = load_pipeline_config(args.config)
        # Allow --seed / --output-dir to override what's in the YAML
        if args.seed is not None:
            pipeline_cfg.seed = args.seed
        if args.output_dir is not None:
            pipeline_cfg.output_dir = args.output_dir
        print(f"Running pipeline from config: {args.config}")
        print(f"  output_dir : {pipeline_cfg.output_dir}")
        print(f"  seed       : {pipeline_cfg.seed}")
        print(f"  counts     : {pipeline_cfg.counts}")
        results = run_pipeline_sync(pipeline_cfg)
        print(f"\nDone. {len(results)} edit type(s) generated:")
        for et, p in results.items():
            print(f"  {et}: {p}")
        sys.exit(0)

    # ---------------------------------------------------------------------------
    # Layout-first mode: --layout overrides --edit-type
    # ---------------------------------------------------------------------------
    if args.layout is not None:
        if args.output_dir is None or args.count is None:
            parser.error("--layout requires --output-dir and --count")
        gen_cfg = GenerateConfig(edit_type="mixed", output_dir=args.output_dir)
        id_offset = _count_existing_records(args.output_dir / "pairs.jsonl")
        if id_offset:
            print(f"Found {id_offset} existing records — resuming from id {id_offset + 1:03d}")
        pairs = build_layout_pairs(args.layout, count=args.count, seed=args.seed, id_offset=id_offset)
        print(f"Generating {len(pairs)} pairs for layout '{args.layout}' -> {args.output_dir}")
        jsonl_path = generate_pairs_sync(pairs, gen_cfg)
        print(f"\nDone. JSONL written to: {jsonl_path}")
        sys.exit(0)

    # ---------------------------------------------------------------------------
    # Single edit-type mode (original behaviour)
    # ---------------------------------------------------------------------------
    if args.output_dir is None or args.count is None:
        parser.error("--edit-type mode requires --output-dir and --count")

    if args.edit_type not in SUPPORTED_EDIT_TYPES:
        parser.error(f"Unknown edit type: {args.edit_type!r}. Supported: {', '.join(SUPPORTED_EDIT_TYPES)}")

    gen_cfg = GenerateConfig(edit_type=args.edit_type, output_dir=args.output_dir)
    id_offset = _count_existing_records(args.output_dir / "pairs.jsonl", edit_type=args.edit_type)
    if id_offset:
        print(f"Found {id_offset} existing '{args.edit_type}' records — resuming from id {id_offset + 1:03d}")
    pairs = build_edit_pairs(args.edit_type, count=args.count, seed=args.seed, id_offset=id_offset)

    print(f"Generating {len(pairs)} {args.edit_type} pairs -> {args.output_dir}")
    jsonl_path = generate_pairs_sync(pairs, gen_cfg)
    print(f"\nDone. JSONL written to: {jsonl_path}")
