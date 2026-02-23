"""
Generate source/target HTML pairs for quantitative text editing.

Writes JSONL records and renders PNG images via render.py.
This module is intentionally minimal — only color-change pairs
are supported right now, hand-crafted for pipeline development.
"""

import json
import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass, field

# Make src/ importable so utils.ocr can be found regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from render import Renderer, RenderConfig
from utils.ocr import find_text_bbox


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EditPair:
    pair_id: str
    edit_type: str
    source_html: str
    target_html: str
    instruction: str
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "edit_type": self.edit_type,
            "source_html": self.source_html,
            "target_html": self.target_html,
            "instruction": self.instruction,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Hard-coded color-change examples (placeholder until LLM generation is wired)
# ---------------------------------------------------------------------------

def _make_color_pair(
    index: int,
    element: str,
    old_hex: str,
    new_hex: str,
    text: str,
    font_size: str = "64px",
    bg: str = "#f5f5f5",
) -> EditPair:
    pair_id = f"color_{index:03d}"
    base_style = (
        "margin:0; background:{bg}; display:flex;"
        "justify-content:center; align-items:center; height:100vh;"
    ).format(bg=bg)
    elem_style = f"font-size:{font_size}; font-family:Arial,sans-serif; color:{{color}};"

    def _html(color: str) -> str:
        return (
            "<!DOCTYPE html>"
            "<html><body style=\"{body}\">"
            "<{el} style=\"{el_style}\">{text}</{el}>"
            "</body></html>"
        ).format(
            body=base_style,
            el=element,
            el_style=elem_style.format(color=color),
            text=text,
        )

    return EditPair(
        pair_id=pair_id,
        edit_type="color",
        source_html=_html(old_hex),
        target_html=_html(new_hex),
        instruction=f"Change the {element} color to {new_hex}",
        metadata={
            "element": element,
            "property": "color",
            "old_value": old_hex,
            "new_value": new_hex,
            "text_content": text,
        },
    )


# Ten hand-crafted color-change pairs covering a range of hues/elements.
HARDCODED_COLOR_PAIRS: list[EditPair] = [
    _make_color_pair(1,  "h1", "#333333", "#3B82F6", "MINIMALIST"),
    _make_color_pair(2,  "h1", "#111111", "#EF4444", "BOLD"),
    _make_color_pair(3,  "h2", "#444444", "#10B981", "Emerald Title", font_size="48px"),
    _make_color_pair(4,  "h1", "#222222", "#F59E0B", "AMBER", bg="#1a1a2e"),
    _make_color_pair(5,  "p",  "#555555", "#8B5CF6", "Body paragraph text sample.", font_size="32px"),
    _make_color_pair(6,  "h1", "#0a0a0a", "#EC4899", "PINK HEADER"),
    _make_color_pair(7,  "h3", "#333333", "#06B6D4", "Cyan subheading", font_size="40px"),
    _make_color_pair(8,  "h1", "#1a1a1a", "#84CC16", "LIME", bg="#0f172a"),
    _make_color_pair(9,  "p",  "#666666", "#F97316", "Orange paragraph content.", font_size="28px"),
    _make_color_pair(10, "h2", "#2d2d2d", "#6366F1", "Indigo heading", font_size="52px"),
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@dataclass
class GenerateConfig:
    edit_type: str = "color"
    output_root: Path = Path("data")
    # If True, writes to data/<edit_type>/test/ instead of data/<edit_type>/
    test_run: bool = False
    render: RenderConfig = field(default_factory=RenderConfig)


async def generate_pairs(
    pairs: list[EditPair],
    config: GenerateConfig,
) -> Path:
    """Render all pairs and write a JSONL file. Returns the JSONL path."""
    subfolder = "test" if config.test_run else ""
    image_dir = config.output_root / config.edit_type / subfolder / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = config.output_root / config.edit_type / subfolder / "pairs.jsonl"
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
# Test entry-point  (mirrors render.py's __main__ block)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = GenerateConfig(
        edit_type="color",
        output_root=Path("data"),
        test_run=True,
    )

    print(f"Generating {len(HARDCODED_COLOR_PAIRS)} color-change pairs → data/color/test/")
    jsonl_path = generate_pairs_sync(HARDCODED_COLOR_PAIRS, config)
    print(f"\nDone. JSONL written to: {jsonl_path}")
