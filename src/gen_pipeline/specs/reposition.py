"""
specs/reposition.py — Reposition edit specs and spec generator.

generate_reposition_specs() returns a list of RepositionEditSpec instances.
Each spec describes moving text from start_position to end_position using
named grid positions (top-left, center, bottom-right, etc.).

For now: 3 hardcoded examples for pipeline verification.
The real spec set will be generated separately.
"""

from dataclasses import dataclass

from gen_pipeline.specs.base import EditSpec
from gen_pipeline.templates.base import classify_background


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class RepositionEditSpec(EditSpec):
    """Edit spec for a reposition: text moves from start_position to end_position."""
    element: str = "h1"
    start_position: str = "center"
    end_position: str = "top-left"
    font_size: str = "64px"
    font_color: str = "#333333"
    # font_style: "normal" | "italic" | "bold" | "bold italic"
    font_style: str = "normal"
    bg: str = "#f5f5f5"


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

# Maps target (end) position to a natural-language move instruction.
# The start position is implicit — the model should move text to the target.
_INSTRUCTION_TEMPLATES: dict[str, str] = {
    "top-left":      "Move the text to the top-left corner",
    "top-center":    "Move the text to the top",
    "top-right":     "Move the text to the top-right corner",
    "middle-left":   "Align the text to the left",
    "center":        "Center the text",
    "middle-right":  "Align the text to the right",
    "bottom-left":   "Move the text to the bottom-left corner",
    "bottom-center": "Move the text to the bottom",
    "bottom-right":  "Move the text to the bottom-right corner",
}


def _make_instruction(end_position: str) -> str:
    return _INSTRUCTION_TEMPLATES.get(end_position, f"Move the text to the {end_position}")


# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------

def _make_reposition_spec(
    index: int,
    element: str,
    start_position: str,
    end_position: str,
    text: str,
    font_size: str = "64px",
    font_color: str = "#333333",
    font_style: str = "normal",
    bg: str = "#f5f5f5",
) -> RepositionEditSpec:
    pair_id = f"reposition_{index:03d}"
    return RepositionEditSpec(
        pair_id=pair_id,
        edit_type="reposition",
        instruction=_make_instruction(end_position),
        text_content=text,
        element=element,
        start_position=start_position,
        end_position=end_position,
        font_size=font_size,
        font_color=font_color,
        font_style=font_style,
        bg=bg,
        metadata={
            "element": element,
            "property": "position",
            "old_value": start_position,
            "new_value": end_position,
            "text_content": text,
            "background_css": bg,
            "background_type": classify_background(bg),
            "font_size": font_size,
            "font_color": font_color,
            "font_style": font_style,
        },
    )


# ---------------------------------------------------------------------------
# Public entry point — 3 hardcoded examples for pipeline verification
# ---------------------------------------------------------------------------

def generate_reposition_specs() -> list[RepositionEditSpec]:
    """Return a small set of reposition specs for pipeline end-to-end verification."""
    return [
        # 1. Diagonal move: center → top-left, solid background
        _make_reposition_spec(
            1, "h1", "center", "top-left",
            "MINIMALIST",
            font_size="64px", font_color="#333333", font_style="normal",
            bg="#f5f5f5",
        ),
        # 2. Corner to bottom: top-right → bottom-center, dark background, bold
        _make_reposition_spec(
            2, "h2", "top-right", "bottom-center",
            "BOLD MOVE",
            font_size="48px", font_color="#ffffff", font_style="bold",
            bg="#1a1a2e",
        ),
        # 3. Horizontal move: middle-left → middle-right, gradient background, italic
        _make_reposition_spec(
            3, "p", "middle-left", "middle-right",
            "Side to side",
            font_size="36px", font_color="#2d3748", font_style="italic",
            bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        ),
    ]
