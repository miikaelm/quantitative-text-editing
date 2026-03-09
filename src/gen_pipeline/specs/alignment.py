"""
specs/alignment.py — Alignment edit specs and spec generator.

generate_alignment_specs() returns a list of AlignmentEditSpec instances.
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
class AlignmentEditSpec(EditSpec):
    """Edit spec for an alignment change: text moves from start_position to end_position."""
    element: str = "h1"
    start_position: str = "center"
    end_position: str = "top-left"
    font_size: str = "64px"
    font_color: str = "#333333"
    # font_style: "normal" | "italic" | "bold" | "bold italic"
    font_style: str = "normal"
    # font_family: any CSS font-family stack, e.g. "'Times New Roman', serif"
    font_family: str = "Arial, sans-serif"
    # letter_spacing: CSS letter-spacing value, e.g. "normal", "0.05em", "-0.02em"
    letter_spacing: str = "normal"
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

def _make_alignment_spec(
    index: int,
    element: str,
    start_position: str,
    end_position: str,
    text: str,
    font_size: str = "64px",
    font_color: str = "#333333",
    font_style: str = "normal",
    font_family: str = "Arial, sans-serif",
    letter_spacing: str = "normal",
    bg: str = "#f5f5f5",
) -> AlignmentEditSpec:
    pair_id = f"alignment_{index:03d}"
    return AlignmentEditSpec(
        pair_id=pair_id,
        edit_type="alignment",
        instruction=_make_instruction(end_position),
        text_content=text,
        element=element,
        start_position=start_position,
        end_position=end_position,
        font_size=font_size,
        font_color=font_color,
        font_style=font_style,
        font_family=font_family,
        letter_spacing=letter_spacing,
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
            "font_family": font_family,
            "letter_spacing": letter_spacing,
        },
    )


# ---------------------------------------------------------------------------
# Public entry point — 3 hardcoded examples for pipeline verification
# ---------------------------------------------------------------------------

def generate_alignment_specs() -> list[AlignmentEditSpec]:
    """Return a small set of alignment specs for pipeline end-to-end verification."""
    return [
        # 1. Diagonal move: center → top-left, solid background, Arial
        _make_alignment_spec(
            1, "h1", "center", "top-left",
            "MINIMALIST",
            font_size="64px", font_color="#333333", font_style="normal",
            font_family="Arial, sans-serif", letter_spacing="normal",
            bg="#f5f5f5",
        ),
        # 2. Corner to bottom: top-right → bottom-center, dark background, bold, Times New Roman
        _make_alignment_spec(
            2, "h2", "top-right", "bottom-center",
            "BOLD MOVE",
            font_size="48px", font_color="#ffffff", font_style="bold",
            font_family="'Times New Roman', serif", letter_spacing="0.05em",
            bg="#1a1a2e",
        ),
        # 3. Horizontal move: middle-left → middle-right, gradient, italic, Georgia
        _make_alignment_spec(
            3, "p", "middle-left", "middle-right",
            "Side to side",
            font_size="36px", font_color="#2d3748", font_style="italic",
            font_family="Georgia, serif", letter_spacing="normal",
            bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        ),
        # 4. Bottom to top: bottom-left → top-center, Courier New monospace, wide spacing
        _make_alignment_spec(
            4, "h1", "bottom-left", "top-center",
            "TERMINAL",
            font_size="52px", font_color="#00ff41", font_style="normal",
            font_family="'Courier New', monospace", letter_spacing="0.12em",
            bg="#0d0d0d",
        ),
        # 5. Center to bottom-right, Verdana, tight spacing
        _make_alignment_spec(
            5, "h2", "center", "bottom-right",
            "Compact",
            font_size="40px", font_color="#1a202c", font_style="normal",
            font_family="Verdana, sans-serif", letter_spacing="-0.02em",
            bg="#edf2f7",
        ),
        # 6. Top-center to middle-left, Trebuchet MS, bold italic
        _make_alignment_spec(
            6, "h1", "top-center", "middle-left",
            "DYNAMIC",
            font_size="56px", font_color="#e53e3e", font_style="bold italic",
            font_family="'Trebuchet MS', sans-serif", letter_spacing="0.08em",
            bg="#fff5f5",
        ),
    ]
