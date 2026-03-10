"""
specs/scaling.py — Scaling edit specs and spec generator.

generate_scaling_specs() returns a list of ScalingEditSpec instances.
Each spec describes changing the font size of a text element, either by
a relative factor ("30% larger") or to an absolute target ("set to 96px").

For now: 6 hardcoded examples for pipeline verification.
The real spec set will be generated separately.
"""

from dataclasses import dataclass

from gen_pipeline.specs.base import EditSpec
from gen_pipeline.templates.base import classify_background


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScalingEditSpec(EditSpec):
    """Edit spec for a font-size change: text scales from old_size_px to new_size_px."""
    element: str = "h1"
    old_size_px: int = 64
    new_size_px: int = 80
    font_color: str = "#333333"
    # font_style: "normal" | "italic" | "bold" | "bold italic"
    font_style: str = "normal"
    # font_family: any CSS font-family stack
    font_family: str = "Arial, sans-serif"
    # letter_spacing: CSS letter-spacing value, e.g. "normal", "0.05em"
    letter_spacing: str = "normal"
    bg: str = "#f5f5f5"


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

def _make_instruction(old_px: int, new_px: int) -> str:
    """
    Generate a natural-language instruction for the font-size change.
    Alternates between relative-percentage and absolute-pixel phrasings.
    """
    ratio = new_px / old_px
    pct = round(abs(ratio - 1.0) * 100)

    if ratio > 1.0:
        # Use absolute phrasing for round percentages, relative otherwise
        if pct in (25, 50, 75, 100):
            return f"Increase the font size by {pct}%"
        return f"Set the font size to {new_px}px"
    else:
        if pct in (25, 50, 75):
            return f"Reduce the font size by {pct}%"
        return f"Set the font size to {new_px}px"


# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------

def _make_scaling_spec(
    index: int,
    element: str,
    old_size_px: int,
    new_size_px: int,
    text: str,
    font_color: str = "#333333",
    font_style: str = "normal",
    font_family: str = "Arial, sans-serif",
    letter_spacing: str = "normal",
    bg: str = "#f5f5f5",
) -> ScalingEditSpec:
    pair_id = f"scaling_{index:03d}"
    scale_factor = round(new_size_px / old_size_px, 4)
    return ScalingEditSpec(
        pair_id=pair_id,
        edit_type="scaling",
        instruction=_make_instruction(old_size_px, new_size_px),
        text_content=text,
        element=element,
        old_size_px=old_size_px,
        new_size_px=new_size_px,
        font_color=font_color,
        font_style=font_style,
        font_family=font_family,
        letter_spacing=letter_spacing,
        bg=bg,
        metadata={
            "element": element,
            "property": "font-size",
            "old_value": f"{old_size_px}px",
            "new_value": f"{new_size_px}px",
            "scale_factor": scale_factor,
            "text_content": text,
            "background_css": bg,
            "background_type": classify_background(bg),
            "font_color": font_color,
            "font_style": font_style,
            "font_family": font_family,
            "letter_spacing": letter_spacing,
        },
    )


# ---------------------------------------------------------------------------
# Public entry point — 6 hardcoded examples for pipeline verification
# ---------------------------------------------------------------------------

def generate_scaling_specs() -> list[ScalingEditSpec]:
    """Return a small set of scaling specs for pipeline end-to-end verification."""
    return [
        # 1. 30% increase, solid background, Arial
        _make_scaling_spec(
            1, "h1", 64, 83, "SCALE UP",
            font_color="#333333", font_style="normal",
            font_family="Arial, sans-serif", letter_spacing="normal",
            bg="#f5f5f5",
        ),
        # 2. 50% increase (absolute instruction), dark background, bold, Times New Roman
        _make_scaling_spec(
            2, "h1", 64, 96, "BIGGER",
            font_color="#ffffff", font_style="bold",
            font_family="'Times New Roman', serif", letter_spacing="0.05em",
            bg="#1a1a2e",
        ),
        # 3. 25% decrease, gradient background, italic, Georgia
        _make_scaling_spec(
            3, "h2", 64, 48, "smaller",
            font_color="#2d3748", font_style="italic",
            font_family="Georgia, serif", letter_spacing="normal",
            bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        ),
        # 4. 2x increase (100% larger), dark terminal look, Courier New
        _make_scaling_spec(
            4, "h1", 32, 64, "DOUBLE",
            font_color="#00ff41", font_style="normal",
            font_family="'Courier New', monospace", letter_spacing="0.12em",
            bg="#0d0d0d",
        ),
        # 5. 25% decrease, light background, Verdana, tight spacing
        _make_scaling_spec(
            5, "p", 48, 36, "Compact",
            font_color="#1a202c", font_style="normal",
            font_family="Verdana, sans-serif", letter_spacing="-0.02em",
            bg="#edf2f7",
        ),
        # 6. Absolute target (set to 80px), Trebuchet MS, bold italic
        _make_scaling_spec(
            6, "h1", 56, 80, "RESIZE",
            font_color="#e53e3e", font_style="bold italic",
            font_family="'Trebuchet MS', sans-serif", letter_spacing="0.08em",
            bg="#fff5f5",
        ),
    ]
