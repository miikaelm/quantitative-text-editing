"""
specs/typography.py — Typography edit specs and spec generator.

generate_typography_specs() returns a list of TypographyEditSpec instances.
Each spec describes changing a single typographic CSS property of a text element.

Supported subcategories (stored in typography_subcategory):
    font_weight    — font-weight: normal → bold (or reverse)
    font_style     — font-style: normal → italic (or reverse)
    font_family    — font-family: one typeface → another
    letter_spacing — letter-spacing: normal → expanded / condensed

For now: 8 hardcoded examples for pipeline verification.
"""

from dataclasses import dataclass

from gen_pipeline.specs.base import EditSpec
from gen_pipeline.templates.base import classify_background


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class TypographyEditSpec(EditSpec):
    """Edit spec for a typographic CSS property change."""
    element: str = "h1"
    # Which sub-type of typography edit this is.
    # One of: "font_weight" | "font_style" | "font_family" | "letter_spacing"
    typography_subcategory: str = "font_weight"
    # The CSS property being changed (e.g. "font-weight", "font-style", ...)
    css_property: str = "font-weight"
    # Old and new CSS values for that property
    old_value: str = "normal"
    new_value: str = "bold"
    # Fixed properties (unchanged across source / target)
    font_size_px: int = 64
    # Base font family (source font for font_family edits; fixed for all others)
    font_family: str = "Arial, sans-serif"
    font_color: str = "#333333"
    bg: str = "#f5f5f5"


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

_WEIGHT_INSTRUCTIONS = {
    ("normal", "bold"):         "Make the text bold",
    ("bold", "normal"):         "Make the text normal weight",
    ("400", "700"):             "Make the text bold",
    ("700", "400"):             "Make the text normal weight",
}

_STYLE_INSTRUCTIONS = {
    ("normal", "italic"):       "Make the text italic",
    ("italic", "normal"):       "Remove the italic style from the text",
}

_SPACING_INSTRUCTIONS_POS = "Increase the letter spacing"
_SPACING_INSTRUCTIONS_NEG = "Decrease the letter spacing"


def _make_instruction(spec: "TypographyEditSpec") -> str:
    sub = spec.typography_subcategory
    key = (spec.old_value, spec.new_value)
    if sub == "font_weight":
        return _WEIGHT_INSTRUCTIONS.get(key, f"Set the font weight to {spec.new_value}")
    if sub == "font_style":
        return _STYLE_INSTRUCTIONS.get(key, f"Set the font style to {spec.new_value}")
    if sub == "font_family":
        return f"Change the font family to {spec.new_value}"
    if sub == "letter_spacing":
        # Determine direction by comparing numeric-ish values
        def _em_val(v: str) -> float:
            v = v.strip()
            if v == "normal":
                return 0.0
            return float(v.replace("em", "").replace("px", ""))
        delta = _em_val(spec.new_value) - _em_val(spec.old_value)
        return _SPACING_INSTRUCTIONS_POS if delta >= 0 else _SPACING_INSTRUCTIONS_NEG
    return f"Set the {spec.css_property} to {spec.new_value}"


# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------

def _make_typography_spec(
    index: int,
    subcategory: str,
    css_property: str,
    element: str,
    old_value: str,
    new_value: str,
    text: str,
    font_size_px: int = 64,
    font_family: str = "Arial, sans-serif",
    font_color: str = "#333333",
    bg: str = "#f5f5f5",
) -> TypographyEditSpec:
    pair_id = f"typography_{index:03d}"
    spec = TypographyEditSpec(
        pair_id=pair_id,
        edit_type="typography",
        instruction="",          # filled below after spec is constructed
        text_content=text,
        element=element,
        typography_subcategory=subcategory,
        css_property=css_property,
        old_value=old_value,
        new_value=new_value,
        font_size_px=font_size_px,
        font_family=font_family,
        font_color=font_color,
        bg=bg,
        metadata={
            "element": element,
            "typography_subcategory": subcategory,
            "property": css_property,
            "old_value": old_value,
            "new_value": new_value,
            "text_content": text,
            "font_size_px": font_size_px,
            "font_family": font_family,
            "font_color": font_color,
            "background_css": bg,
            "background_type": classify_background(bg),
        },
    )
    spec.instruction = _make_instruction(spec)
    return spec


# ---------------------------------------------------------------------------
# Public entry point — 8 hardcoded examples for pipeline verification
# ---------------------------------------------------------------------------

def generate_typography_specs() -> list[TypographyEditSpec]:
    """Return a small set of typography specs for pipeline end-to-end verification."""
    return [
        # 1. font_weight: normal → bold, Arial, light background
        _make_typography_spec(
            1, "font_weight", "font-weight",
            "h1", "normal", "bold", "TYPEFACE",
            font_size_px=64, font_family="Arial, sans-serif",
            font_color="#333333", bg="#f5f5f5",
        ),
        # 2. font_weight: bold → normal (reverse), white on dark
        _make_typography_spec(
            2, "font_weight", "font-weight",
            "h1", "bold", "normal", "LIGHTER",
            font_size_px=64, font_family="'Times New Roman', serif",
            font_color="#ffffff", bg="#1a1a2e",
        ),
        # 3. font_style: normal → italic, Georgia, gradient background
        _make_typography_spec(
            3, "font_style", "font-style",
            "h2", "normal", "italic", "Emphasis",
            font_size_px=56, font_family="Georgia, serif",
            font_color="#2d3748", bg="linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
        ),
        # 4. font_style: italic → normal (de-italicise), Verdana
        _make_typography_spec(
            4, "font_style", "font-style",
            "h1", "italic", "normal", "UPRIGHT",
            font_size_px=60, font_family="Verdana, sans-serif",
            font_color="#1a202c", bg="#edf2f7",
        ),
        # 5. font_family: Arial → Courier New (sans-serif → monospace)
        _make_typography_spec(
            5, "font_family", "font-family",
            "h1", "Arial, sans-serif", "'Courier New', monospace", "MONO",
            font_size_px=64, font_family="Arial, sans-serif",
            font_color="#00cc66", bg="#0d0d0d",
        ),
        # 6. font_family: Georgia → Arial (serif → sans-serif)
        _make_typography_spec(
            6, "font_family", "font-family",
            "h2", "Georgia, serif", "Arial, sans-serif", "Sans",
            font_size_px=56, font_family="Georgia, serif",
            font_color="#4a5568", bg="#ffffff",
        ),
        # 7. letter_spacing: normal → expanded (0.2em)
        _make_typography_spec(
            7, "letter_spacing", "letter-spacing",
            "h1", "normal", "0.2em", "SPACING",
            font_size_px=60, font_family="Arial, sans-serif",
            font_color="#2d3748", bg="#f7fafc",
        ),
        # 8. letter_spacing: condensed (−0.03em) → expanded (0.12em)
        _make_typography_spec(
            8, "letter_spacing", "letter-spacing",
            "h1", "-0.03em", "0.12em", "EXPAND",
            font_size_px=64, font_family="'Trebuchet MS', sans-serif",
            font_color="#e53e3e", bg="#fff5f5",
        ),
    ]
