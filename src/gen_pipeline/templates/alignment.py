"""
templates/alignment.py — HTML template builder for alignment edit pairs.

build_alignment_html(spec) takes an AlignmentEditSpec and returns (source_html, target_html).
The two documents have identical text content and styling; they differ only in
the flexbox alignment of the body, which controls where the text sits on the canvas.

Position grid (justify-content × align-items):

    top-left      top-center      top-right
    middle-left   center          middle-right
    bottom-left   bottom-center   bottom-right
"""

from gen_pipeline.specs.alignment import AlignmentEditSpec


# ---------------------------------------------------------------------------
# Position → CSS flexbox alignment mapping
# ---------------------------------------------------------------------------

# Maps position name → (justify-content, align-items)
POSITION_CSS: dict[str, tuple[str, str]] = {
    "top-left":      ("flex-start", "flex-start"),
    "top-center":    ("center",     "flex-start"),
    "top-right":     ("flex-end",   "flex-start"),
    "middle-left":   ("flex-start", "center"),
    "center":        ("center",     "center"),
    "middle-right":  ("flex-end",   "center"),
    "bottom-left":   ("flex-start", "flex-end"),
    "bottom-center": ("center",     "flex-end"),
    "bottom-right":  ("flex-end",   "flex-end"),
}


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _body_style(position: str, bg: str) -> str:
    justify, align = POSITION_CSS[position]
    return (
        f"margin:0; padding:20px; background:{bg}; display:flex;"
        f"justify-content:{justify}; align-items:{align}; height:100vh;"
        "box-sizing:border-box;"
    )


def _parse_font_style(font_style: str) -> tuple[str, str]:
    """Parse 'bold italic' / 'italic' / 'bold' / 'normal' → (css font-style, css font-weight)."""
    lower = font_style.lower()
    css_style = "italic" if "italic" in lower else "normal"
    css_weight = "bold" if "bold" in lower else "normal"
    return css_style, css_weight


def _element_style(spec: AlignmentEditSpec) -> str:
    css_style, css_weight = _parse_font_style(spec.font_style)
    return (
        f"font-size:{spec.font_size}; font-family:{spec.font_family};"
        f"color:{spec.font_color}; font-style:{css_style}; font-weight:{css_weight};"
        f"letter-spacing:{spec.letter_spacing};"
        "margin:0;"
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_alignment_html(spec: AlignmentEditSpec) -> tuple[str, str]:
    """
    Build source and target HTML from an AlignmentEditSpec.

    Returns:
        (source_html, target_html) — identical text and styling, differing only
        in the body flexbox alignment (start_position vs end_position).
    """
    el_style = _element_style(spec)

    def _html(position: str) -> str:
        return (
            "<!DOCTYPE html>"
            "<html><body style=\"{body}\">"
            "<{el} style=\"{el_style}\">{text}</{el}>"
            "</body></html>"
        ).format(
            body=_body_style(position, spec.bg),
            el=spec.element,
            el_style=el_style,
            text=spec.text_content,
        )

    return _html(spec.start_position), _html(spec.end_position)
