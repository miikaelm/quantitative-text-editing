"""
templates/scaling.py — HTML template builder for scaling edit pairs.

build_scaling_html(spec) takes a ScalingEditSpec and returns (source_html, target_html).
The two documents have identical text content, styling, and layout; they differ only in
the font-size of the text element. Both versions center the text on the canvas.
"""

from gen_pipeline.specs.scaling import ScalingEditSpec


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _parse_font_style(font_style: str) -> tuple[str, str]:
    """Parse 'bold italic' / 'italic' / 'bold' / 'normal' → (css font-style, css font-weight)."""
    lower = font_style.lower()
    css_style = "italic" if "italic" in lower else "normal"
    css_weight = "bold" if "bold" in lower else "normal"
    return css_style, css_weight


def _body_style(bg: str) -> str:
    return (
        f"margin:0; padding:20px; background:{bg}; display:flex;"
        "justify-content:center; align-items:center; height:100vh;"
        "box-sizing:border-box;"
    )


def _element_style(spec: ScalingEditSpec, size_px: int) -> str:
    css_style, css_weight = _parse_font_style(spec.font_style)
    return (
        f"font-size:{size_px}px; font-family:{spec.font_family};"
        f"color:{spec.font_color}; font-style:{css_style}; font-weight:{css_weight};"
        f"letter-spacing:{spec.letter_spacing};"
        "margin:0;"
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_scaling_html(spec: ScalingEditSpec) -> tuple[str, str]:
    """
    Build source and target HTML from a ScalingEditSpec.

    Returns:
        (source_html, target_html) — identical text, layout, and styling, differing
        only in the font-size of the text element (old_size_px vs new_size_px).
    """
    def _html(size_px: int) -> str:
        return (
            "<!DOCTYPE html>"
            "<html><body style=\"{body}\">"
            "<{el} style=\"{el_style}\">{text}</{el}>"
            "</body></html>"
        ).format(
            body=_body_style(spec.bg),
            el=spec.element,
            el_style=_element_style(spec, size_px),
            text=spec.text_content,
        )

    return _html(spec.old_size_px), _html(spec.new_size_px)
