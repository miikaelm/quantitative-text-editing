"""
templates/typography.py — HTML template builder for typography edit pairs.

build_typography_html(spec) takes a TypographyEditSpec and returns (source_html, target_html).
The two documents have identical text content, layout, and all other CSS styling; they
differ only in the single CSS property identified by spec.css_property.

Supported subcategories / properties:
    font_weight    → font-weight
    font_style     → font-style
    font_family    → font-family
    letter_spacing → letter-spacing
"""

from gen_pipeline.specs.typography import TypographyEditSpec


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _body_style(bg: str) -> str:
    return (
        f"margin:0; padding:20px; background:{bg}; display:flex;"
        "justify-content:center; align-items:center; height:100vh;"
        "box-sizing:border-box;"
    )


def _element_style(spec: TypographyEditSpec, use_new_value: bool) -> str:
    """
    Build the full inline style for the text element.

    All CSS typography properties are set explicitly to avoid browser defaults
    leaking through.  Only the property named by spec.css_property changes
    between source and target.
    """
    value = spec.new_value if use_new_value else spec.old_value

    # Derive fixed values for each typography axis
    if spec.css_property == "font-family":
        font_family = value
    else:
        font_family = spec.font_family

    if spec.css_property == "font-weight":
        font_weight = value
    else:
        font_weight = "normal"

    if spec.css_property == "font-style":
        font_style = value
    else:
        font_style = "normal"

    if spec.css_property == "letter-spacing":
        letter_spacing = value
    else:
        letter_spacing = "normal"

    return (
        f"font-size:{spec.font_size_px}px;"
        f"font-family:{font_family};"
        f"color:{spec.font_color};"
        f"font-weight:{font_weight};"
        f"font-style:{font_style};"
        f"letter-spacing:{letter_spacing};"
        "margin:0;"
        "white-space:nowrap;"  # keep text on one line for consistent bbox measurement
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_typography_html(spec: TypographyEditSpec) -> tuple[str, str]:
    """
    Build source and target HTML from a TypographyEditSpec.

    Returns:
        (source_html, target_html) — identical in every respect except the
        single CSS property identified by spec.css_property.
    """
    def _html(use_new: bool) -> str:
        return (
            "<!DOCTYPE html>"
            "<html><body style=\"{body}\">"
            "<{el} style=\"{el_style}\">{text}</{el}>"
            "</body></html>"
        ).format(
            body=_body_style(spec.bg),
            el=spec.element,
            el_style=_element_style(spec, use_new),
            text=spec.text_content,
        )

    return _html(False), _html(True)
