"""
templates/rotation.py — HTML template builder for rotation edit pairs.

build_rotation_html(spec) takes a RotationEditSpec and returns (source_html, target_html).
The two documents are identical in every respect except the CSS transform: rotate() value
applied to the text element.

Note: CSS transform: rotate(Xdeg) rotates clockwise for positive X.
"""

from gen_pipeline.specs.rotation import RotationEditSpec


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _body_style(bg: str) -> str:
    return (
        f"margin:0; padding:20px; background:{bg}; display:flex;"
        "justify-content:center; align-items:center; height:100vh;"
        "box-sizing:border-box;"
    )


def _element_style(spec: RotationEditSpec, angle_deg: float) -> str:
    return (
        f"font-size:{spec.font_size_px}px;"
        f"font-family:{spec.font_family};"
        f"color:{spec.font_color};"
        f"font-weight:{spec.font_weight};"
        "font-style:normal;"
        f"transform:rotate({angle_deg}deg);"
        "transform-origin:center center;"
        "display:inline-block;"  # required for transform to apply to inline elements
        "margin:0;"
        "white-space:nowrap;"
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_rotation_html(spec: RotationEditSpec) -> tuple[str, str]:
    """
    Build source and target HTML from a RotationEditSpec.

    Returns:
        (source_html, target_html) — identical in every respect except the
        transform: rotate() angle (old_angle_deg vs new_angle_deg).
    """
    def _html(angle_deg: float) -> str:
        return (
            "<!DOCTYPE html>"
            "<html><body style=\"{body}\">"
            "<{el} style=\"{el_style}\">{text}</{el}>"
            "</body></html>"
        ).format(
            body=_body_style(spec.bg),
            el=spec.element,
            el_style=_element_style(spec, angle_deg),
            text=spec.text_content,
        )

    return _html(spec.old_angle_deg), _html(spec.new_angle_deg)
