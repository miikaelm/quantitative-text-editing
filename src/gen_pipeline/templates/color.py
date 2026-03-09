"""
templates/color.py — HTML template builder for color edit pairs.

build_color_html(spec) takes a ColorEditSpec and returns (source_html, target_html).
The two documents differ only in the text color: old_hex vs new_hex.
"""

from gen_pipeline.specs.color import ColorEditSpec


def build_color_html(spec: ColorEditSpec) -> tuple[str, str]:
    """
    Build source and target HTML strings from a ColorEditSpec.

    Returns:
        (source_html, target_html) — identical layout, differing only in text color.
    """
    base_style = (
        "margin:0; background:{bg}; display:flex;"
        "justify-content:center; align-items:center; height:100vh;"
    ).format(bg=spec.bg)

    elem_style_template = (
        "font-size:{font_size}; font-family:Arial,sans-serif; color:{{color}};"
    ).format(font_size=spec.font_size)

    def _html(color: str) -> str:
        return (
            "<!DOCTYPE html>"
            "<html><body style=\"{body}\">"
            "<{el} style=\"{el_style}\">{text}</{el}>"
            "</body></html>"
        ).format(
            body=base_style,
            el=spec.element,
            el_style=elem_style_template.format(color=color),
            text=spec.text_content,
        )

    return _html(spec.old_hex), _html(spec.new_hex)
