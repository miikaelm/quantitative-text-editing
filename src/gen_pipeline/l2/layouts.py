"""
l2/layouts.py — Layout registry, definitions, and HTML builders for Level 2 scenes.

A LayoutDefinition declares:
  - roles: text elements present in the scene (e.g. ["title", "subtitle"])
  - supported_edits: which edit types work for this layout
  - role_constraints: per-role allowed edits and palette slot assignments
  - role_base_styles: default font-size, weight, style, spacing per role
  - html_builder: function (contents, src_styles, tgt_styles, bg) -> (src_html, tgt_html)

Font size should be kept over 30 px

Usage:
    from gen_pipeline.l2.layouts import get_layouts_for_edit, get_layout, all_layouts

    color_layouts = get_layouts_for_edit("color")
    layout = get_layout("title_subtitle")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RoleConstraints:
    """Per-role constraints within a layout."""
    # Semantic palette slots this role's color is drawn from (for color edits).
    palette_slots: list[str]
    # Whether the role supports CSS rotation edits.
    can_rotate: bool = False
    rotation_range: tuple[int, int] = (-30, 30)
    # Whether the role supports alignment (position) edits.
    can_align: bool = False
    # Valid alignment positions for this role (if can_align is True).
    alignment_positions: list[str] = field(default_factory=list)


@dataclass
class LayoutDefinition:
    """Complete description of a multi-element layout."""
    name: str
    # Ordered list of role names present in the scene.
    roles: list[str]
    # Edit types this layout supports. Only roles with matching constraints
    # can be the target of a given edit.
    supported_edits: list[str]
    # Per-role edit constraints and palette slot assignments.
    role_constraints: dict[str, RoleConstraints]
    # Default non-color styles for each role (font_size_px, font_weight, etc.).
    # Colors are supplied per-generation from the style package.
    role_base_styles: dict[str, dict]
    # Pure rendering function. Signature:
    #   (contents, src_styles, tgt_styles, bg) -> (source_html, target_html)
    # contents:   {role: text_string}
    # src_styles: {role: resolved_style_dict}  (source)
    # tgt_styles: {role: resolved_style_dict}  (target — identical to src except one property)
    # bg:         CSS background value for the body
    html_builder: Callable


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, LayoutDefinition] = {}


def register(layout: LayoutDefinition) -> LayoutDefinition:
    """Register a layout and return it (for use as a statement)."""
    _REGISTRY[layout.name] = layout
    return layout


def get_layouts_for_edit(edit_type: str) -> list[LayoutDefinition]:
    """Return all layouts that declare support for the given edit type."""
    return [layout for layout in _REGISTRY.values() if edit_type in layout.supported_edits]


def get_layout(name: str) -> LayoutDefinition:
    return _REGISTRY[name]


def all_layouts() -> list[LayoutDefinition]:
    return list(_REGISTRY.values())


# ---------------------------------------------------------------------------
# Shared HTML rendering utilities
# ---------------------------------------------------------------------------

def _role_css(s: dict) -> str:
    """
    Convert a resolved role style dict to a CSS inline string.

    Expected keys: color, font_size_px, font_family, font_weight, font_style,
                   letter_spacing (optional), rotation_deg (optional).
    """
    parts = [
        f"color:{s['color']}",
        f"font-size:{s['font_size_px']}px",
        f"font-family:{s['font_family']}",
        f"font-weight:{s['font_weight']}",
        f"font-style:{s['font_style']}",
        f"letter-spacing:{s.get('letter_spacing', 'normal')}",
        "margin:0",
    ]
    rot = s.get("rotation_deg", 0.0)
    if rot:
        parts.append(f"transform:rotate({rot:.1f}deg)")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Layout 1: title_subtitle — centered vertical stack
#
# Supports: color, scaling, typography.
# No rotation (elements too close together). No alignment.
# ---------------------------------------------------------------------------

def _build_title_subtitle(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        body = (
            f"margin:0; background:{bg}; display:flex; flex-direction:column;"
            "justify-content:center; align-items:center; height:100vh; gap:20px;"
            "box-sizing:border-box; padding:40px;"
        )
        title_css = _role_css(styles["title"]) + "; text-align:center"
        subtitle_css = _role_css(styles["subtitle"]) + "; text-align:center"
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<h1 style="{title_css}">{contents["title"]}</h1>'
            f'<h2 style="{subtitle_css}">{contents["subtitle"]}</h2>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="title_subtitle",
    roles=["title", "subtitle"],
    supported_edits=["color", "scaling", "typography"],
    role_constraints={
        "title":    RoleConstraints(palette_slots=["primary", "accent"]),
        "subtitle": RoleConstraints(palette_slots=["secondary"]),
    },
    role_base_styles={
        "title":    {"font_size_px": 60, "font_weight": "bold",   "font_style": "normal", "letter_spacing": "normal", "is_heading": True},
        "subtitle": {"font_size_px": 30, "font_weight": "normal", "font_style": "normal", "letter_spacing": "normal", "is_heading": False},
    },
    html_builder=_build_title_subtitle,
))


# ---------------------------------------------------------------------------
# Layout 2: title_byline — title centered, byline absolutely positioned
#
# Supports: all edit types.
# Rotation only on title (±30°). Alignment on byline (bottom-left/center/right).
# ---------------------------------------------------------------------------

# Maps byline alignment → CSS for position:absolute element.
# Uses left/right/width instead of transform:translateX to avoid conflicts.
_BYLINE_POS_CSS: dict[str, str] = {
    "bottom-left":   "bottom:24px; left:24px",
    "bottom-center": "bottom:24px; left:0; right:0; text-align:center",
    "bottom-right":  "bottom:24px; right:24px",
}


def _build_title_byline(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        body = (
            f"margin:0; background:{bg}; position:relative; height:100vh;"
            "display:flex; justify-content:center; align-items:center;"
        )
        title_css = _role_css(styles["title"]) + "; text-align:center"
        byline_alignment = styles["byline"].get("alignment", "bottom-left")
        pos_css = _BYLINE_POS_CSS[byline_alignment]
        byline_css = _role_css(styles["byline"]) + "; white-space:nowrap"
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<h1 style="{title_css}">{contents["title"]}</h1>'
            f'<span style="position:absolute; {pos_css}; {byline_css}">{contents["byline"]}</span>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="title_byline",
    roles=["title", "byline"],
    supported_edits=["color", "scaling", "typography", "rotation", "alignment"],
    role_constraints={
        "title":  RoleConstraints(
            palette_slots=["primary"],
            can_rotate=True,
            rotation_range=(-30, 30),
        ),
        "byline": RoleConstraints(
            palette_slots=["secondary", "accent"],
            can_align=True,
            alignment_positions=["bottom-left", "bottom-center", "bottom-right"],
        ),
    },
    role_base_styles={
        "title":  {"font_size_px": 60, "font_weight": "bold",   "font_style": "normal", "letter_spacing": "normal", "is_heading": True},
        "byline": {"font_size_px": 30, "font_weight": "normal", "font_style": "normal", "letter_spacing": "0.05em", "is_heading": False,
                   "default_alignment": "bottom-left"},
    },
    html_builder=_build_title_byline,
))


# ---------------------------------------------------------------------------
# Layout 3: header_body — left-aligned column
#
# Supports: color, scaling, typography. No rotation, no alignment.
# ---------------------------------------------------------------------------

def _build_header_body(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        body = (
            f"margin:0; background:{bg}; display:flex; flex-direction:column;"
            "justify-content:center; padding:56px; height:100vh; box-sizing:border-box;"
        )
        header_css = _role_css(styles["header"]) + "; margin-bottom:16px"
        body_css = _role_css(styles["body"]) + "; line-height:1.6; max-width:680px"
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<h2 style="{header_css}">{contents["header"]}</h2>'
            f'<p style="{body_css}">{contents["body"]}</p>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="header_body",
    roles=["header", "body"],
    supported_edits=["color", "scaling", "typography"],
    role_constraints={
        "header": RoleConstraints(palette_slots=["primary"]),
        "body":   RoleConstraints(palette_slots=["secondary"]),
    },
    role_base_styles={
        "header": {"font_size_px": 48, "font_weight": "bold",   "font_style": "normal", "letter_spacing": "normal", "is_heading": True},
        "body":   {"font_size_px": 30, "font_weight": "normal", "font_style": "normal", "letter_spacing": "normal", "is_heading": False},
    },
    html_builder=_build_header_body,
))


# ---------------------------------------------------------------------------
# Layout 4: name_card — three-element centered stack
#
# Supports: color, scaling, typography. No rotation, no alignment.
# ---------------------------------------------------------------------------

def _build_name_card(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        body = (
            f"margin:0; background:{bg}; display:flex; flex-direction:column;"
            "justify-content:center; align-items:center; height:100vh; gap:10px;"
            "box-sizing:border-box; padding:40px;"
        )
        name_css = _role_css(styles["name"]) + "; text-align:center"
        job_css = _role_css(styles["job_title"]) + "; text-align:center"
        org_css = _role_css(styles["organization"]) + "; text-align:center; opacity:0.85"
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<h1 style="{name_css}">{contents["name"]}</h1>'
            f'<p style="{job_css}">{contents["job_title"]}</p>'
            f'<p style="{org_css}">{contents["organization"]}</p>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="name_card",
    roles=["name", "job_title", "organization"],
    supported_edits=["color", "scaling", "typography"],
    role_constraints={
        "name":         RoleConstraints(palette_slots=["primary"]),
        "job_title":    RoleConstraints(palette_slots=["secondary"]),
        "organization": RoleConstraints(palette_slots=["accent", "secondary"]),
    },
    role_base_styles={
        "name":         {"font_size_px": 56, "font_weight": "bold",   "font_style": "normal", "letter_spacing": "normal",   "is_heading": True},
        "job_title":    {"font_size_px": 30, "font_weight": "normal", "font_style": "normal", "letter_spacing": "0.08em",   "is_heading": False},
        "organization": {"font_size_px": 30, "font_weight": "normal", "font_style": "normal", "letter_spacing": "0.05em",   "is_heading": False},
    },
    html_builder=_build_name_card,
))


# ---------------------------------------------------------------------------
# Layout 5: split_panel — horizontal two-column layout
#
# Supports: color, scaling, typography, alignment (descriptor in right panel).
# No rotation.
# ---------------------------------------------------------------------------

# Maps descriptor alignment → CSS for the right-panel flex container.
# Horizontal position is fixed to the left edge of the right panel (= image center).
# Only vertical position varies so the alignment edit produces a clearly visible change.
_DESCRIPTOR_ALIGN_CSS: dict[str, str] = {
    "top":    "justify-content:flex-start; align-items:flex-start",
    "center": "justify-content:flex-start; align-items:center",
    "bottom": "justify-content:flex-start; align-items:flex-end",
}


def _build_split_panel(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        body = f"margin:0; background:{bg}; display:flex; height:100vh;"
        label_css = _role_css(styles["label"]) + "; text-align:center"
        descriptor_alignment = styles["descriptor"].get("alignment", "top-left")
        desc_align_css = _DESCRIPTOR_ALIGN_CSS[descriptor_alignment]
        descriptor_css = _role_css(styles["descriptor"])
        # Left panel: centered, slightly darker surface
        left_panel = (
            'display:flex; flex:1; justify-content:center; align-items:center;'
            'padding:40px; box-sizing:border-box;'
        )
        right_panel = (
            f'display:flex; flex:1; {desc_align_css};'
            'padding:40px; box-sizing:border-box;'
        )
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<div style="{left_panel}">'
            f'<span style="{label_css}">{contents["label"]}</span>'
            f'</div>'
            f'<div style="{right_panel}">'
            f'<span style="{descriptor_css}">{contents["descriptor"]}</span>'
            f'</div>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="split_panel",
    roles=["label", "descriptor"],
    supported_edits=["color", "scaling", "typography", "alignment"],
    role_constraints={
        "label":      RoleConstraints(palette_slots=["primary", "accent"]),
        "descriptor": RoleConstraints(
            palette_slots=["secondary"],
            can_align=True,
            alignment_positions=["top", "center", "bottom"],
        ),
    },
    role_base_styles={
        "label":      {"font_size_px": 52, "font_weight": "bold",   "font_style": "normal", "letter_spacing": "0.05em", "is_heading": True},
        "descriptor": {"font_size_px": 30, "font_weight": "normal", "font_style": "normal", "letter_spacing": "normal", "is_heading": False},
    },
    html_builder=_build_split_panel,
))

# ---------------------------------------------------------------------------
# Layout 6: solo_headline — single centered text element
#
# Supports: all edit types.
# The only single-role layout. Rotation is safe (no neighbouring elements).
# Alignment moves the headline to different screen regions.
# Ideal for isolating individual edit effects with zero confounds.
# ---------------------------------------------------------------------------

_SOLO_POS_CSS: dict[str, str] = {
    "center":     "justify-content:center; align-items:center",
    "top-center": "justify-content:center; align-items:flex-start; padding-top:80px",
    "bottom-center": "justify-content:center; align-items:flex-end; padding-bottom:80px",
}


def _build_solo_headline(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        alignment = styles["headline"].get("alignment", "center")
        pos_css = _SOLO_POS_CSS[alignment]
        body = (
            f"margin:0; background:{bg}; display:flex; {pos_css};"
            "height:100vh; box-sizing:border-box;"
        )
        headline_css = _role_css(styles["headline"]) + "; text-align:center; max-width:80%"
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<h1 style="{headline_css}">{contents["headline"]}</h1>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="solo_headline",
    roles=["headline"],
    supported_edits=["color", "scaling", "typography", "rotation", "alignment"],
    role_constraints={
        "headline": RoleConstraints(
            palette_slots=["primary", "accent"],
            can_rotate=True,
            rotation_range=(-25, 25),
            can_align=True,
            alignment_positions=["center", "top-center", "bottom-center"],
        ),
    },
    role_base_styles={
        "headline": {"font_size_px": 64, "font_weight": "bold", "font_style": "normal",
                     "letter_spacing": "normal", "is_heading": True,
                     "default_alignment": "center"},
    },
    html_builder=_build_solo_headline,
))


# ---------------------------------------------------------------------------
# Layout 7: quote_attribution — block quote with attribution
#
# Supports: color, scaling, typography. No rotation, no alignment.
# Quote role defaults to italic, providing a natural typography-edit baseline
# (italic -> normal) not present in other layouts.
# ---------------------------------------------------------------------------

def _build_quote_attribution(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        body = (
            f"margin:0; background:{bg}; display:flex; flex-direction:column;"
            "justify-content:center; padding:64px 80px; height:100vh;"
            "box-sizing:border-box;"
        )
        quote_css = _role_css(styles["quote"]) + "; line-height:1.5; max-width:720px"
        attr_css = _role_css(styles["attribution"]) + "; margin-top:24px"
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<p style="{quote_css}">{contents["quote"]}</p>'
            f'<p style="{attr_css}">\u2014 {contents["attribution"]}</p>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="quote_attribution",
    roles=["quote", "attribution"],
    supported_edits=["color", "scaling", "typography"],
    role_constraints={
        "quote":       RoleConstraints(palette_slots=["primary"]),
        "attribution": RoleConstraints(palette_slots=["secondary", "accent"]),
    },
    role_base_styles={
        "quote":       {"font_size_px": 48, "font_weight": "normal", "font_style": "italic",
                        "letter_spacing": "normal", "is_heading": False},
        "attribution": {"font_size_px": 30, "font_weight": "bold",   "font_style": "normal",
                        "letter_spacing": "0.05em", "is_heading": False},
    },
    html_builder=_build_quote_attribution,
))


# ---------------------------------------------------------------------------
# Layout 8: corner_badge — centered label with a corner-positioned badge
#
# Supports: color, scaling, typography, alignment.
# Badge alignment uses four 2D corner positions (unique among layouts).
# No rotation — badge text is short; rotation on it would be barely visible.
# ---------------------------------------------------------------------------

_BADGE_POS_CSS: dict[str, str] = {
    "top-left":     "top:24px; left:24px",
    "top-right":    "top:24px; right:24px",
    "bottom-left":  "bottom:24px; left:24px",
    "bottom-right": "bottom:24px; right:24px",
}


def _build_corner_badge(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        body = (
            f"margin:0; background:{bg}; position:relative; height:100vh;"
            "display:flex; justify-content:center; align-items:center;"
        )
        label_css = _role_css(styles["label"]) + "; text-align:center"
        badge_alignment = styles["badge"].get("alignment", "top-right")
        pos_css = _BADGE_POS_CSS[badge_alignment]
        badge_css = _role_css(styles["badge"]) + "; white-space:nowrap"
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<h1 style="{label_css}">{contents["label"]}</h1>'
            f'<span style="position:absolute; {pos_css}; {badge_css}">'
            f'{contents["badge"]}</span>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="corner_badge",
    roles=["label", "badge"],
    supported_edits=["color", "scaling", "typography", "alignment"],
    role_constraints={
        "label": RoleConstraints(palette_slots=["primary"]),
        "badge": RoleConstraints(
            palette_slots=["accent", "secondary"],
            can_align=True,
            alignment_positions=["top-left", "top-right", "bottom-left", "bottom-right"],
        ),
    },
    role_base_styles={
        "label": {"font_size_px": 56, "font_weight": "bold",   "font_style": "normal",
                  "letter_spacing": "normal", "is_heading": True},
        "badge": {"font_size_px": 30, "font_weight": "bold",   "font_style": "normal",
                  "letter_spacing": "0.1em", "is_heading": False,
                  "default_alignment": "top-right"},
    },
    html_builder=_build_corner_badge,
))


# ---------------------------------------------------------------------------
# Layout 9: banner_caption — large top banner + bottom caption
#
# Supports: color, scaling, typography, rotation, alignment.
# Banner sits at the top with plenty of vertical clearance — safe to rotate.
# Caption at the bottom can shift left/center/right.
# Inverted spatial logic compared to title_byline (top-heavy vs center+bottom).
# ---------------------------------------------------------------------------

_CAPTION_ALIGN_CSS: dict[str, str] = {
    "left":   "text-align:left",
    "center": "text-align:center",
    "right":  "text-align:right",
}


def _build_banner_caption(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        body = (
            f"margin:0; background:{bg}; display:flex; flex-direction:column;"
            "justify-content:space-between; height:100vh;"
            "box-sizing:border-box; padding:60px 48px;"
        )
        banner_css = _role_css(styles["banner"]) + "; text-align:center"
        caption_alignment = styles["caption"].get("alignment", "center")
        align_css = _CAPTION_ALIGN_CSS[caption_alignment]
        caption_css = _role_css(styles["caption"]) + f"; {align_css}"
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<h1 style="{banner_css}">{contents["banner"]}</h1>'
            f'<p style="{caption_css}">{contents["caption"]}</p>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="banner_caption",
    roles=["banner", "caption"],
    supported_edits=["color", "scaling", "typography", "alignment"],
    role_constraints={
        "banner": RoleConstraints(
            palette_slots=["primary", "accent"],
        ),
        "caption": RoleConstraints(
            palette_slots=["secondary"],
            can_align=True,
            alignment_positions=["left", "center", "right"],
        ),
    },
    role_base_styles={
        "banner":  {"font_size_px": 72, "font_weight": "bold",   "font_style": "normal",
                    "letter_spacing": "0.03em", "is_heading": True},
        "caption": {"font_size_px": 30, "font_weight": "normal", "font_style": "normal",
                    "letter_spacing": "normal", "is_heading": False,
                    "default_alignment": "center"},
    },
    html_builder=_build_banner_caption,
))


# ---------------------------------------------------------------------------
# Layout 10: two_column_heading — two side-by-side headings
#
# Supports: color, scaling, typography, rotation.
# Both roles can rotate independently (each in its own flex cell).
# This is the only layout with two rotation-capable roles, giving training
# pairs where either element can be the rotation target.
# No alignment — both elements are pinned to their cell centers.
# ---------------------------------------------------------------------------

def _build_two_column_heading(
    contents: dict[str, str],
    src_styles: dict[str, dict],
    tgt_styles: dict[str, dict],
    bg: str,
) -> tuple[str, str]:
    def _html(styles: dict[str, dict]) -> str:
        body = f"margin:0; background:{bg}; display:flex; height:100vh;"
        cell = (
            "display:flex; flex:1; justify-content:center; align-items:center;"
            "padding:40px; box-sizing:border-box;"
        )
        left_css = _role_css(styles["left_heading"]) + "; text-align:center"
        right_css = _role_css(styles["right_heading"]) + "; text-align:center"
        return (
            f'<!DOCTYPE html><html><body style="{body}">'
            f'<div style="{cell}">'
            f'<span style="{left_css}">{contents["left_heading"]}</span>'
            f'</div>'
            f'<div style="{cell}">'
            f'<span style="{right_css}">{contents["right_heading"]}</span>'
            f'</div>'
            f'</body></html>'
        )
    return _html(src_styles), _html(tgt_styles)


register(LayoutDefinition(
    name="two_column_heading",
    roles=["left_heading", "right_heading"],
    supported_edits=["color", "scaling", "typography", "rotation"],
    role_constraints={
        "left_heading":  RoleConstraints(
            palette_slots=["primary"],
            can_rotate=True,
            rotation_range=(-20, 20),
        ),
        "right_heading": RoleConstraints(
            palette_slots=["accent", "secondary"],
            can_rotate=True,
            rotation_range=(-20, 20),
        ),
    },
    role_base_styles={
        "left_heading":  {"font_size_px": 48, "font_weight": "bold",   "font_style": "normal",
                          "letter_spacing": "normal", "is_heading": True},
        "right_heading": {"font_size_px": 48, "font_weight": "bold",   "font_style": "normal",
                          "letter_spacing": "normal", "is_heading": True},
    },
    html_builder=_build_two_column_heading,
))