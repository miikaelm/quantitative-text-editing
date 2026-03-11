"""
l2/generate.py — Generation orchestration for Level 2 multi-element scenes.

Public entry point:
    generate_l2_pairs(edit_type, count, seed) -> list[EditPair]

The returned EditPair list is compatible with generate_pairs() in generate.py
and produces the same JSONL format, extended with:
    metadata.difficulty     = 2
    metadata.layout_type    = "<layout_name>"
    metadata.target_role    = "<role>"
    metadata.style_package  = "<package_name>"
    metadata.all_roles      = [...]
    metadata.role_contents  = {role: text, ...}

Flow for each sample:
    1. Query registry for layouts supporting the edit_type.
    2. Pick a layout and a style package (randomly).
    3. Sample a ContentSet from the layout's content pool.
    4. Resolve base styles (colors from palette, fonts from package, sizes from layout).
    5. Pick the target role and build edit parameters (old_value → new_value).
    6. Apply the edit to produce target_styles (copy of base_styles with one property changed).
    7. Build source and target HTML via the layout's html_builder.
    8. Compose the instruction (randomly using role name or text content).
    9. Return an EditPair with full metadata.

Supported edit types (L2 suffixed):
    color_l2, scaling_l2, typography_l2, rotation_l2, alignment_l2
"""

from __future__ import annotations

import random
from typing import Literal

from gen_pipeline.build_pairs import EditPair
from gen_pipeline.templates.base import classify_background
from gen_pipeline.l2.layouts import (
    LayoutDefinition,
    get_layouts_for_edit,
)
from gen_pipeline.l2.styles import STYLE_PACKAGES, StylePackage
from gen_pipeline.l2.content import CONTENT_POOLS, ContentSet


# ---------------------------------------------------------------------------
# Style resolution
# ---------------------------------------------------------------------------

def _resolve_base_styles(
    layout: LayoutDefinition,
    style_pkg: StylePackage,
    rng: random.Random,
) -> tuple[dict[str, dict], dict[str, str]]:
    """
    Build a complete resolved style dict for each role.

    Returns:
        role_styles:  {role: {color, font_size_px, font_weight, font_style,
                               letter_spacing, font_family, [alignment]}}
        role_slots:   {role: palette_slot_name}  — which slot was picked for
                       each role's color (needed to find color-edit alternatives)
    """
    role_styles: dict[str, dict] = {}
    role_slots: dict[str, str] = {}

    for role in layout.roles:
        base = layout.role_base_styles[role]
        constraint = layout.role_constraints[role]

        # Pick one palette slot for this role's color.
        slot = rng.choice(constraint.palette_slots)
        role_slots[role] = slot
        color = style_pkg.palette[slot]

        font_family = style_pkg.font_heading if base.get("is_heading") else style_pkg.font_body

        style: dict = {
            "color":          color,
            "font_size_px":   base["font_size_px"],
            "font_weight":    base["font_weight"],
            "font_style":     base["font_style"],
            "letter_spacing": base.get("letter_spacing", "normal"),
            "font_family":    font_family,
        }

        # Alignment-capable roles start at a randomly chosen position so the
        # source image varies and the edit target is always distinct from it.
        if constraint.can_align:
            style["alignment"] = rng.choice(constraint.alignment_positions)

        role_styles[role] = style

    return role_styles, role_slots


# ---------------------------------------------------------------------------
# Edit type: color
# ---------------------------------------------------------------------------

def _editable_roles_color(layout: LayoutDefinition) -> list[str]:
    return layout.roles  # all roles support color edits in layouts that include "color"


def _build_color_edit(
    target_role: str,
    role_styles: dict[str, dict],
    role_slots: dict[str, str],
    style_pkg: StylePackage,
    rng: random.Random,
) -> tuple[str, str]:
    """Return (old_hex, new_hex) for a color edit on target_role."""
    old_hex = role_styles[target_role]["color"]
    slot = role_slots[target_role]
    alternatives = style_pkg.edit_alternatives.get(slot, [])
    # Filter out the current color to guarantee a visible change.
    choices = [c for c in alternatives if c.upper() != old_hex.upper()]
    if not choices:
        choices = alternatives or ["#000000"]
    new_hex = rng.choice(choices)
    return old_hex, new_hex


def _make_color_instruction(target_role: str, target_text: str, new_hex: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return f"Change the {target_role.replace('_', ' ')} color to {new_hex}"
    return f"Change the color of '{target_text}' to {new_hex}"


# ---------------------------------------------------------------------------
# Edit type: scaling (font-size)
# ---------------------------------------------------------------------------

_SCALING_FACTORS = [0.75, 0.80, 1.25, 1.50]
_SCALING_FACTOR_LABELS = {
    0.75: ("Reduce", "by 25%"),
    0.80: ("Reduce", "by 20%"),
    1.25: ("Increase", "by 25%"),
    1.50: ("Increase", "by 50%"),
}


def _editable_roles_scaling(layout: LayoutDefinition) -> list[str]:
    return layout.roles


_MIN_FONT_PX = 30


def _build_scaling_edit(
    target_role: str,
    role_styles: dict[str, dict],
    rng: random.Random,
) -> tuple[int, int]:
    """Return (old_size_px, new_size_px). Target is always >= _MIN_FONT_PX."""
    old_px = role_styles[target_role]["font_size_px"]
    # Only consider factors whose result stays >= _MIN_FONT_PX and differs from source.
    valid_factors = [
        f for f in _SCALING_FACTORS
        if round(old_px * f) >= _MIN_FONT_PX and round(old_px * f) != old_px
    ]
    if not valid_factors:
        # Fallback: only grow (should not happen if base sizes are >= _MIN_FONT_PX).
        valid_factors = [f for f in _SCALING_FACTORS if f > 1.0]
    factor = rng.choice(valid_factors)
    new_px = max(_MIN_FONT_PX, round(old_px * factor))
    return old_px, new_px


def _make_scaling_instruction(target_role: str, target_text: str, old_px: int, new_px: int, rng: random.Random) -> str:
    factor = new_px / old_px
    pct = round(abs(factor - 1.0) * 100)
    if factor > 1.0 and pct in (25, 50):
        change_desc = f"Increase the font size by {pct}%"
    elif factor < 1.0 and pct in (20, 25):
        change_desc = f"Reduce the font size by {pct}%"
    else:
        change_desc = f"Set the font size to {new_px}px"

    if rng.random() < 0.5:
        return f"{change_desc} of the {target_role.replace('_', ' ')}"
    return f"{change_desc} of '{target_text}'"


# ---------------------------------------------------------------------------
# Edit type: typography
# ---------------------------------------------------------------------------

_TYPOGRAPHY_SUBCATEGORIES = ["font_weight", "font_style", "letter_spacing", "font_family"]

_FONT_WEIGHT_PAIRS = [("normal", "bold"), ("bold", "normal")]
_FONT_STYLE_PAIRS  = [("normal", "italic"), ("italic", "normal")]
_LETTER_SPACING_OPTIONS = ["normal", "0.05em", "0.10em", "0.15em", "0.20em", "-0.02em"]
_FONT_FAMILY_PAIRS = [
    ("'Roboto', Arial, sans-serif",                   "Georgia, 'Times New Roman', serif"),
    ("Georgia, 'Times New Roman', serif",             "'Roboto', Arial, sans-serif"),
    ("'Trebuchet MS', 'Segoe UI', Arial, sans-serif", "'Courier New', Courier, monospace"),
    ("'Courier New', Courier, monospace",             "'Trebuchet MS', 'Segoe UI', Arial, sans-serif"),
    ("Arial, sans-serif",                             "'Times New Roman', serif"),
    ("'Times New Roman', serif",                      "Arial, sans-serif"),
]


def _editable_roles_typography(layout: LayoutDefinition) -> list[str]:
    return layout.roles


def _build_typography_edit(
    target_role: str,
    role_styles: dict[str, dict],
    rng: random.Random,
) -> tuple[str, str, str, str]:
    """Return (subcategory, css_property, old_value, new_value)."""
    subcategory = rng.choice(_TYPOGRAPHY_SUBCATEGORIES)

    if subcategory == "font_weight":
        current = role_styles[target_role]["font_weight"]
        old_v, new_v = ("bold", "normal") if current == "bold" else ("normal", "bold")
        return "font_weight", "font-weight", old_v, new_v

    if subcategory == "font_style":
        current = role_styles[target_role]["font_style"]
        old_v, new_v = ("italic", "normal") if current == "italic" else ("normal", "italic")
        return "font_style", "font-style", old_v, new_v

    if subcategory == "letter_spacing":
        current = role_styles[target_role]["letter_spacing"]
        choices = [v for v in _LETTER_SPACING_OPTIONS if v != current]
        new_v = rng.choice(choices)
        return "letter_spacing", "letter-spacing", current, new_v

    # font_family
    current_family = role_styles[target_role]["font_family"]
    # Find a pair that starts with the current family, or pick a random pair.
    matching = [(o, n) for o, n in _FONT_FAMILY_PAIRS if o == current_family]
    if matching:
        old_v, new_v = rng.choice(matching)
    else:
        old_v, new_v = rng.choice(_FONT_FAMILY_PAIRS)
        old_v = current_family  # keep old as the actual current value
    return "font_family", "font-family", old_v, new_v


def _make_typography_instruction(
    target_role: str,
    target_text: str,
    subcategory: str,
    old_value: str,
    new_value: str,
    rng: random.Random,
) -> str:
    role_ref = target_role.replace("_", " ")
    text_ref = f"'{target_text}'"
    ref = role_ref if rng.random() < 0.5 else text_ref

    if subcategory == "font_weight":
        action = "Make bold" if new_value == "bold" else "Remove bold from"
        if new_value == "bold":
            return f"Make the {ref} bold" if ref == role_ref else f"Make {ref} bold"
        return f"Remove the bold style from the {ref}" if ref == role_ref else f"Remove bold from {ref}"

    if subcategory == "font_style":
        if new_value == "italic":
            return f"Make the {ref} italic" if ref == role_ref else f"Make {ref} italic"
        return f"Remove the italic style from the {ref}" if ref == role_ref else f"Remove italic from {ref}"

    if subcategory == "letter_spacing":
        def _em_val(v: str) -> float:
            v = v.strip()
            if v == "normal":
                return 0.0
            return float(v.replace("em", "").replace("px", ""))
        direction = "Increase" if _em_val(new_value) >= _em_val(old_value) else "Decrease"
        if ref == role_ref:
            return f"{direction} the letter spacing of the {ref}"
        return f"{direction} the letter spacing of {ref}"

    # font_family
    if ref == role_ref:
        return f"Change the font family of the {ref} to {new_value}"
    return f"Change the font family of {ref} to {new_value}"


# ---------------------------------------------------------------------------
# Edit type: rotation (title_byline only — title role)
# ---------------------------------------------------------------------------

_ROTATION_ANGLES = [-30, -25, -20, -15, -10, 10, 15, 20, 25, 30]


def _editable_roles_rotation(layout: LayoutDefinition) -> list[str]:
    return [r for r in layout.roles if layout.role_constraints[r].can_rotate]


def _build_rotation_edit(rng: random.Random) -> tuple[float, float]:
    """Return (old_deg=0.0, new_deg) — source is always upright."""
    new_deg = float(rng.choice(_ROTATION_ANGLES))
    return 0.0, new_deg


def _make_rotation_instruction(
    target_role: str,
    target_text: str,
    old_deg: float,
    new_deg: float,
    rng: random.Random,
) -> str:
    abs_deg = abs(new_deg)
    direction = "clockwise" if new_deg > 0 else "counterclockwise"
    if rng.random() < 0.5:
        return f"Rotate the {target_role.replace('_', ' ')} {abs_deg:.0f} degrees {direction}"
    return f"Rotate '{target_text}' {abs_deg:.0f} degrees {direction}"


# ---------------------------------------------------------------------------
# Edit type: alignment
# ---------------------------------------------------------------------------

def _editable_roles_alignment(layout: LayoutDefinition) -> list[str]:
    return [r for r in layout.roles if layout.role_constraints[r].can_align]


def _build_alignment_edit(
    target_role: str,
    role_styles: dict[str, dict],
    layout: LayoutDefinition,
    rng: random.Random,
) -> tuple[str, str]:
    """Return (old_position, new_position)."""
    old_pos = role_styles[target_role].get("alignment", "top-left")
    positions = layout.role_constraints[target_role].alignment_positions
    choices = [p for p in positions if p != old_pos]
    if not choices:
        choices = positions
    new_pos = rng.choice(choices)
    return old_pos, new_pos


_ALIGNMENT_HUMAN: dict[str, str] = {
    # split_panel vertical positions
    "top":           "top",
    "center":        "center",
    "bottom":        "bottom",
    # title_byline horizontal positions
    "bottom-left":   "bottom-left corner",
    "bottom-center": "bottom center",
    "bottom-right":  "bottom-right corner",
}


def _make_alignment_instruction(
    target_role: str,
    target_text: str,
    new_pos: str,
    rng: random.Random,
) -> str:
    pos_human = _ALIGNMENT_HUMAN.get(new_pos, new_pos)
    if rng.random() < 0.5:
        return f"Move the {target_role.replace('_', ' ')} to the {pos_human}"
    return f"Move '{target_text}' to the {pos_human}"


# ---------------------------------------------------------------------------
# Core pair generator
# ---------------------------------------------------------------------------

def _apply_edit_to_styles(
    role_styles: dict[str, dict],
    target_role: str,
    css_key: str,
    new_value,
) -> dict[str, dict]:
    """
    Return a deep copy of role_styles with one property of target_role changed.
    css_key is the Python key in the style dict (not the CSS property name).
    """
    import copy
    tgt = copy.deepcopy(role_styles)
    tgt[target_role][css_key] = new_value
    return tgt


def generate_l2_pairs(
    edit_type: str,
    count: int,
    seed: int | None = None,
) -> list[EditPair]:
    """
    Generate `count` Level 2 EditPair objects for the given base edit type.

    Args:
        edit_type: Base edit type without "_l2" suffix (e.g. "color", "scaling").
        count:     Number of pairs to generate.
        seed:      Optional RNG seed for reproducibility.

    Returns:
        List of EditPair objects ready for rendering via generate_pairs().
    """
    rng = random.Random(seed)

    compatible_layouts = get_layouts_for_edit(edit_type)
    if not compatible_layouts:
        raise ValueError(f"No layouts support edit type '{edit_type}'")

    pairs: list[EditPair] = []

    for i in range(count):
        pair_id = f"{edit_type}_l2_{i + 1:03d}"

        # 1. Pick layout and style package.
        layout: LayoutDefinition = rng.choice(compatible_layouts)
        style_pkg: StylePackage = rng.choice(STYLE_PACKAGES)

        # 2. Sample content from the pool.
        content_pool = CONTENT_POOLS[layout.name]
        contents: ContentSet = rng.choice(content_pool)

        # 3. Resolve base styles (colors from palette, fonts from package, sizes from layout).
        role_styles, role_slots = _resolve_base_styles(layout, style_pkg, rng)

        # 4. Pick background.
        bg = style_pkg.pick_background(rng)

        # 5. Determine editable roles for this edit type and pick target.
        if edit_type == "color":
            editable = _editable_roles_color(layout)
        elif edit_type == "scaling":
            editable = _editable_roles_scaling(layout)
        elif edit_type == "typography":
            editable = _editable_roles_typography(layout)
        elif edit_type == "rotation":
            editable = _editable_roles_rotation(layout)
        elif edit_type == "alignment":
            editable = _editable_roles_alignment(layout)
        else:
            raise ValueError(f"Unsupported L2 edit type: '{edit_type}'")

        if not editable:
            # Rare: layout was included in compatible_layouts but no role
            # satisfies fine-grained constraints. Skip and regenerate.
            count_remaining = count - len(pairs)
            continue

        target_role = rng.choice(editable)
        target_text = contents[target_role]

        # 6. Build edit parameters and derive target styles.
        metadata: dict = {
            "difficulty":     2,
            "layout_type":    layout.name,
            "style_package":  style_pkg.name,
            "target_role":    target_role,
            "all_roles":      list(layout.roles),
            "role_contents":  dict(contents),
            "background_css": bg,
            "background_type": classify_background(bg),
            "text_content":   target_text,  # for OCR bbox lookup (target element only)
        }

        if edit_type == "color":
            old_hex, new_hex = _build_color_edit(target_role, role_styles, role_slots, style_pkg, rng)
            target_styles = _apply_edit_to_styles(role_styles, target_role, "color", new_hex)
            instruction = _make_color_instruction(target_role, target_text, new_hex, rng)
            metadata.update({
                "property":  "color",
                "old_value": old_hex,
                "new_value": new_hex,
            })

        elif edit_type == "scaling":
            old_px, new_px = _build_scaling_edit(target_role, role_styles, rng)
            target_styles = _apply_edit_to_styles(role_styles, target_role, "font_size_px", new_px)
            instruction = _make_scaling_instruction(target_role, target_text, old_px, new_px, rng)
            metadata.update({
                "property":     "font-size",
                "old_value":    f"{old_px}px",
                "new_value":    f"{new_px}px",
                "scale_factor": round(new_px / old_px, 4),
            })

        elif edit_type == "typography":
            subcategory, css_prop, old_v, new_v = _build_typography_edit(target_role, role_styles, rng)
            # Map subcategory → style dict key for _apply_edit_to_styles.
            style_key_map = {
                "font_weight":    "font_weight",
                "font_style":     "font_style",
                "letter_spacing": "letter_spacing",
                "font_family":    "font_family",
            }
            target_styles = _apply_edit_to_styles(role_styles, target_role, style_key_map[subcategory], new_v)
            instruction = _make_typography_instruction(target_role, target_text, subcategory, old_v, new_v, rng)
            metadata.update({
                "typography_subcategory": subcategory,
                "property":  css_prop,
                "old_value": old_v,
                "new_value": new_v,
            })

        elif edit_type == "rotation":
            old_deg, new_deg = _build_rotation_edit(rng)
            target_styles = _apply_edit_to_styles(role_styles, target_role, "rotation_deg", new_deg)
            instruction = _make_rotation_instruction(target_role, target_text, old_deg, new_deg, rng)
            metadata.update({
                "property":      "transform:rotate",
                "old_value":     f"{old_deg:.1f}deg",
                "new_value":     f"{new_deg:.1f}deg",
                "old_angle_deg": old_deg,
                "new_angle_deg": new_deg,
            })

        elif edit_type == "alignment":
            old_pos, new_pos = _build_alignment_edit(target_role, role_styles, layout, rng)
            target_styles = _apply_edit_to_styles(role_styles, target_role, "alignment", new_pos)
            instruction = _make_alignment_instruction(target_role, target_text, new_pos, rng)
            metadata.update({
                "property":  "position",
                "old_value": old_pos,
                "new_value": new_pos,
            })

        # 7. Build HTML pair.
        source_html, target_html = layout.html_builder(contents, role_styles, target_styles, bg)

        pairs.append(EditPair(
            pair_id=pair_id,
            edit_type=f"{edit_type}_l2",
            source_html=source_html,
            target_html=target_html,
            instruction=instruction,
            metadata=metadata,
        ))

    return pairs
