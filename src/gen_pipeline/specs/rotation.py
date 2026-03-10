"""
specs/rotation.py — Rotation edit specs and spec generator.

generate_rotation_specs() returns a list of RotationEditSpec instances.
Each spec describes rotating a text element via CSS transform: rotate(Xdeg).

Rotation is specified in degrees, clockwise (matching CSS transform: rotate(Xdeg)).
Negative values = counterclockwise.

For now: 8 hardcoded examples for pipeline verification.
"""

from dataclasses import dataclass

from gen_pipeline.specs.base import EditSpec
from gen_pipeline.templates.base import classify_background


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class RotationEditSpec(EditSpec):
    """Edit spec for a CSS rotation change (transform: rotate)."""
    element: str = "h1"
    old_angle_deg: float = 0.0
    new_angle_deg: float = 20.0
    font_size_px: int = 64
    font_color: str = "#333333"
    font_family: str = "Arial, sans-serif"
    font_weight: str = "normal"
    bg: str = "#f5f5f5"


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

def _make_instruction(old_deg: float, new_deg: float) -> str:
    delta = round(new_deg - old_deg, 2)
    abs_delta = abs(delta)
    direction = "clockwise" if delta > 0 else "counterclockwise"

    if new_deg == 0.0:
        return "Remove the rotation and make the text upright"
    if old_deg == 0.0:
        return f"Rotate the text {abs_delta:.0f} degrees {direction}"
    return f"Rotate the text {abs_delta:.0f} degrees {direction}"


# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------

def _make_rotation_spec(
    index: int,
    element: str,
    old_angle_deg: float,
    new_angle_deg: float,
    text: str,
    font_size_px: int = 64,
    font_color: str = "#333333",
    font_family: str = "Arial, sans-serif",
    font_weight: str = "normal",
    bg: str = "#f5f5f5",
) -> RotationEditSpec:
    pair_id = f"rotation_{index:03d}"
    delta = round(new_angle_deg - old_angle_deg, 2)
    return RotationEditSpec(
        pair_id=pair_id,
        edit_type="rotation",
        instruction=_make_instruction(old_angle_deg, new_angle_deg),
        text_content=text,
        element=element,
        old_angle_deg=old_angle_deg,
        new_angle_deg=new_angle_deg,
        font_size_px=font_size_px,
        font_color=font_color,
        font_family=font_family,
        font_weight=font_weight,
        bg=bg,
        metadata={
            "element": element,
            "property": "transform",
            "old_value": f"rotate({old_angle_deg}deg)",
            "new_value": f"rotate({new_angle_deg}deg)",
            "old_angle_deg": old_angle_deg,
            "new_angle_deg": new_angle_deg,
            "rotation_delta_deg": delta,
            "text_content": text,
            "font_size_px": font_size_px,
            "font_color": font_color,
            "font_family": font_family,
            "font_weight": font_weight,
            "background_css": bg,
            "background_type": classify_background(bg),
        },
    )


# ---------------------------------------------------------------------------
# Public entry point — 8 hardcoded examples for pipeline verification
# ---------------------------------------------------------------------------

def generate_rotation_specs() -> list[RotationEditSpec]:
    """Return a small set of rotation specs for pipeline end-to-end verification."""
    return [
        # 1. 0° → 20° clockwise, Arial, light background
        _make_rotation_spec(
            1, "h1", 0.0, 20.0, "TILT",
            font_size_px=64, font_color="#333333",
            font_family="Arial, sans-serif", font_weight="normal",
            bg="#f5f5f5",
        ),
        # 2. 0° → 45° clockwise, white on dark
        _make_rotation_spec(
            2, "h1", 0.0, 45.0, "DIAGONAL",
            font_size_px=64, font_color="#ffffff",
            font_family="'Times New Roman', serif", font_weight="bold",
            bg="#1a1a2e",
        ),
        # 3. 0° → -15° counterclockwise, gradient background
        _make_rotation_spec(
            3, "h2", 0.0, -15.0, "Lean",
            font_size_px=56, font_color="#2d3748",
            font_family="Georgia, serif", font_weight="normal",
            bg="linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
        ),
        # 4. 30° → 0° (straighten), light background
        _make_rotation_spec(
            4, "h1", 30.0, 0.0, "UPRIGHT",
            font_size_px=60, font_color="#1a202c",
            font_family="Verdana, sans-serif", font_weight="normal",
            bg="#edf2f7",
        ),
        # 5. 0° → 90° clockwise (vertical), terminal look
        _make_rotation_spec(
            5, "h1", 0.0, 90.0, "VERTICAL",
            font_size_px=48, font_color="#00ff41",
            font_family="'Courier New', monospace", font_weight="normal",
            bg="#0d0d0d",
        ),
        # 6. -20° → 20° (flip tilt direction), red on light
        _make_rotation_spec(
            6, "h1", -20.0, 20.0, "FLIP",
            font_size_px=60, font_color="#e53e3e",
            font_family="'Trebuchet MS', sans-serif", font_weight="bold",
            bg="#fff5f5",
        ),
        # 7. 0° → 10° slight clockwise, Courier on white
        _make_rotation_spec(
            7, "h1", 0.0, 10.0, "SLIGHT",
            font_size_px=56, font_color="#4a5568",
            font_family="'Courier New', monospace", font_weight="normal",
            bg="#ffffff",
        ),
        # 8. 45° → -45° (large symmetric flip), blue on pale
        _make_rotation_spec(
            8, "h1", 45.0, -45.0, "MIRROR",
            font_size_px=64, font_color="#5a67d8",
            font_family="Arial, sans-serif", font_weight="bold",
            bg="#ebf4ff",
        ),
    ]
