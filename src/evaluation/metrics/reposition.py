"""
metrics/reposition.py — Reposition edit measurement.

evaluate_reposition_edit() checks whether a text bounding box falls within
the expected zone for a given position name (top-left, center, etc.).

Used by:
  - validate.py (pipeline validation): checks the rendered ground-truth target bbox
  - evaluate.py (model evaluation):   checks the OCR bbox of model output

Zone definitions:
  Each named position maps to a rectangular region covering 50% of the image
  in each axis — generous enough to tolerate varying text widths while still
  distinguishing all nine grid positions.

  The zone center is also computed so callers can report pixel distance from
  the text centroid to where it nominally "should" be.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Position zones — fractional (x0, y0, x1, y1) of image dimensions
# ---------------------------------------------------------------------------

POSITION_ZONES: dict[str, tuple[float, float, float, float]] = {
    #                    x0    y0    x1    y1
    "top-left":       (0.0,  0.0,  0.5,  0.5),
    "top-center":     (0.25, 0.0,  0.75, 0.5),
    "top-right":      (0.5,  0.0,  1.0,  0.5),
    "middle-left":    (0.0,  0.25, 0.5,  0.75),
    "center":         (0.25, 0.25, 0.75, 0.75),
    "middle-right":   (0.5,  0.25, 1.0,  0.75),
    "bottom-left":    (0.0,  0.5,  0.5,  1.0),
    "bottom-center":  (0.25, 0.5,  0.75, 1.0),
    "bottom-right":   (0.5,  0.5,  1.0,  1.0),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RepositionMeasurement:
    """Measurement result for a single reposition check."""
    centroid: tuple[float, float]       # (cx, cy) of the measured text bbox, in pixels
    zone_center: tuple[float, float]    # center of the expected zone, in pixels
    pixel_distance: float               # Euclidean distance from centroid to zone center
    in_correct_zone: bool               # True if centroid falls inside the expected zone
    target_position: str                # The position that was checked against


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bbox_centroid(bbox: dict) -> tuple[float, float]:
    """Return (cx, cy) of a {x, y, width, height} bbox dict."""
    return (bbox["x"] + bbox["width"] / 2.0, bbox["y"] + bbox["height"] / 2.0)


def zone_bounds(
    position: str,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Return (x0, y0, x1, y1) zone bounds in pixels for the given position."""
    x0f, y0f, x1f, y1f = POSITION_ZONES[position]
    return (x0f * img_width, y0f * img_height, x1f * img_width, y1f * img_height)


def zone_center(
    position: str,
    img_width: int,
    img_height: int,
) -> tuple[float, float]:
    """Return (cx, cy) center of the zone for the given position, in pixels."""
    x0, y0, x1, y1 = zone_bounds(position, img_width, img_height)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# ---------------------------------------------------------------------------
# Primary metric function
# ---------------------------------------------------------------------------

def evaluate_reposition_edit(
    bbox: dict,
    target_position: str,
    img_width: int = 512,
    img_height: int = 512,
) -> RepositionMeasurement:
    """
    Check whether a text bounding box is in the correct zone for target_position.

    Args:
        bbox:            {x, y, width, height} of the text element (from OCR or metadata).
        target_position: Expected position name (must be a key in POSITION_ZONES).
        img_width:       Image width in pixels (default 512, matching RenderConfig).
        img_height:      Image height in pixels.

    Returns:
        RepositionMeasurement with in_correct_zone flag and pixel_distance to zone center.

    Raises:
        ValueError: If target_position is not a recognised position name.
    """
    if target_position not in POSITION_ZONES:
        raise ValueError(
            f"Unknown position: {target_position!r}. "
            f"Valid positions: {sorted(POSITION_ZONES)}"
        )

    cx, cy = bbox_centroid(bbox)
    x0, y0, x1, y1 = zone_bounds(target_position, img_width, img_height)
    zc = zone_center(target_position, img_width, img_height)

    in_zone = (x0 <= cx <= x1) and (y0 <= cy <= y1)
    dist = ((cx - zc[0]) ** 2 + (cy - zc[1]) ** 2) ** 0.5

    return RepositionMeasurement(
        centroid=(round(cx, 1), round(cy, 1)),
        zone_center=(round(zc[0], 1), round(zc[1], 1)),
        pixel_distance=round(dist, 2),
        in_correct_zone=in_zone,
        target_position=target_position,
    )
