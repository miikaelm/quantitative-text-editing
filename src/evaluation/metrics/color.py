"""
color.py — Color change measurement via template matching and edge-based text segmentation.

Method:
    1. Extract text region from source image using bbox metadata.
    2. Convert to edges (Canny) to get color-invariant template.
    3. Template match against output image to find where text landed (handles small shifts).
    4. In the matched region, use edge detection + flood fill to create text pixel mask.
    5. Take mode RGB of masked text pixels → measured color.
    6. ΔE (CIEDE2000) against target → accuracy score.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from collections import Counter

import cv2
from skimage.color import rgb2lab, deltaE_ciede2000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColorMeasurement:
    """Result of measuring a color change via template matching + edge segmentation."""
    measured_rgb: tuple[int, int, int]
    target_rgb: tuple[int, int, int]
    delta_e: float
    exact_match: bool
    text_pixel_count: int
    template_match_score: float
    match_offset: tuple[int, int]           # (dx, dy) shift from original bbox
    measured_hex: str
    target_hex: str
    edit_completion_ratio: float | None  # None if planned_delta_e not provided
    planned_delta_e: float | None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' or 'RRGGBB' to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert (R, G, B) to '#RRGGBB'."""
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def compute_delta_e(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    """
    CIEDE2000 color difference.
        < 1.0  = imperceptible
        < 3.0  = barely noticeable
        > 5.0  = clearly different
    """
    lab1 = rgb2lab(np.array([[rgb1]], dtype=np.float64) / 255.0)
    lab2 = rgb2lab(np.array([[rgb2]], dtype=np.float64) / 255.0)
    return float(deltaE_ciede2000(lab1, lab2)[0, 0])


def mode_rgb(pixels: NDArray) -> tuple[int, int, int]:
    """
    Get the most common RGB value from an N×3 array.
    """
    keys = (pixels[:, 0].astype(np.int64) * 65536
            + pixels[:, 1].astype(np.int64) * 256
            + pixels[:, 2].astype(np.int64))
    counter = Counter(keys)
    most_common = counter.most_common(1)[0][0]
    r = (most_common >> 16) & 0xFF
    g = (most_common >> 8) & 0xFF
    b = most_common & 0xFF
    return (r, g, b)


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------

def find_text_in_output(
    source_img: NDArray,
    output_img: NDArray,
    bbox: tuple[int, int, int, int],
    search_margin: int = 20,
) -> tuple[tuple[int, int, int, int], float, tuple[int, int]]:
    """
    Locate the text region in the output image using edge-based template matching.

    Args:
        source_img: H×W×3 uint8 source image.
        output_img: H×W×3 uint8 output image.
        bbox: (x, y, w, h) original text bounding box.
        search_margin: pixels to expand search area beyond original bbox.

    Returns:
        (matched_bbox, match_score, offset) where:
            matched_bbox: (x, y, w, h) in output image.
            match_score: normalized cross-correlation (0-1).
            offset: (dx, dy) shift from original position.
    """
    x, y, w, h = bbox
    img_h, img_w = source_img.shape[:2]

    # Extract source template and convert to edges
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
    template = source_gray[y:y+h, x:x+w]
    template_edges = cv2.Canny(template, 50, 150)

    # Define search region in output (original bbox + margin)
    search_x0 = max(0, x - search_margin)
    search_y0 = max(0, y - search_margin)
    search_x1 = min(img_w, x + w + search_margin)
    search_y1 = min(img_h, y + h + search_margin)

    output_gray = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
    search_region = output_gray[search_y0:search_y1, search_x0:search_x1]
    search_edges = cv2.Canny(search_region, 50, 150)

    # Template match on edge maps
    if (search_edges.shape[0] < template_edges.shape[0]
            or search_edges.shape[1] < template_edges.shape[1]):
        return bbox, 0.0, (0, 0)

    result = cv2.matchTemplate(search_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # max_loc is (x, y) within search_region
    matched_x = search_x0 + max_loc[0]
    matched_y = search_y0 + max_loc[1]
    offset = (matched_x - x, matched_y - y)

    return (matched_x, matched_y, w, h), float(max_val), offset


# ---------------------------------------------------------------------------
# Text pixel segmentation
# ---------------------------------------------------------------------------

def segment_text_pixels(
    image_region: NDArray,
) -> NDArray:
    """
    Create a binary mask of text pixels using Otsu thresholding.

    For a text ROI, there are two populations: text and background.
    Otsu's method finds the optimal threshold to separate them.
    We then determine which side is text by checking which population
    is smaller (text occupies less area than background in a bbox).

    Args:
        image_region: H×W×3 uint8 ROI.

    Returns:
        H×W boolean mask where True = text pixel.
    """
    gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)

    # Otsu threshold — automatically finds the best split between two populations
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Text is the smaller population (less area than background in a bbox)
    white_count = np.sum(binary == 255)
    black_count = np.sum(binary == 0)

    if black_count < white_count:
        # Dark text on light background — text is the black pixels
        mask = binary == 0
    else:
        # Light text on dark background — text is the white pixels
        mask = binary == 255

    return mask


# ---------------------------------------------------------------------------
# High-level evaluation
# ---------------------------------------------------------------------------

def evaluate_color_edit(
    source_image: NDArray,
    output_image: NDArray,
    bbox: tuple[int, int, int, int],
    target_color_hex: str,
    search_margin: int = 20,
    planned_delta_e: float | None = None,
) -> ColorMeasurement:
    """
    Evaluate a color edit by finding text via template matching and reading its color.

    Args:
        source_image: H×W×3 uint8 (ground truth source / before edit).
        output_image: H×W×3 uint8 (model output or ground truth target).
        bbox: (x, y, w, h) original bounding box of edited text.
        target_color_hex: expected new color as '#RRGGBB'.
        search_margin: pixels to search around original bbox for shifted text.

    Returns:
        ColorMeasurement with measured color, ΔE, and diagnostics.
    """
    target_rgb = hex_to_rgb(target_color_hex)

    # Step 1: Find where the text is in the output image
    matched_bbox, match_score, offset = find_text_in_output(
        source_image, output_image, bbox, search_margin
    )

    # Step 2: Extract the matched region from output
    mx, my, mw, mh = matched_bbox
    img_h, img_w = output_image.shape[:2]
    rx0 = max(0, mx)
    ry0 = max(0, my)
    rx1 = min(img_w, mx + mw)
    ry1 = min(img_h, my + mh)
    output_region = output_image[ry0:ry1, rx0:rx1]

    # Step 3: Segment text pixels using edges
    text_mask = segment_text_pixels(output_region)
    text_pixel_count = int(text_mask.sum())

    # Step 4: Extract color from text pixels
    if text_pixel_count == 0:
        return ColorMeasurement(
            measured_rgb=(0, 0, 0),
            target_rgb=target_rgb,
            delta_e=100.0,
            exact_match=False,
            text_pixel_count=0,
            template_match_score=match_score,
            match_offset=offset,
            measured_hex="#000000",
            target_hex=target_color_hex.upper(),
        )

    text_pixels = output_region[text_mask]  # N×3
    measured_rgb = mode_rgb(text_pixels)

    # Step 5: Compute ΔE
    de = compute_delta_e(measured_rgb, target_rgb)
    exact = (measured_rgb == target_rgb)

    ecr = _compute_ecr(de, planned_delta_e)

    return ColorMeasurement(
        measured_rgb=measured_rgb,
        target_rgb=target_rgb,
        delta_e=de,
        exact_match=exact,
        text_pixel_count=text_pixel_count,
        template_match_score=match_score,
        match_offset=offset,
        measured_hex=rgb_to_hex(measured_rgb),
        target_hex=target_color_hex.upper(),
        edit_completion_ratio=ecr,
        planned_delta_e=planned_delta_e,
    )

def _compute_ecr(
    delta_e_measured_target: float,
    planned_delta_e: float | None,
    floor: float = 1.0,
) -> float | None:
    """
    Edit Completion Ratio.

        ECR = 1 - (ΔE(measured, target) / ΔE(original, target))

        1.0  = perfect edit
        0.0  = model did nothing (output ≈ original)
        <0   = moved in wrong direction
        >1   = overshot

    Returns None if planned_delta_e is unavailable or below floor
    (near-identity edits where the ratio is unstable).
    """
    if planned_delta_e is None or planned_delta_e < floor:
        return None
    return round(1.0 - (delta_e_measured_target / planned_delta_e), 4)