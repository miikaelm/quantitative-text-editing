"""
color.py — Color change measurement via histogram differencing.

Method:
    1. Extract ROI from source and output images (bbox or OCR-located).
    2. Compute 3D RGB histograms (bin_width=2) for both ROIs.
    3. Subtract: output_hist - source_hist.
    4. Bin with largest positive increase → measured new text color.
    5. ΔE (CIEDE2000) against target → accuracy score.

The histogram diff cancels out background (same in both images),
leaving the text color change as the dominant signal. Works with
solid, gradient, or image backgrounds.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from skimage.color import rgb2lab, deltaE_ciede2000
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BIN_WIDTH = 2
BINS_PER_CHANNEL = 256 // DEFAULT_BIN_WIDTH  # 128


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColorMeasurement:
    """Result of measuring a color change via histogram diff."""
    measured_rgb: tuple[int, int, int]      # mode color from the dominant positive bin
    target_rgb: tuple[int, int, int]        # expected color from metadata
    delta_e: float                          # CIEDE2000 distance
    exact_match: bool                       # measured == target (within bin width)
    peak_bin_count: int                     # how many pixels in the dominant bin shift
    confidence: float                       # peak relative to total positive shift
    # Optional diagnostic info
    measured_hex: str
    target_hex: str


# ---------------------------------------------------------------------------
# Core functions
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
    Compute CIEDE2000 color difference between two RGB colors.
    Returns a float where:
        < 1.0  = imperceptible
        < 3.0  = barely noticeable
        < 5.0  = noticeable
        > 5.0  = clearly different
    """
    # skimage expects (1,1,3) shaped arrays in [0,1] range for rgb2lab
    lab1 = rgb2lab(np.array([[rgb1]], dtype=np.float64) / 255.0)
    lab2 = rgb2lab(np.array([[rgb2]], dtype=np.float64) / 255.0)
    return float(deltaE_ciede2000(lab1, lab2)[0, 0])


def extract_roi_pixels(
    image: NDArray,
    bbox: tuple[int, int, int, int],
) -> NDArray:
    """
    Extract pixels from a bounding box region.

    Args:
        image: H×W×3 uint8 numpy array.
        bbox: (x, y, w, h) in pixels.

    Returns:
        N×3 array of RGB pixel values.
    """
    x, y, w, h = bbox
    # Clamp to image bounds
    img_h, img_w = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)

    roi = image[y0:y1, x0:x1]
    return roi.reshape(-1, 3)


def compute_rgb_histogram(
    pixels: NDArray,
    bin_width: int = DEFAULT_BIN_WIDTH,
) -> NDArray:
    """
    Compute a 3D RGB histogram with the given bin width.

    Args:
        pixels: N×3 uint8 array of RGB values.
        bin_width: width of each bin (default 2).

    Returns:
        3D integer array of shape (n_bins, n_bins, n_bins).
    """
    n_bins = 256 // bin_width
    bin_edges = np.arange(0, 257, bin_width, dtype=np.float64)

    hist, _ = np.histogramdd(
        pixels.astype(np.float64),
        bins=[bin_edges, bin_edges, bin_edges],
    )
    return hist.astype(np.int64)


def find_dominant_color_shift(
    source_hist: NDArray,
    output_hist: NDArray,
    bin_width: int = DEFAULT_BIN_WIDTH,
) -> tuple[tuple[int, int, int], int, float]:
    """
    Find the color that increased most between source and output.

    Args:
        source_hist: 3D histogram of source ROI.
        output_hist: 3D histogram of output ROI.
        bin_width: bin width used for histograms.

    Returns:
        (rgb, peak_count, confidence) where:
            rgb: center of the bin with largest positive increase.
            peak_count: number of pixels in that increase.
            confidence: peak_count / total_positive_increase.
    """
    diff = output_hist - source_hist

    # Only look at positive increases (new colors in output)
    positive_diff = np.maximum(diff, 0)
    total_positive = positive_diff.sum()

    if total_positive == 0:
        # No color change detected
        return (0, 0, 0), 0, 0.0

    # Find the bin with the largest increase
    peak_idx = np.unravel_index(np.argmax(diff), diff.shape)
    peak_count = int(diff[peak_idx])

    # Convert bin index to RGB value (center of the bin)
    rgb = tuple(int(idx * bin_width + bin_width // 2) for idx in peak_idx)

    confidence = peak_count / total_positive if total_positive > 0 else 0.0

    return rgb, peak_count, confidence


# ---------------------------------------------------------------------------
# High-level evaluation function
# ---------------------------------------------------------------------------

def evaluate_color_edit(
    source_image: NDArray,
    output_image: NDArray,
    bbox: tuple[int, int, int, int],
    target_color_hex: str,
    bin_width: int = DEFAULT_BIN_WIDTH,
) -> ColorMeasurement:
    """
    Evaluate a color edit by comparing histogram distributions.

    Uses the ground truth source image to establish background distribution,
    then finds the dominant new color in the output image's ROI.

    Args:
        source_image: H×W×3 uint8 array (ground truth source / before edit).
        output_image: H×W×3 uint8 array (model output or ground truth target).
        bbox: (x, y, w, h) bounding box of the edited text element.
        target_color_hex: expected new color as '#RRGGBB'.
        bin_width: histogram bin width (default 2).

    Returns:
        ColorMeasurement with measured color, target, ΔE, and diagnostics.
    """
    target_rgb = hex_to_rgb(target_color_hex)

    # Extract ROI pixels
    source_pixels = extract_roi_pixels(source_image, bbox)
    output_pixels = extract_roi_pixels(output_image, bbox)

    # Compute histograms
    source_hist = compute_rgb_histogram(source_pixels, bin_width)
    output_hist = compute_rgb_histogram(output_pixels, bin_width)

    plot_top_color_differences(source_hist, output_hist)
    # Find dominant color shift
    measured_rgb, peak_count, confidence = find_dominant_color_shift(
        source_hist, output_hist, bin_width
    )

    # Compute ΔE
    de = compute_delta_e(measured_rgb, target_rgb)

    # Exact match: within one bin width on each channel
    exact = all(abs(m - t) <= bin_width for m, t in zip(measured_rgb, target_rgb))

    return ColorMeasurement(
        measured_rgb=measured_rgb,
        target_rgb=target_rgb,
        delta_e=de,
        exact_match=exact,
        peak_bin_count=peak_count,
        confidence=confidence,
        measured_hex=rgb_to_hex(measured_rgb),
        target_hex=target_color_hex.upper(),
    )


    

def plot_top_color_differences(source_hist, output_hist, bin_width=2, top_n=10):
    """
    Plots a 1D bar chart of the top N added and removed colors.
    Each bar is colored with the exact RGB color it represents.
    """
    diff_hist = output_hist - source_hist
    flat_diff = diff_hist.flatten()
    
    # Find indices of the largest positive (added) and negative (removed) changes
    pos_indices = np.argsort(flat_diff)[-top_n:][::-1]
    neg_indices = np.argsort(flat_diff)[:top_n]
    
    # Filter out zeros (in case there are fewer than top_n changes)
    pos_indices = [idx for idx in pos_indices if flat_diff[idx] > 0]
    neg_indices = [idx for idx in neg_indices if flat_diff[idx] < 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def plot_bars(ax, indices, title, is_positive):
        if not indices:
            ax.set_title(title + " (No changes)")
            return
            
        counts = flat_diff[indices]
        if not is_positive:
            counts = np.abs(counts) # Show magnitudes for removed colors
            
        # Convert 1D flat indices back to 3D coordinates
        coords = np.unravel_index(indices, diff_hist.shape)
        
        rgbs, labels = [], []
        for i in range(len(indices)):
            r = coords[0][i] * bin_width + bin_width // 2
            g = coords[1][i] * bin_width + bin_width // 2
            b = coords[2][i] * bin_width + bin_width // 2
            rgbs.append((r/255., g/255., b/255.))
            labels.append(f"#{r:02X}{g:02X}{b:02X}")
            
        # Plot the bars and apply the actual colors!
        ax.bar(range(len(indices)), counts, color=rgbs, edgecolor='black')
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylabel("Pixel Count Difference")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    plot_bars(ax1, pos_indices, "Added Colors (New Text Color)", is_positive=True)
    plot_bars(ax2, neg_indices, "Removed Colors (Old Text Color)", is_positive=False)
    
    plt.tight_layout()
    plt.show()