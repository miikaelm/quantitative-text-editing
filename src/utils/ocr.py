"""
OCR utilities shared by the generation pipeline and evaluation metrics.

Primary entry-point: find_text_bbox(image_path, text_content) → dict | None
Returns {x, y, width, height} in pixels (top-left origin).

The EasyOCR reader is lazy-initialized and module-level cached so it is only
loaded once per process.
"""

import re
from pathlib import Path

import easyocr

_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    """Lazy-initialize the EasyOCR reader (downloads/loads model on first call)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _normalize(s: str) -> str:
    """Strip punctuation and lowercase for fuzzy text matching."""
    return re.sub(r"[^\w\s]", "", s).lower().strip()


def _bbox_from_pts(pts: list) -> dict:
    """Convert EasyOCR polygon points to an axis-aligned {x, y, width, height} dict."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x, y = min(xs), min(ys)
    return {
        "x": int(x),
        "y": int(y),
        "width": int(max(xs) - x),
        "height": int(max(ys) - y),
    }


def _merge_bboxes(bboxes: list[dict]) -> dict:
    x = min(b["x"] for b in bboxes)
    y = min(b["y"] for b in bboxes)
    right = max(b["x"] + b["width"] for b in bboxes)
    bottom = max(b["y"] + b["height"] for b in bboxes)
    return {"x": x, "y": y, "width": right - x, "height": bottom - y}


def find_text_bbox(
    image_path: Path | str,
    text_content: str,
) -> dict | None:
    """
    Find the bounding box of text_content in an image using OCR.

    Returns {x, y, width, height} in pixels (top-left origin), or None if not found.
    Matching is case-insensitive and punctuation-tolerant.

    For multi-word text that spans multiple OCR result regions, consecutive regions
    whose concatenated text matches are merged into a single bounding box.
    """
    reader = _get_reader()
    # results: list of (polygon_pts, text, confidence)
    results = reader.readtext(str(image_path))

    target_norm = _normalize(text_content)
    if not target_norm:
        return None

    # Single-region match (most common for centered single-element layouts)
    for pts, text, _conf in results:
        if _normalize(text) == target_norm:
            return _bbox_from_pts(pts)

    # Multi-region match: sliding window over consecutive OCR regions
    regions = [(pts, text) for pts, text, _ in results]
    for start in range(len(regions)):
        for end in range(start + 1, len(regions) + 1):
            combined = " ".join(_normalize(t) for _, t in regions[start:end])
            if combined == target_norm:
                bboxes = [_bbox_from_pts(pts) for pts, _ in regions[start:end]]
                return _merge_bboxes(bboxes)

    return None
