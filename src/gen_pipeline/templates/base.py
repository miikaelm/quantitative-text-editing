"""
templates/base.py — Shared HTML template utilities.
"""


def classify_background(bg: str) -> str:
    """Classify a CSS background value as 'image', 'gradient', or 'solid'."""
    if "url(" in bg:
        return "image"
    if "gradient" in bg:
        return "gradient"
    return "solid"
