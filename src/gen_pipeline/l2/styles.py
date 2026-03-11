"""
l2/styles.py — Style packages (design token system) for Level 2 scenes.

A StylePackage groups:
  - palette:            semantic color slots (primary, secondary, accent, background, surface)
  - font_heading:       CSS font-family for heading roles
  - font_body:          CSS font-family for body/detail roles
  - background_options: CSS background values to randomly select from
  - edit_alternatives:  per-slot, 2-3 visually distinct replacement colors for color edits

Three packages are provided, inspired by Material Design, warm earth tones,
and Nordic minimalism. Adding a new package requires no changes elsewhere.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass, field


@dataclass
class StylePackage:
    """A coherent set of design tokens for a multi-element scene."""
    name: str
    # Semantic color palette slots used by layout role mappings.
    palette: dict[str, str]   # keys: primary, secondary, accent, background, surface
    font_heading: str          # CSS font-family for heading/title roles
    font_body: str             # CSS font-family for body/detail roles
    # CSS background values: solid hex, gradient string, or picsum image URL.
    background_options: list[str]
    # For color edits: 2-3 alternative values per semantic slot.
    # Alternatives should be visually distinct from the base but still plausible.
    edit_alternatives: dict[str, list[str]]

    def pick_background(self, rng: random.Random | None = None) -> str:
        r = rng or random
        chosen = r.choice(self.background_options)
        # Randomize picsum seed so each render gets a different photo.
        if "picsum.photos/seed/" in chosen:
            rand_seed = r.randint(1, 9999)
            chosen = re.sub(r"picsum\.photos/seed/[^/]+/", f"picsum.photos/seed/{rand_seed}/", chosen)
        return chosen


# ---------------------------------------------------------------------------
# Package 1: Material Modern
# Inspired by Material Design 3 — blues, cyans, blue-greys on light surfaces.
# ---------------------------------------------------------------------------
MATERIAL_MODERN = StylePackage(
    name="material_modern",
    palette={
        "primary":    "#1565C0",   # deep blue (M3 primary)
        "secondary":  "#546E7A",   # blue-grey 600
        "accent":     "#00838F",   # cyan 800
        "background": "#FAFAFA",
        "surface":    "#FFFFFF",
    },
    font_heading="'Roboto', Arial, sans-serif",
    font_body="'Roboto', Arial, sans-serif",
    background_options=[
        "#FAFAFA",
        "#F5F5F5",
        "linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%)",
        "linear-gradient(160deg, #ECEFF1 0%, #CFD8DC 100%)",
        "linear-gradient(90deg, #E8EAF6 0%, #C5CAE9 100%)",
        "linear-gradient(180deg, #E0F7FA 0%, #B2EBF2 100%)",
        "linear-gradient(120deg, #F3E5F5 0%, #E1BEE7 100%)",
        "radial-gradient(ellipse at center, #E3F2FD 0%, #90CAF9 100%)",
        "linear-gradient(rgba(255,255,255,0.88), rgba(255,255,255,0.88)), url('https://picsum.photos/seed/1/800/600') center/cover",
    ],
    edit_alternatives={
        "primary":   ["#1B5E20", "#4A148C", "#B71C1C"],   # green / purple / red
        "secondary": ["#37474F", "#455A64", "#607D8B"],   # darker/lighter blue-grey
        "accent":    ["#E65100", "#6A1B9A", "#2E7D32"],   # orange / purple / dark-green
    },
)


# ---------------------------------------------------------------------------
# Package 2: Warm Earth
# Inspired by earthy browns, ambers, and terracotta on warm off-white surfaces.
# ---------------------------------------------------------------------------
WARM_EARTH = StylePackage(
    name="warm_earth",
    palette={
        "primary":    "#4E342E",   # brown darken-3
        "secondary":  "#6D4C41",   # brown darken-2
        "accent":     "#E65100",   # deep orange darken-3
        "background": "#FBE9E7",   # deep orange lighten-5
        "surface":    "#FFF3E0",   # orange lighten-5
    },
    font_heading="Georgia, 'Times New Roman', serif",
    font_body="Georgia, 'Times New Roman', serif",
    background_options=[
        "#FBE9E7",
        "#FFF3E0",
        "linear-gradient(120deg, #FFECD2 0%, #FCB69F 100%)",
        "linear-gradient(135deg, #FFF8E1 0%, #FFE0B2 100%)",
        "linear-gradient(150deg, #FFECB3 0%, #FFD54F 100%)",
        "linear-gradient(90deg, #FBE9E7 0%, #FFCCBC 100%)",
        "linear-gradient(180deg, #FFF3E0 0%, #FFE0B2 100%)",
        "radial-gradient(ellipse at top, #FFECD2 0%, #FFAB91 100%)",
        "linear-gradient(rgba(78,52,46,0.55), rgba(78,52,46,0.55)), url('https://picsum.photos/seed/1/800/600') center/cover",
    ],
    edit_alternatives={
        "primary":   ["#1A237E", "#1B5E20", "#880E4F"],   # dark blue / dark green / dark pink
        "secondary": ["#795548", "#8D6E63", "#A1887F"],   # brown variants
        "accent":    ["#558B2F", "#00695C", "#AD1457"],   # olive / teal / pink
    },
)


# ---------------------------------------------------------------------------
# Package 3: Nordic Minimal
# Inspired by Scandinavian design — dark teal-greys, teal accent, cool stone.
# ---------------------------------------------------------------------------
NORDIC_MINIMAL = StylePackage(
    name="nordic_minimal",
    palette={
        "primary":    "#263238",   # blue-grey 900
        "secondary":  "#546E7A",   # blue-grey 600
        "accent":     "#00897B",   # teal 600
        "background": "#ECEFF1",   # blue-grey 50
        "surface":    "#FFFFFF",
    },
    font_heading="'Trebuchet MS', 'Segoe UI', Arial, sans-serif",
    font_body="'Trebuchet MS', 'Segoe UI', Arial, sans-serif",
    background_options=[
        "#ECEFF1",
        "#F5F5F5",
        "linear-gradient(135deg, #ECEFF1 0%, #CFD8DC 100%)",
        "linear-gradient(160deg, #E0F2F1 0%, #B2DFDB 100%)",
        "linear-gradient(90deg, #E8F5E9 0%, #C8E6C9 100%)",
        "linear-gradient(120deg, #F5F5F5 0%, #E0E0E0 100%)",
        "linear-gradient(180deg, #E8EAF6 0%, #C5CAE9 100%)",
        "radial-gradient(ellipse at bottom, #ECEFF1 0%, #B0BEC5 100%)",
        "linear-gradient(rgba(38,50,56,0.65), rgba(38,50,56,0.65)), url('https://picsum.photos/seed/1/800/600') center/cover",
    ],
    edit_alternatives={
        "primary":   ["#1A237E", "#4A148C", "#880E4F"],   # dark blue / dark purple / dark pink
        "secondary": ["#607D8B", "#78909C", "#B0BEC5"],   # blue-grey variants
        "accent":    ["#E65100", "#C62828", "#283593"],   # orange / red / indigo
    },
)


# ---------------------------------------------------------------------------
# Public collection
# ---------------------------------------------------------------------------

STYLE_PACKAGES: list[StylePackage] = [MATERIAL_MODERN, WARM_EARTH, NORDIC_MINIMAL]
