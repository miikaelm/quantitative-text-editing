"""
styles.py — Style packages loaded from JSONL.

Loads style packages from a JSONL file (one JSON object per line),
each containing: name, palette, font_heading, font_body,
background_options, edit_alternatives.

Usage:
    from gen_pipeline.styles import STYLE_PACKAGES, StylePackage

    pkg = random.choice(STYLE_PACKAGES)
    bg  = pkg.pick_background()
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StylePackage:
    """A coherent set of design tokens for a multi-element scene."""
    name: str
    palette: dict[str, str]        # keys: primary, secondary, accent, background, surface
    font_heading: str              # CSS font-family for heading/title roles
    font_body: str                 # CSS font-family for body/detail roles
    background_options: list[str]  # CSS background values
    edit_alternatives: dict[str, list[str]]  # per-slot replacement colors

    def pick_background(self, rng: random.Random | None = None) -> str:
        r = rng or random
        chosen = r.choice(self.background_options)
        # Randomize picsum seed so each render gets a different photo.
        if "picsum.photos/seed/" in chosen:
            rand_seed = r.randint(1, 9999)
            chosen = re.sub(
                r"picsum\.photos/seed/[^/]+/",
                f"picsum.photos/seed/{rand_seed}/",
                chosen,
            )
        return chosen

    @classmethod
    def from_dict(cls, d: dict) -> StylePackage:
        return cls(
            name=d["name"],
            palette=d["palette"],
            font_heading=d["font_heading"],
            font_body=d["font_body"],
            background_options=d["background_options"],
            edit_alternatives=d["edit_alternatives"],
        )


def load_style_packages(
    path: str | Path | None = None,
) -> list[StylePackage]:
    """Load style packages from a JSONL file.

    Args:
        path: Path to the .jsonl file. Defaults to
              ``style_packages.jsonl`` next to this module.

    Returns:
        List of StylePackage instances.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "style_packages.jsonl"
    else:
        path = Path(path)

    packages: list[StylePackage] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                packages.append(StylePackage.from_dict(d))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid style package on line {line_num} of {path}: {exc}"
                ) from exc

    if not packages:
        raise ValueError(f"No style packages found in {path}")

    return packages


# ── Eager load on import (same interface as before) ────────────────────────
STYLE_PACKAGES: list[StylePackage] = load_style_packages()