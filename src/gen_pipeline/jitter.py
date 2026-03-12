"""
jitter.py — Per-scene style jitter for training diversity.

A SceneJitter is sampled once per scene and shared by both source and target
HTML builders. Source and target differ only in the one edited property; all
jitter values are identical between them so the model learns the edit, not
scene-level noise.

Usage:
    from gen_pipeline.jitter import SceneJitter, sample_jitter, jitter_color

    jitter = sample_jitter(rng)
    shifted = jitter_color("#3B82F6", jitter)
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass


@dataclass
class SceneJitter:
    """All randomized offsets for one scene. Shared by source and target."""

    # Multiplier on each role's base font-size. Caller clamps to 30 px min.
    font_size_scale: float
    # Additive offset (em) on each role's letter-spacing.
    letter_spacing_delta: float
    # Additive offset on each role's line-height.
    line_height_delta: float
    # Multiplier on outer container padding.
    container_padding_scale: float
    # Multiplier on inter-element gaps / margins.
    gap_scale: float
    # Hue rotation in degrees applied to every palette color in the scene.
    hue_delta: float
    # Lightness shift (0–1 scale) applied to every palette color in the scene.
    lightness_delta: float


def sample_jitter(rng: random.Random) -> SceneJitter:
    """Sample a fresh SceneJitter for one scene."""
    return SceneJitter(
        font_size_scale=rng.uniform(0.82, 1.22),
        letter_spacing_delta=rng.uniform(-0.025, 0.025),
        line_height_delta=rng.uniform(-0.15, 0.15),
        container_padding_scale=rng.uniform(0.75, 1.30),
        gap_scale=rng.uniform(0.75, 1.30),
        hue_delta=rng.uniform(-25.0, 25.0),
        lightness_delta=rng.uniform(-0.06, 0.06),
    )


def jitter_color(hex_color: str, jitter: SceneJitter) -> str:
    """Return hex_color shifted by the scene's hue and lightness jitter."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0

    hue, lum, sat = colorsys.rgb_to_hls(r, g, b)
    hue = (hue + jitter.hue_delta / 360.0) % 1.0
    lum = max(0.05, min(0.95, lum + jitter.lightness_delta))

    r2, g2, b2 = colorsys.hls_to_rgb(hue, lum, sat)
    return "#{:02X}{:02X}{:02X}".format(
        round(r2 * 255), round(g2 * 255), round(b2 * 255)
    )
