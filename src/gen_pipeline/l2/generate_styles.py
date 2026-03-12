"""
generate_styles.py — Generate 100 diverse StylePackages as JSONL.

Each package is a coherent design-token system with:
  - A semantic color palette (primary, secondary, accent, background, surface)
  - Font pairings for heading and body roles
  - Background options (solid, gradient, image overlay)
  - Edit alternatives per semantic slot (visually distinct replacements)

Color generation uses HSL manipulation for perceptual coherence:
  - Primary anchors the hue
  - Secondary is a muted/shifted variant
  - Accent contrasts the primary hue
  - Background/surface are light tints derived from the palette
  - Edit alternatives rotate hue significantly to ensure visual distinctness

Run:  python generate_styles.py > style_packages.jsonl
"""

from __future__ import annotations

import colorsys
import json
import random


# ── Font pairings ──────────────────────────────────────────────────────────
# (heading_font, body_font) — web-safe or commonly available
FONT_PAIRINGS = [
    ("'Roboto', Arial, sans-serif", "'Roboto', Arial, sans-serif"),
    ("Georgia, 'Times New Roman', serif", "Georgia, 'Times New Roman', serif"),
    ("'Trebuchet MS', 'Segoe UI', Arial, sans-serif", "'Trebuchet MS', 'Segoe UI', Arial, sans-serif"),
    ("Arial, Helvetica, sans-serif", "Arial, Helvetica, sans-serif"),
    ("'Courier New', Courier, monospace", "'Courier New', Courier, monospace"),
    ("'Lucida Console', Monaco, monospace", "Arial, Helvetica, sans-serif"),
    ("'Palatino Linotype', 'Book Antiqua', Palatino, serif", "'Palatino Linotype', 'Book Antiqua', Palatino, serif"),
    ("Verdana, Geneva, sans-serif", "Verdana, Geneva, sans-serif"),
    ("Impact, 'Arial Black', sans-serif", "Arial, Helvetica, sans-serif"),
    ("'Segoe UI', Tahoma, Geneva, sans-serif", "'Segoe UI', Tahoma, Geneva, sans-serif"),
    ("'Franklin Gothic Medium', Arial, sans-serif", "Verdana, Geneva, sans-serif"),
    ("Garamond, 'Times New Roman', serif", "Garamond, 'Times New Roman', serif"),
    ("'Gill Sans', 'Gill Sans MT', Calibri, sans-serif", "'Gill Sans', 'Gill Sans MT', Calibri, sans-serif"),
    ("'Century Gothic', CenturyGothic, AppleGothic, sans-serif", "Arial, Helvetica, sans-serif"),
    ("Cambria, Georgia, serif", "Cambria, Georgia, serif"),
    ("Tahoma, Geneva, sans-serif", "Tahoma, Geneva, sans-serif"),
    ("'Lucida Sans', 'Lucida Grande', sans-serif", "'Lucida Sans', 'Lucida Grande', sans-serif"),
    ("Constantia, Georgia, serif", "Verdana, Geneva, sans-serif"),
    ("'Bookman Old Style', serif", "'Bookman Old Style', serif"),
    ("'Copperplate', 'Copperplate Gothic Light', fantasy", "Arial, Helvetica, sans-serif"),
]

# ── Theme name parts ───────────────────────────────────────────────────────
THEME_ADJECTIVES = [
    "bold", "soft", "vivid", "muted", "warm", "cool", "deep", "light",
    "rich", "pastel", "earthy", "neon", "dusty", "bright", "dark",
    "faded", "crisp", "smoky", "luminous", "subtle", "stark", "hazy",
    "electric", "gentle", "fierce", "calm", "stormy", "sunny", "misty",
    "frosted",
]

THEME_NOUNS = [
    "ocean", "forest", "sunset", "arctic", "desert", "meadow", "slate",
    "coral", "ember", "moss", "dusk", "dawn", "stone", "clay", "sage",
    "plum", "rust", "storm", "bloom", "frost", "ash", "sand", "night",
    "copper", "jade", "ruby", "amber", "ivory", "steel", "linen",
    "cobalt", "saffron", "indigo", "peach", "tundra", "canyon", "reef",
    "prairie", "glacier", "volcanic",
]


# ── Color utilities ────────────────────────────────────────────────────────

def hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL (h in [0,360], s,l in [0,1]) to hex string."""
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    """Convert hex to HSL (h in [0,360], s,l in [0,1])."""
    hc = hex_color.lstrip("#")
    r, g, b = int(hc[0:2], 16) / 255.0, int(hc[2:4], 16) / 255.0, int(hc[4:6], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s, l


def rotate_hue(h: float, degrees: float) -> float:
    return (h + degrees) % 360


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def hex_to_rgb_tuple(hex_color: str) -> tuple[int, int, int]:
    hc = hex_color.lstrip("#")
    return int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)


# ── Palette generation ─────────────────────────────────────────────────────

def generate_palette(rng: random.Random) -> dict[str, str]:
    """Generate a coherent color palette from a random primary hue."""
    primary_h = rng.uniform(0, 360)
    primary_s = rng.uniform(0.50, 0.85)
    primary_l = rng.uniform(0.25, 0.45)

    # Secondary: shift hue slightly, desaturate
    sec_h = rotate_hue(primary_h, rng.uniform(-30, 30))
    sec_s = clamp(primary_s - rng.uniform(0.10, 0.30))
    sec_l = clamp(primary_l + rng.uniform(0.05, 0.20))

    # Accent: complementary or triadic shift
    accent_shift = rng.choice([120, 150, 180, 210, -120, -150])
    acc_h = rotate_hue(primary_h, accent_shift + rng.uniform(-15, 15))
    acc_s = rng.uniform(0.55, 0.90)
    acc_l = rng.uniform(0.30, 0.50)

    # Background: very light tint from primary hue
    bg_h = primary_h
    bg_s = rng.uniform(0.05, 0.20)
    bg_l = rng.uniform(0.92, 0.97)

    # Surface: near-white, slightly tinted
    surf_h = primary_h
    surf_s = rng.uniform(0.00, 0.10)
    surf_l = rng.uniform(0.96, 1.00)

    return {
        "primary":    hsl_to_hex(primary_h, primary_s, primary_l),
        "secondary":  hsl_to_hex(sec_h, sec_s, sec_l),
        "accent":     hsl_to_hex(acc_h, acc_s, acc_l),
        "background": hsl_to_hex(bg_h, bg_s, bg_l),
        "surface":    hsl_to_hex(surf_h, surf_s, surf_l),
    }


def generate_edit_alternatives(palette: dict[str, str], rng: random.Random) -> dict[str, list[str]]:
    """Generate 3 visually distinct alternatives per editable slot."""
    alts: dict[str, list[str]] = {}
    for slot in ("primary", "secondary", "accent"):
        h, s, l = hex_to_hsl(palette[slot])
        offsets = [90, 180, 270]
        rng.shuffle(offsets)
        candidates = []
        for offset in offsets[:3]:
            new_h = rotate_hue(h, offset + rng.uniform(-20, 20))
            new_s = clamp(s + rng.uniform(-0.15, 0.15), 0.30, 0.90)
            new_l = clamp(l + rng.uniform(-0.10, 0.10), 0.20, 0.55)
            candidates.append(hsl_to_hex(new_h, new_s, new_l))
        alts[slot] = candidates
    return alts


def generate_background_options(palette: dict[str, str], rng: random.Random) -> list[str]:
    """Generate 7-9 background options: solids, gradients, image overlay."""
    bg = palette["background"]
    bg_h, bg_s, bg_l = hex_to_hsl(bg)
    primary = palette["primary"]

    options: list[str] = []

    # 2 solid colors
    options.append(bg)
    alt_solid = hsl_to_hex(bg_h, clamp(bg_s + 0.03), clamp(bg_l - 0.02, 0.90, 0.97))
    options.append(alt_solid)

    # 5-6 gradients
    num_grads = rng.randint(5, 6)
    gradient_angles = rng.sample([90, 120, 135, 150, 160, 180, 200, 225, 270, 315], num_grads)
    for angle in gradient_angles:
        hue_shift = rng.uniform(-30, 30)
        start_h = rotate_hue(bg_h, hue_shift)
        end_h = rotate_hue(bg_h, hue_shift + rng.uniform(10, 40))
        start_s = clamp(bg_s + rng.uniform(0.05, 0.20), 0.05, 0.40)
        end_s = clamp(bg_s + rng.uniform(0.10, 0.35), 0.10, 0.50)
        start_l = clamp(bg_l - rng.uniform(0.0, 0.05), 0.85, 0.97)
        end_l = clamp(bg_l - rng.uniform(0.05, 0.15), 0.75, 0.92)

        c1 = hsl_to_hex(start_h, start_s, start_l)
        c2 = hsl_to_hex(end_h, end_s, end_l)

        if rng.random() < 0.2:  # 20% chance radial
            pos = rng.choice(["center", "top", "bottom", "top left", "bottom right"])
            options.append(f"radial-gradient(ellipse at {pos}, {c1} 0%, {c2} 100%)")
        else:
            options.append(f"linear-gradient({angle}deg, {c1} 0%, {c2} 100%)")

    # 1 image overlay
    pr, pg, pb = hex_to_rgb_tuple(primary)
    opacity = rng.choice(["0.55", "0.60", "0.65", "0.70", "0.75"])
    options.append(
        f"linear-gradient(rgba({pr},{pg},{pb},{opacity}), "
        f"rgba({pr},{pg},{pb},{opacity})), "
        f"url('https://picsum.photos/seed/1/800/600') center/cover"
    )

    return options


def generate_name(rng: random.Random, used_names: set[str]) -> str:
    """Generate a unique descriptive theme name."""
    for _ in range(200):
        adj = rng.choice(THEME_ADJECTIVES)
        noun = rng.choice(THEME_NOUNS)
        name = f"{adj}_{noun}"
        if name not in used_names:
            used_names.add(name)
            return name
    # fallback with counter
    n = f"style_{len(used_names):03d}"
    used_names.add(n)
    return n


def generate_style_package(rng: random.Random, used_names: set[str]) -> dict:
    """Generate one complete style package as a dict."""
    palette = generate_palette(rng)
    font_h, font_b = rng.choice(FONT_PAIRINGS)

    return {
        "name": generate_name(rng, used_names),
        "palette": palette,
        "font_heading": font_h,
        "font_body": font_b,
        "background_options": generate_background_options(palette, rng),
        "edit_alternatives": generate_edit_alternatives(palette, rng),
    }


def main():
    seed = 42
    count = 100
    rng = random.Random(seed)
    used_names: set[str] = set()

    for _ in range(count):
        pkg = generate_style_package(rng, used_names)
        print(json.dumps(pkg, ensure_ascii=False))


if __name__ == "__main__":
    main()