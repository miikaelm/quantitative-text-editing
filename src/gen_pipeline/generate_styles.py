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


def _pick_tier(rng: random.Random) -> str:
    """Pick a contrast tier: subtle (pastel), moderate, or bold (vivid/dark)."""
    return rng.choices(["subtle", "moderate", "bold"], weights=[0.40, 0.35, 0.25])[0]


def _solid_color(rng: random.Random, base_h: float, tier: str, hue_shift: float = 0.0) -> str:
    """A solid hex color whose saturation/lightness fits the contrast tier."""
    hue = rotate_hue(base_h, hue_shift + rng.uniform(-20, 20))
    if tier == "subtle":
        s = rng.uniform(0.05, 0.35)
        lit = rng.uniform(0.78, 0.97)
    elif tier == "moderate":
        s = rng.uniform(0.35, 0.70)
        lit = rng.uniform(0.35, 0.78)
    else:  # bold
        s = rng.uniform(0.65, 1.00)
        lit = rng.uniform(0.08, 0.62)
    return hsl_to_hex(hue, clamp(s), clamp(lit))


def _line_opacity(rng: random.Random, tier: str) -> float:
    """Opacity for rgba line/overlay patterns, scaled to tier."""
    if tier == "subtle":
        return round(rng.uniform(0.05, 0.18), 2)
    elif tier == "moderate":
        return round(rng.uniform(0.22, 0.55), 2)
    else:  # bold
        return round(rng.uniform(0.60, 1.00), 2)


def _complex_pattern_options(
    bg: str,
    primary: str,
    accent: str,
    rng: random.Random,
) -> list[str]:
    """
    Return a list of complex CSS pattern strings using multi-layer gradients.
    Each pattern independently picks a contrast tier (subtle / moderate / bold).
    """
    bg_h, bg_s, bg_l = hex_to_hsl(bg)
    pr, pg, pb = hex_to_rgb_tuple(primary)
    ar, ag, ab = hex_to_rgb_tuple(accent)

    patterns: list[str] = []

    # ── Checkerboard (repeating-conic-gradient, two solid hex colors) ──────
    tier = _pick_tier(rng)
    size = rng.choice([20, 28, 36, 44])
    hue_shift2 = rng.uniform(30, 120) if tier == "bold" else rng.uniform(10, 40)
    c1 = _solid_color(rng, bg_h, tier, hue_shift=0.0)
    c2 = _solid_color(rng, bg_h, tier, hue_shift=hue_shift2)
    patterns.append(
        f"repeating-conic-gradient({c1} 0% 25%, {c2} 0% 50%) 0 0 / {size}px {size}px"
    )

    # ── Crosshatch (two repeating-linear-gradient layers at 45°/135°) ─────
    tier = _pick_tier(rng)
    spacing = rng.choice([16, 20, 24, 32])
    thickness = rng.choice([2, 3, 4]) if tier == "bold" else rng.choice([1, 1, 2])
    op = _line_opacity(rng, tier)
    patterns.append(
        f"repeating-linear-gradient(45deg, rgba({pr},{pg},{pb},{op}) 0,"
        f" rgba({pr},{pg},{pb},{op}) {thickness}px,"
        f" transparent {thickness}px, transparent {spacing}px) 0 0 / {spacing}px {spacing}px,"
        f" repeating-linear-gradient(135deg, rgba({pr},{pg},{pb},{op}) 0,"
        f" rgba({pr},{pg},{pb},{op}) {thickness}px,"
        f" transparent {thickness}px, transparent {spacing}px) 0 0 / {spacing}px {spacing}px,"
        f" {bg}"
    )

    # ── Grid lines (horizontal + vertical) ────────────────────────────────
    tier = _pick_tier(rng)
    grid_spacing = rng.choice([20, 28, 36, 48])
    grid_thickness = rng.choice([2, 3, 4]) if tier == "bold" else rng.choice([1, 1, 2])
    op = _line_opacity(rng, tier)
    patterns.append(
        f"repeating-linear-gradient(0deg, rgba({ar},{ag},{ab},{op}) 0,"
        f" rgba({ar},{ag},{ab},{op}) {grid_thickness}px,"
        f" transparent {grid_thickness}px, transparent {grid_spacing}px) 0 0 / {grid_spacing}px {grid_spacing}px,"
        f" repeating-linear-gradient(90deg, rgba({ar},{ag},{ab},{op}) 0,"
        f" rgba({ar},{ag},{ab},{op}) {grid_thickness}px,"
        f" transparent {grid_thickness}px, transparent {grid_spacing}px) 0 0 / {grid_spacing}px {grid_spacing}px,"
        f" {bg}"
    )

    # ── Offset dots / honeycomb-like (solid hex dots, correct CSS tiling) ─
    tier = _pick_tier(rng)
    sx = rng.choice([20, 24, 30, 36])
    dot_r = rng.choice([3, 5, 7, 9]) if tier == "bold" else rng.choice([2, 3, 4, 5])
    dot_color = _solid_color(rng, bg_h, tier, hue_shift=rng.uniform(-40, 40))
    patterns.append(
        f"radial-gradient(circle, {dot_color} {dot_r}px, transparent {dot_r}px) 0 0 / {sx}px {sx}px,"
        f" radial-gradient(circle, {dot_color} {dot_r}px, transparent {dot_r}px) {sx // 2}px {sx // 2}px / {sx}px {sx}px,"
        f" {bg}"
    )

    # ── Diamond / argyle ───────────────────────────────────────────────────
    tier = _pick_tier(rng)
    half = rng.choice([12, 16, 20, 24])
    border = rng.choice([2, 3, 4]) if tier == "bold" else rng.choice([1, 2])
    op = _line_opacity(rng, tier)
    patterns.append(
        f"repeating-linear-gradient(45deg, rgba({pr},{pg},{pb},{op}) 0,"
        f" rgba({pr},{pg},{pb},{op}) {border}px,"
        f" transparent {border}px, transparent {half}px),"
        f" repeating-linear-gradient(-45deg, rgba({ar},{ag},{ab},{op}) 0,"
        f" rgba({ar},{ag},{ab},{op}) {border}px,"
        f" transparent {border}px, transparent {half}px),"
        f" {bg}"
    )

    # ── Zigzag / herringbone ───────────────────────────────────────────────
    tier = _pick_tier(rng)
    zz = rng.choice([10, 14, 18, 22])
    zz_border = rng.choice([2, 3, 4]) if tier == "bold" else rng.choice([1, 2])
    op = _line_opacity(rng, tier)
    patterns.append(
        f"repeating-linear-gradient(135deg, rgba({pr},{pg},{pb},{op}) 0,"
        f" rgba({pr},{pg},{pb},{op}) {zz_border}px,"
        f" transparent {zz_border}px, transparent {zz}px),"
        f" repeating-linear-gradient(45deg, rgba({pr},{pg},{pb},{op}) 0,"
        f" rgba({pr},{pg},{pb},{op}) {zz_border}px,"
        f" transparent {zz_border}px, transparent {zz}px),"
        f" {bg}"
    )

    return patterns


def generate_background_options(palette: dict[str, str], rng: random.Random) -> list[str]:
    """
    Generate 18-24 background options covering:
      - 2 solid colours
      - 3-4 two-stop linear gradients (palette hues, varied angles)
      - 1-2 multi-stop linear gradients (3-5 stops)
      - 1-2 cross-palette gradients (primary → accent or secondary)
      - 1-2 radial gradients
      - 1-2 repeating-linear-gradient (simple stripe patterns)
      - 1   repeating-radial-gradient (dot grid)
      - 0-1 conic gradient
      - 2-4 complex CSS patterns (checkerboard, crosshatch, grid, offset-dots, diamond, zigzag)
      - 5-8 image overlays with varied opacity (0.10–0.75, sometimes near-transparent)
    """
    bg = palette["background"]
    bg_h, bg_s, bg_l = hex_to_hsl(bg)
    primary = palette["primary"]
    accent = palette["accent"]
    secondary = palette["secondary"]

    options: list[str] = []

    # ── 2 solid colours ───────────────────────────────────────────────────
    options.append(bg)
    alt_solid = hsl_to_hex(bg_h, clamp(bg_s + 0.03), clamp(bg_l - 0.02, 0.90, 0.97))
    options.append(alt_solid)

    all_angles = [45, 60, 90, 120, 135, 150, 160, 180, 200, 225, 240, 270, 300, 315]

    # ── 3-4 two-stop linear gradients (tier-aware) ────────────────────────
    num_two_stop = rng.randint(3, 4)
    chosen_angles = rng.sample(all_angles, num_two_stop)
    for angle in chosen_angles:
        tier = _pick_tier(rng)
        hue_shift = rng.uniform(30, 120) if tier == "bold" else rng.uniform(-30, 30)
        c1 = _solid_color(rng, bg_h, tier, hue_shift=0.0)
        c2 = _solid_color(rng, bg_h, tier, hue_shift=hue_shift)
        options.append(f"linear-gradient({angle}deg, {c1} 0%, {c2} 100%)")

    # ── 1-2 multi-stop linear gradients (3-5 colour stops, tier-aware) ────
    for _ in range(rng.randint(1, 2)):
        tier = _pick_tier(rng)
        num_stops = rng.randint(3, 5)
        angle = rng.choice(all_angles)
        stops = []
        for k in range(num_stops):
            pct = round(k * 100 / (num_stops - 1))
            shift = k * (rng.uniform(30, 90) if tier == "bold" else rng.uniform(10, 40))
            stops.append(f"{_solid_color(rng, bg_h, tier, hue_shift=shift)} {pct}%")
        options.append(f"linear-gradient({angle}deg, {', '.join(stops)})")

    # ── 1-2 cross-palette gradients (primary/accent → background) ─────────
    for cross_color in rng.sample([primary, accent, secondary], rng.randint(1, 2)):
        tier = _pick_tier(rng)
        angle = rng.choice(all_angles)
        cr, cg, cb = hex_to_rgb_tuple(cross_color)
        opacity = _line_opacity(rng, tier)
        options.append(
            f"linear-gradient({angle}deg, rgba({cr},{cg},{cb},{opacity}) 0%, {bg} 100%)"
        )

    # ── 1-2 radial gradients (tier-aware) ─────────────────────────────────
    for _ in range(rng.randint(1, 2)):
        tier = _pick_tier(rng)
        pos = rng.choice(["center", "top", "bottom", "top left", "bottom right"])
        c1 = _solid_color(rng, bg_h, tier, hue_shift=rng.uniform(-40, 40))
        c2 = _solid_color(rng, bg_h, tier, hue_shift=rng.uniform(20, 80))
        options.append(f"radial-gradient(ellipse at {pos}, {c1} 0%, {c2} 100%)")

    # ── 1-2 stripe patterns (tier-aware — bold = vivid bee-stripe style) ──
    for _ in range(rng.randint(1, 2)):
        tier = _pick_tier(rng)
        angle = rng.choice([30, 45, 60, 90, 120, 135, 150])
        stripe_size = rng.choice([8, 12, 16, 20, 24])
        c1 = _solid_color(rng, bg_h, tier, hue_shift=0.0)
        # Bold: two vivid contrasting colors; subtle/moderate: one stripe on bg
        if tier == "bold":
            c2 = _solid_color(rng, bg_h, tier, hue_shift=rng.uniform(90, 180))
        else:
            c2 = bg
        half = stripe_size // 2
        options.append(
            f"repeating-linear-gradient({angle}deg,"
            f" {c1} 0px, {c1} {half}px, {c2} {half}px, {c2} {stripe_size}px)"
        )

    # ── 1 dot grid (tier-aware, correct CSS tiling syntax) ────────────────
    tier = _pick_tier(rng)
    dot_size = rng.choice([4, 6, 8, 10]) if tier == "bold" else rng.choice([3, 4, 6])
    spacing = rng.choice([20, 24, 30, 36])
    dot_c = _solid_color(rng, bg_h, tier, hue_shift=rng.uniform(-30, 30))
    options.append(
        f"radial-gradient(circle, {dot_c} {dot_size}px, transparent {dot_size}px)"
        f" 0 0 / {spacing}px {spacing}px,"
        f" {bg}"
    )

    # ── 0-1 conic gradient (tier-aware) ───────────────────────────────────
    if rng.random() < 0.60:
        tier = _pick_tier(rng)
        from_h = rotate_hue(bg_h, rng.uniform(-40, 40))
        num_conic_stops = rng.randint(3, 5)
        hue_step = 360 / num_conic_stops if tier == "bold" else (360 / num_conic_stops) * 0.5
        colors = []
        for k in range(num_conic_stops):
            pct = round(k * 100 / (num_conic_stops - 1))
            colors.append(f"{_solid_color(rng, from_h, tier, hue_shift=k * hue_step)} {pct}%")
        options.append(f"conic-gradient({', '.join(colors)})")

    # ── 5-8 image overlays (beta-distributed opacity, skewed toward low) ─────
    pr, pg, pb = hex_to_rgb_tuple(primary)
    num_images = rng.randint(5, 8)
    img_seeds = rng.sample(range(1, 9999), num_images)
    for seed in img_seeds:
        # Beta(2,5) ≈ mode ~0.17, mean ~0.29 — photo usually dominates.
        raw = rng.betavariate(2, 5)
        opacity = round(0.10 + raw * 0.65, 2)  # maps [0,1] → [0.10, 0.75]
        options.append(
            f"linear-gradient(rgba({pr},{pg},{pb},{opacity}), "
            f"rgba({pr},{pg},{pb},{opacity})), "
            f"url('https://picsum.photos/seed/{seed}/800/600') center/cover"
        )

    # ── 2-4 complex CSS patterns ──────────────────────────────────────────
    all_complex = _complex_pattern_options(bg, primary, accent, rng)
    num_complex = rng.randint(2, 4)
    rng.shuffle(all_complex)
    options.extend(all_complex[:num_complex])

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