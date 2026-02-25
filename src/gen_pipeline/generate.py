"""
Generate source/target HTML pairs for quantitative text editing.

Writes JSONL records and renders PNG images via render.py.
This module is intentionally minimal — only color-change pairs
are supported right now, hand-crafted for pipeline development.
"""

import json
import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass, field

# Make src/ importable so utils.ocr can be found regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from render import Renderer, RenderConfig
from utils.ocr import find_text_bbox
from evaluation.metrics.color import compute_delta_e, hex_to_rgb


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EditPair:
    pair_id: str
    edit_type: str
    source_html: str
    target_html: str
    instruction: str
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "edit_type": self.edit_type,
            "source_html": self.source_html,
            "target_html": self.target_html,
            "instruction": self.instruction,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Hard-coded color-change examples (placeholder until LLM generation is wired)
# ---------------------------------------------------------------------------

def _make_color_pair(
    index: int,
    element: str,
    old_hex: str,
    new_hex: str,
    text: str,
    font_size: str = "64px",
    bg: str = "#f5f5f5",
) -> EditPair:
    pair_id = f"color_{index:03d}"
    
    # Because bg drops directly into "background:{bg};", we can pass 
    # gradients, url()s, or even chain additional CSS rules with semicolons.
    base_style = (
        "margin:0; background:{bg}; display:flex;"
        "justify-content:center; align-items:center; height:100vh;"
    ).format(bg=bg)
    elem_style = f"font-size:{font_size}; font-family:Arial,sans-serif; color:{{color}};"

    def _html(color: str) -> str:
        return (
            "<!DOCTYPE html>"
            "<html><body style=\"{body}\">"
            "<{el} style=\"{el_style}\">{text}</{el}>"
            "</body></html>"
        ).format(
            body=base_style,
            el=element,
            el_style=elem_style.format(color=color),
            text=text,
        )

    def _classify_background(bg: str) -> str:
        if "url(" in bg:
            return "image"
        if "gradient" in bg:
            return "gradient"
        return "solid"

    planned_delta_e = compute_delta_e(hex_to_rgb(old_hex), hex_to_rgb(new_hex))

    return EditPair(
        pair_id=pair_id,
        edit_type="color",
        source_html=_html(old_hex),
        target_html=_html(new_hex),
        instruction=f"Change the {element} color to {new_hex}",
        metadata={
            "element": element,
            "property": "color",
            "old_value": old_hex,
            "new_value": new_hex,
            "text_content": text,
            "background_css": bg, # Added to track the background complexity
            "background_type": _classify_background(bg),
            "planned_delta_e": round(planned_delta_e, 4),
        },
    )

def _make_image_bg_variants(
    base_index: int,
    element: str,
    old_hex: str,
    new_hex: str,
    text: str,
    font_size: str,
    seed_base: str,
    bg_template: str = "url('https://picsum.photos/seed/{seed}/800/600') center/cover",
    num_seeds: int = 5,
) -> list[EditPair]:
    return [
        _make_color_pair(
            index=base_index * 100 + i,
            element=element,
            old_hex=old_hex,
            new_hex=new_hex,
            text=text,
            font_size=font_size,
            bg=bg_template.format(seed=f"{seed_base}{i}"),
        )
        for i in range(num_seeds)
    ]

# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------

HARDCODED_COLOR_PAIRS: list[EditPair] = [
    _make_color_pair(1,  "h1", "#333333", "#3B82F6", "MINIMALIST", font_size="64px"),
    _make_color_pair(2,  "h1", "#111111", "#EF4444", "BOLD", font_size="64px"),
    _make_color_pair(3,  "h2", "#444444", "#10B981", "Emerald Title", font_size="48px"),
    _make_color_pair(4,  "h1", "#222222", "#F59E0B", "AMBER", font_size="64px", bg="#1a1a2e"),
    _make_color_pair(5,  "p",  "#555555", "#8B5CF6", "Body paragraph text sample.", font_size="32px"),
    _make_color_pair(6,  "h1", "#0a0a0a", "#EC4899", "PINK HEADER", font_size="64px"),
    _make_color_pair(7,  "h3", "#333333", "#06B6D4", "Cyan subheading", font_size="40px"),
    _make_color_pair(8,  "h1", "#1a1a1a", "#84CC16", "LIME", font_size="64px", bg="#0f172a"),
    _make_color_pair(9,  "p",  "#666666", "#F97316", "Orange paragraph content.", font_size="30px"),
    _make_color_pair(10, "h2", "#2d2d2d", "#6366F1", "Indigo heading", font_size="52px"),
    _make_color_pair(11, "h1", "#ffffff", "#FCD34D", "LINEAR GRADIENT", font_size="64px", bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"),
    _make_color_pair(12, "h2", "#333333", "#8B5CF6", "Radial Center", font_size="56px", bg="radial-gradient(circle, #ff9a9e 0%, #fecfef 99%, #fecfef 100%)"),
    _make_color_pair(13, "h1", "#111111", "#EF4444", "STRIPES", font_size="64px", bg="repeating-linear-gradient(45deg, #e5e5f7 0%, #e5e5f7 10%, #ffffff 10%, #ffffff 20%)"),
    _make_color_pair(14, "h3", "#222222", "#3B82F6", "Grid Overlay", font_size="48px", bg="#fdfdfd; background-image: linear-gradient(#e5e5e5 1px, transparent 1px), linear-gradient(90deg, #e5e5e5 1px, transparent 1px); background-size: 20px 20px"),
    _make_color_pair(15, "h1", "#ffffff", "#10B981", "CONIC", font_size="64px", bg="conic-gradient(from 90deg, #3f51b5, #00bcd4, #4caf50, #8bc34a, #3f51b5)"),
    _make_color_pair(16, "p", "#444444", "#EC4899", "Soft pastel paragraph.", font_size="36px", bg="linear-gradient(to right, #ffecd2 0%, #fcb69f 100%)"),
    
    *_make_image_bg_variants(17, "h1", "#fffff0", "#F59E0B", "NATURE", "64px", "nature"),
    *_make_image_bg_variants(18, "h2", "#e2e8f0", "#06B6D4", "Cityscape Overlay", "52px", "city", bg_template="linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(19, "h1", "#1a1a1a", "#EAB308", "LIGHT WASH", "64px", "abstract", bg_template="linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(20, "h3", "#ffffff", "#84CC16", "Repeating Pattern", "40px", "pattern", bg_template="#1a1a1a url('https://picsum.photos/seed/{seed}/100/100') repeat"),

    _make_color_pair(21, "h1", "#1a202c", "#F43F5E", "CORAL REEF", font_size="64px", bg="#ffe4e6"),
    _make_color_pair(22, "p", "#e2e8f0", "#38BDF8", "Deep ocean paragraph text.", font_size="30px", bg="#0f172a"),
    _make_color_pair(23, "h2", "#2d3748", "#D97706", "Sunset Boulevard", font_size="44px", bg="linear-gradient(to top, #ffecd2 0%, #fcb69f 100%)"),
    _make_color_pair(24, "h1", "#f8fafc", "#A855F7", "NEON NIGHTS", font_size="64px", bg="#09090b"),
    _make_color_pair(25, "h3", "#064e3b", "#14B8A6", "Forest Canopy Subheading", font_size="38px"),
    _make_color_pair(26, "p", "#475569", "#F97316", "A warm, inviting text snippet for a blog post.", font_size="30px", bg="#fff7ed"),
    _make_color_pair(27, "h1", "#000000", "#EC4899", "VAPORWAVE", font_size="64px", bg="linear-gradient(to right, #4facfe 0%, #00f2fe 100%)"),
    _make_color_pair(28, "h2", "#ffffff", "#E11D48", "Crimson Peak", font_size="50px", bg="radial-gradient(circle at center, #7f0020 0%, #000000 100%)"),
    _make_color_pair(29, "h1", "#1e293b", "#3B82F6", "BLUEPRINT", font_size="64px", bg="#0f172a; background-image: linear-gradient(rgba(59, 130, 246, 0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(59, 130, 246, 0.2) 1px, transparent 1px); background-size: 30px 30px"),
    _make_color_pair(30, "p", "#f1f5f9", "#22C55E", "Eco-friendly sustainability message.", font_size="30px", bg="#14532d"),
    _make_color_pair(31, "h1", "#ffffff", "#F59E0B", "SUNBURST", font_size="64px", bg="repeating-radial-gradient(circle, #fbbf24, #fbbf24 10px, #f59e0b 10px, #f59e0b 20px)"),
    _make_color_pair(32, "h3", "#334155", "#8B5CF6", "Polka Dots", font_size="42px", bg="#ffffff; background-image: radial-gradient(#cbd5e1 20%, transparent 20%); background-size: 20px 20px"),
    _make_color_pair(33, "h2", "#ffffff", "#2DD4BF", "Holographic Title", font_size="54px", bg="conic-gradient(from 180deg at 50% 50%, #2a8af6 0deg, #a853ba 180deg, #e92a67 360deg)"),
    _make_color_pair(34, "p", "#7f1d1d", "#DC2626", "Warning message containing critical alerts.", font_size="30px", bg="repeating-linear-gradient(45deg, #fee2e2, #fee2e2 10px, #fecaca 10px, #fecaca 20px)"),
    _make_color_pair(35, "h1", "#fbbf24", "#94A3B8", "LUXURY", font_size="64px", bg="linear-gradient(135deg, #111827 0%, #374151 100%)"),
    _make_color_pair(36, "h2", "#022c22", "#10B981", "Minty Fresh", font_size="46px", bg="linear-gradient(120deg, #d4fc79 0%, #96e6a1 100%)"),
    _make_color_pair(37, "h1", "#0f172a", "#38BDF8", "WINTER BREEZE", font_size="64px", bg="linear-gradient(to top, #e6e9f0 0%, #eef1f5 100%)"),
    _make_color_pair(38, "p", "#1f2937", "#FBBF24", "Highlight reel text demonstrating readability.", font_size="34px", bg="#fef3c7"),
    _make_color_pair(39, "h3", "#000000", "#FACC15", "Danger Zone", font_size="40px", bg="repeating-linear-gradient(-45deg, #fbbf24, #fbbf24 20px, #000000 20px, #000000 40px)"),
    _make_color_pair(40, "h1", "#ffffff", "#F472B6", "COTTON CANDY", font_size="64px", bg="radial-gradient(circle, #ffc3a0 0%, #ffafbd 100%)"),

    *_make_image_bg_variants(41, "h1", "#f8fafc", "#22C55E", "MOUNTAINS", "64px", "mountain"),
    *_make_image_bg_variants(42, "h2", "#ffffff", "#F97316", "Desert Sand Overlay", "52px", "desert", bg_template="linear-gradient(rgba(120,53,15,0.6), rgba(120,53,15,0.6)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(43, "p", "#e2e8f0", "#60A5FA", "A description of a futuristic tech startup.", "30px", "tech", bg_template="linear-gradient(rgba(15,23,42,0.85), rgba(15,23,42,0.85)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    
    _make_color_pair(44, "h1", "#000000", "#EAB308", "BRUTALISM", font_size="72px", bg="#94a3b8"),
    
    *_make_image_bg_variants(45, "h3", "#ffffff", "#EF4444", "Space Exploration", "44px", "space"),
    *_make_image_bg_variants(46, "p", "#064e3b", "#84CC16", "Organic farming initiatives and community gardens.", "32px", "leaves", bg_template="linear-gradient(rgba(240,2df,244,0.9), rgba(240,253,244,0.9)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(47, "h1", "#ffffff", "#A855F7", "CYBERPUNK", "64px", "neon", bg_template="linear-gradient(135deg, rgba(88,28,135,0.8), rgba(15,23,42,0.9)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(48, "h2", "#ffffff", "#F472B6", "Metropolitan", "48px", "architecture", bg_template="linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(49, "h1", "#0c4a6e", "#0EA5E9", "OCEAN WAVES", "64px", "water", bg_template="linear-gradient(rgba(224,242,254,0.7), rgba(224,242,254,0.7)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(50, "p", "#fef9c3", "#EAB308", "Historical context reflecting golden era architecture.", "30px", "history", bg_template="linear-gradient(rgba(69,26,3,0.8), rgba(69,26,3,0.8)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),

    _make_color_pair(51, "h1", "#22C55E", "#10B981", "THE MATRIX", font_size="64px", bg="linear-gradient(180deg, #022c22 0%, #000000 100%)"),
    _make_color_pair(52, "h2", "#451a03", "#EA580C", "Autumn Leaves", font_size="56px", bg="linear-gradient(120deg, #f6d365 0%, #fda085 100%)"),
    _make_color_pair(53, "h3", "#ffffff", "#4ADE80", "Northern Lights", font_size="46px", bg="conic-gradient(from 225deg, #0f172a, #064e3b, #047857, #0f172a)"),
    _make_color_pair(54, "p", "#4c1d95", "#D946EF", "A romantic or festive paragraph element.", font_size="32px", bg="repeating-radial-gradient(circle, #fce7f3, #fce7f3 15px, #fdf2f8 15px, #fdf2f8 30px)"),
    _make_color_pair(55, "h1", "#ffffff", "#FB7185", "RETRO 80S", font_size="64px", bg="linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); border-bottom: 4px solid #f43f5e"),
    _make_color_pair(56, "h2", "#0f172a", "#3B82F6", "Glassmorphism Concept", font_size="52px", bg="linear-gradient(135deg, rgba(255,255,255,0.4), rgba(255,255,255,0.1))"),
    _make_color_pair(57, "p", "#e0e7ff", "#F59E0B", "Creative agency introductory boilerplate.", font_size="30px", bg="#1e1b4b"),
    
    *_make_image_bg_variants(58, "h1", "#ffffff", "#22C55E", "TROPICAL", "64px", "tropical", bg_template="url('https://picsum.photos/seed/{seed}/200/200') repeat"),

    _make_color_pair(59, "h3", "#1c1917", "#FBBF24", "Sunlight Beam", font_size="42px", bg="linear-gradient(45deg, #fffbeb 0%, #fde68a 50%, #fffbeb 100%)"),
    _make_color_pair(60, "h1", "#f8fafc", "#6366F1", "DEEP SPACE", font_size="64px", bg="radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%)"),
    _make_color_pair(61, "h1", "#1c1c1c", "#2563EB", "Quantum Flux Analyzer", font_size="64px", bg="#f1f5f9"),
    _make_color_pair(62, "p", "#404040", "#0D9488", "The quick brown fox leaps over the completely indifferent dog.", font_size="30px"),
    _make_color_pair(63, "h2", "#f8fafc", "#059669", "Initiating Launch Sequence", font_size="42px", bg="linear-gradient(90deg, #166534 0%, #065f46 100%)"),
    _make_color_pair(64, "h1", "#0b0b0b", "#BE185D", "GALAXY EXPRESS 999", font_size="64px", bg="#fce7f3"),
    _make_color_pair(65, "h3", "#1e293b", "#0284C7", "System Diagnostics: Nominal", font_size="36px"),
    _make_color_pair(66, "p", "#52525B", "#EA580C", "To bake the perfect loaf of bread, you need patience, flour, and a very hot oven.", font_size="30px", bg="#fafaf9"),
    _make_color_pair(67, "h1", "#ffffff", "#D946EF", "SYNTHWAVE RADIO", font_size="64px", bg="linear-gradient(to right, #6b21a8 0%, #312e81 100%)"),
    _make_color_pair(68, "h2", "#fdfdfd", "#B91C1C", "Critical Error 404", font_size="50px", bg="radial-gradient(circle at top left, #991b1b 0%, #450a0a 100%)"),
    _make_color_pair(69, "h1", "#0f172a", "#1D4ED8", "ARCHITECTURE", font_size="64px", bg="#e0e7ff; background-image: repeating-linear-gradient(0deg, transparent, transparent 19px, #c7d2fe 19px, #c7d2fe 20px)"),
    _make_color_pair(70, "p", "#f8fafc", "#16A34A", "Photosynthesis is the process used by plants, algae and certain bacteria.", font_size="30px", bg="#14532d"),
    _make_color_pair(71, "h1", "#333333", "#D97706", "MORNING COFFEE", font_size="64px", bg="repeating-radial-gradient(circle, #fef3c7, #fef3c7 15px, #fde68a 15px, #fde68a 30px)"),
    _make_color_pair(72, "h3", "#3f3f46", "#7C3AED", "Abstract Geometry", font_size="40px", bg="#f4f4f5; background-image: linear-gradient(45deg, #e4e4e7 25%, transparent 25%), linear-gradient(-45deg, #e4e4e7 25%, transparent 25%); background-size: 20px 20px"),
    _make_color_pair(73, "h2", "#ffffff", "#14B8A6", "Prismatic Core", font_size="52px", bg="conic-gradient(from 45deg, #0ea5e9, #8b5cf6, #ec4899, #0ea5e9)"),
    _make_color_pair(74, "p", "#7f1d1d", "#E11D48", "Please ensure all safety harnesses are securely fastened before the ride begins.", font_size="30px", bg="repeating-linear-gradient(-45deg, #fee2e2, #fee2e2 15px, #fecaca 15px, #fecaca 30px)"),
    _make_color_pair(75, "h1", "#fcd34d", "#CBD5E1", "PREMIUM MEMBERSHIP", font_size="64px", bg="linear-gradient(135deg, #18181b 0%, #27272a 100%)"),
    _make_color_pair(76, "h2", "#064e3b", "#059669", "Eucalyptus Leaves", font_size="48px", bg="linear-gradient(120deg, #d9f99d 0%, #bbf7d0 100%)"),
    _make_color_pair(77, "h1", "#020617", "#0284C7", "GLACIER MELT", font_size="64px", bg="linear-gradient(to bottom, #f1f5f9 0%, #e2e8f0 100%)"),
    _make_color_pair(78, "p", "#27272a", "#D97706", "The acoustic resonance of the concert hall was meticulously engineered.", font_size="30px", bg="#fef08a"),
    _make_color_pair(79, "h3", "#111827", "#CA8A04", "Caution: Wet Floor", font_size="38px", bg="repeating-linear-gradient(90deg, #fde047, #fde047 25px, #fef08a 25px, #fef08a 50px)"),
    _make_color_pair(80, "h1", "#ffffff", "#F43F5E", "CHERRY BLOSSOM", font_size="64px", bg="radial-gradient(circle, #fda4af 0%, #fb7185 100%)"),

    *_make_image_bg_variants(81, "h1", "#f1f5f9", "#16A34A", "HIKING TRAILS", "64px", "hike"),
    *_make_image_bg_variants(82, "h2", "#ffffff", "#EA580C", "Martian Landscape", "54px", "mars", bg_template="linear-gradient(rgba(153,27,27,0.7), rgba(153,27,27,0.7)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(83, "p", "#f8fafc", "#3B82F6", "Our new machine learning models have reached 99.8% accuracy on the validation set.", "30px", "data", bg_template="linear-gradient(rgba(30,58,138,0.9), rgba(30,58,138,0.9)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),

    _make_color_pair(84, "h1", "#171717", "#CA8A04", "CONCRETE JUNGLE", font_size="68px", bg="#a1a1aa"),

    *_make_image_bg_variants(85, "h3", "#ffffff", "#DC2626", "Event Horizon", "46px", "galaxy"),
    *_make_image_bg_variants(86, "p", "#065f46", "#65A30D", "Locally sourced organic vegetables delivered straight to your door.", "32px", "farm", bg_template="linear-gradient(rgba(236,253,245,0.85), rgba(236,253,245,0.85)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(87, "h1", "#ffffff", "#9333EA", "NEON CITYSCAPES", "64px", "tokyo", bg_template="linear-gradient(135deg, rgba(107,33,168,0.85), rgba(2,6,23,0.95)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(88, "h2", "#f8fafc", "#F472B6", "Vintage Cinema", "48px", "film", bg_template="linear-gradient(rgba(24,24,27,0.6), rgba(24,24,27,0.6)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(89, "h1", "#082f49", "#0284C7", "PACIFIC TRENCH", "64px", "ocean", bg_template="linear-gradient(rgba(186,230,253,0.75), rgba(186,230,253,0.75)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),
    *_make_image_bg_variants(90, "p", "#fef08a", "#D97706", "The ancient ruins were discovered deep within the uncharted rainforest.", "30px", "ruins", bg_template="linear-gradient(rgba(69,26,3,0.85), rgba(69,26,3,0.85)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),

    _make_color_pair(91, "h1", "#16A34A", "#059669", "TERMINAL ROOT", font_size="64px", bg="linear-gradient(180deg, #064e3b 0%, #020617 100%)"),
    _make_color_pair(92, "h2", "#78350f", "#C2410C", "Campfire Stories", font_size="58px", bg="linear-gradient(120deg, #fcd34d 0%, #fb923c 100%)"),
    _make_color_pair(93, "h3", "#ffffff", "#22C55E", "Aurora Borealis", font_size="48px", bg="conic-gradient(from 300deg, #020617, #065f46, #047857, #020617)"),
    _make_color_pair(94, "p", "#581c87", "#C026D3", "A whimsical journey through the looking glass into a world of pure imagination.", font_size="32px", bg="repeating-radial-gradient(circle, #fae8ff, #fae8ff 20px, #fdf4ff 20px, #fdf4ff 40px)"),
    _make_color_pair(95, "h1", "#ffffff", "#FB7185", "MIAMI NIGHTS", font_size="64px", bg="linear-gradient(135deg, #312e81 0%, #4c1d95 100%); border-left: 8px solid #f43f5e"),
    _make_color_pair(96, "h2", "#1e293b", "#2563EB", "Frosted Glass Overlay", font_size="50px", bg="linear-gradient(135deg, rgba(255,255,255,0.5), rgba(255,255,255,0.15))"),
    _make_color_pair(97, "p", "#e0e7ff", "#D97706", "Welcome to the interactive onboarding module. Please click next to continue.", font_size="30px", bg="#312e81"),
    
    *_make_image_bg_variants(98, "h1", "#ffffff", "#16A34A", "BOTANICAL GARDENS", "64px", "plants", bg_template="url('https://picsum.photos/seed/{seed}/250/250') repeat"),

    _make_color_pair(99, "h3", "#292524", "#D97706", "Golden Hour Glow", font_size="42px", bg="linear-gradient(45deg, #fef3c7 0%, #fde68a 50%, #fef3c7 100%)"),
    _make_color_pair(100, "h1", "#f1f5f9", "#4F46E5", "STELLAR CARTOGRAPHY", font_size="64px", bg="radial-gradient(ellipse at top right, #1e293b 0%, #0f172a 100%)"),

    *_make_image_bg_variants(101, "h1", "#212121", "#FF5722", "RUSTIC CHARM", "64px", "wood"),
    
    _make_color_pair(102, "p", "#E2E8F0", "#14B8A6", "The database migration completed successfully with zero downtime.", font_size="30px", bg="#0F172A"),
    _make_color_pair(103, "h2", "#000000", "#FFD700", "Bumblebee", font_size="44px", bg="repeating-linear-gradient(45deg, #1a1a1a, #1a1a1a 10px, #2a2a2a 10px, #2a2a2a 20px)"),
    _make_color_pair(104, "h3", "#4A044E", "#D946EF", "Fuchsia Dreams", font_size="38px", bg="#FDF4FF"),
    _make_color_pair(105, "p", "#10B981", "#34D399", "admin@server:~$ ./execute_protocol.sh --force", font_size="30px", bg="#000000"),
    _make_color_pair(106, "h1", "#FFFFFF", "#F43F5E", "LAVA FLOW", font_size="64px", bg="linear-gradient(to bottom right, #7F1D1D, #000000)"),
    _make_color_pair(107, "h2", "#1E3A8A", "#60A5FA", "Oceanic Trench", font_size="52px", bg="radial-gradient(circle at 50% 100%, #1D4ED8 0%, #0F172A 100%)"),

    *_make_image_bg_variants(108, "h1", "#FEF3C7", "#D97706", "SEPIA TONES", "64px", "oldphoto", bg_template="linear-gradient(rgba(120,53,15,0.8), rgba(120,53,15,0.8)), url('https://picsum.photos/seed/{seed}/800/600')"),

    _make_color_pair(109, "p", "#333333", "#4F46E5", "User engagement increased by 45% after implementing the new feature set.", font_size="30px", bg="#F3F4F6"),
    _make_color_pair(110, "h3", "#F1F5F9", "#94A3B8", "Steel Interface", font_size="40px", bg="linear-gradient(135deg, #475569 0%, #1E293B 100%)"),

    *_make_image_bg_variants(111, "h1", "#FFFFFF", "#10B981", "VINTAGE BOTANICAL", "64px", "fern", bg_template="linear-gradient(rgba(6,78,59,0.7), rgba(6,78,59,0.7)), url('https://picsum.photos/seed/{seed}/800/600')"),

    _make_color_pair(112, "h2", "#3F3F46", "#F472B6", "Pop Art", font_size="60px", bg="repeating-radial-gradient(circle, #fce7f3, #fce7f3 10px, #fbcfe8 10px, #fbcfe8 20px)"),
    _make_color_pair(113, "p", "#FFFFFF", "#F59E0B", "He stepped into the room, and the temperature immediately dropped by ten degrees.", font_size="30px", bg="conic-gradient(from 180deg, #111827, #374151, #111827)"),
    _make_color_pair(114, "h1", "#111827", "#8B5CF6", "AMETHYST", font_size="64px", bg="#EED2EE; background-image: radial-gradient(#D8BFD8 1px, transparent 1px); background-size: 10px 10px"),
    _make_color_pair(115, "h3", "#FFFFFF", "#06B6D4", "Iced Over", font_size="46px", bg="linear-gradient(to top, #cffafe 0%, #083344 100%)"),
    _make_color_pair(116, "p", "#4C1D95", "#A855F7", "Please verify your email address to unlock premium features and daily rewards.", font_size="30px", bg="#FAF5FF"),
    _make_color_pair(117, "h1", "#000000", "#DC2626", "WARNING SIGN", font_size="80px", bg="repeating-linear-gradient(45deg, #FACC15, #FACC15 30px, #000000 30px, #000000 60px)"),
    _make_color_pair(118, "h2", "#FEF08A", "#EAB308", "Desert Mirage", font_size="48px", bg="linear-gradient(to right, #B45309 0%, #78350F 100%)"),
    _make_color_pair(119, "h1", "#F8FAFC", "#64748B", "MONOLITH", font_size="64px", bg="#0F172A; box-shadow: inset 0 0 50px #000"),
    _make_color_pair(120, "p", "#1E40AF", "#3B82F6", "The quick, bright blue bird darted across the cloudless summer sky.", font_size="32px", bg="#DBEAFE"),
    _make_color_pair(121, "h3", "#FFFFFF", "#EF4444", "Blood Moon", font_size="50px", bg="radial-gradient(circle at center, #991B1B 0%, #000000 100%)"),
    _make_color_pair(122, "h1", "#022C22", "#10B981", "CHLOROPHYLL", font_size="64px", bg="#D1FAE5"),
    _make_color_pair(123, "h2", "#FFFFFF", "#A3E635", "Toxic Sludge", font_size="54px", bg="linear-gradient(135deg, #3F6212 0%, #14532D 100%)"),
    _make_color_pair(124, "p", "#292524", "#A8A29E", "A whisper of wind brushed past the ancient, crumbling stone walls.", font_size="30px", bg="#F5F5F4"),
    _make_color_pair(125, "h1", "#F1F5F9", "#0EA5E9", "STRATOSPHERE", font_size="64px", bg="linear-gradient(to top, #38BDF8 0%, #0284C7 100%)"),
    _make_color_pair(126, "h3", "#111827", "#F97316", "Campfire Embers", font_size="42px", bg="repeating-radial-gradient(circle at bottom, #7C2D12, #7C2D12 10px, #451A03 10px, #451A03 20px)"),
    
    *_make_image_bg_variants(127, "p", "#F3E8FF", "#C026D3", "Magic is just science that we haven't documented properly yet.", "30px", "magic", bg_template="linear-gradient(rgba(49,46,129,0.7), rgba(49,46,129,0.7)), url('https://picsum.photos/seed/{seed}/800/600') center/cover"),

    _make_color_pair(128, "h1", "#000000", "#14B8A6", "CYBER SECURITY", font_size="64px", bg="repeating-linear-gradient(90deg, #CCFBF1 0%, #CCFBF1 5%, #99F6E4 5%, #99F6E4 10%)"),
    _make_color_pair(129, "h2", "#FFF7ED", "#EA580C", "Tangerine Sky", font_size="46px", bg="conic-gradient(from 90deg at 20% 80%, #FED7AA, #F97316, #C2410C)"),
    _make_color_pair(130, "p", "#1F2937", "#6B7280", "Section 4: Hardware specifications and minimum system requirements.", font_size="30px", bg="#E5E7EB"),
    _make_color_pair(131, "h1", "#F8FAFC", "#6366F1", "WORMHOLE", font_size="64px", bg="radial-gradient(circle at 50% 50%, #312E81 0%, #000000 100%)"),
    _make_color_pair(132, "h3", "#7F1D1D", "#B91C1C", "Red Velvet", font_size="55px", bg="#FECACA"),
    _make_color_pair(133, "p", "#E0F2FE", "#0284C7", "Float like a butterfly, sting like a bee.", font_size="36px", bg="#0C4A6E"),
    _make_color_pair(134, "h1", "#27272A", "#D4D4D8", "BRUSHED METAL", font_size="64px", bg="linear-gradient(90deg, #e4e4e7 0%, #f4f4f5 50%, #e4e4e7 100%)"),

    *_make_image_bg_variants(135, "h2", "#FFFFFF", "#FCD34D", "Golden Ratio", "40px", "math", bg_template="linear-gradient(rgba(180,83,9,0.8), rgba(180,83,9,0.8)), url('https://picsum.photos/seed/{seed}/800/600')"),

    _make_color_pair(136, "p", "#4C1D95", "#8B5CF6", "They danced until the stars faded and the morning sun broke the horizon.", font_size="30px", bg="linear-gradient(to right, #EDE9FE 0%, #DDD6FE 100%)"),
    _make_color_pair(137, "h1", "#000000", "#10B981", "CHALKBOARD", font_size="64px", bg="#1C1917; background-image: radial-gradient(#292524 10%, transparent 10%); background-size: 8px 8px"),
    _make_color_pair(138, "h3", "#FCE7F3", "#EC4899", "Bubblegum", font_size="52px", bg="radial-gradient(circle at top right, #DB2777 0%, #831843 100%)"),
    _make_color_pair(139, "p", "#064E3B", "#10B981", "ERROR: Variable 'x' is undefined on line 42 of script.js.", font_size="30px", bg="#D1FAE5; border-left: 6px solid #10B981"),
    _make_color_pair(140, "h1", "#FFFFFF", "#E879F9", "KALEIDOSCOPE", font_size="64px", bg="conic-gradient(from 0deg, #ef4444, #f97316, #eab308, #22c55e, #3b82f6, #a855f7, #ef4444)"),
]

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@dataclass
class GenerateConfig:
    edit_type: str = "color"
    output_root: Path = Path("data")
    # If True, writes to data/<edit_type>/test/ instead of data/<edit_type>/
    test_run: bool = False
    render: RenderConfig = field(default_factory=RenderConfig)


async def generate_pairs(
    pairs: list[EditPair],
    config: GenerateConfig,
) -> Path:
    """Render all pairs and write a JSONL file. Returns the JSONL path."""
    subfolder = "test" if config.test_run else ""
    image_dir = config.output_root / config.edit_type / subfolder / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = config.output_root / config.edit_type / subfolder / "pairs.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    async with Renderer(config.render) as renderer:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                src_result, tgt_result = await renderer.render_pair(
                    source_html=pair.source_html,
                    target_html=pair.target_html,
                    output_dir=image_dir,
                    pair_id=pair.pair_id,
                )

                record = pair.to_record()
                record["source_image"] = str(src_result.image_path)
                record["target_image"] = str(tgt_result.image_path)

                text_content = pair.metadata.get("text_content")
                if text_content:
                    src_bbox = find_text_bbox(src_result.image_path, text_content)
                    tgt_bbox = find_text_bbox(tgt_result.image_path, text_content)
                    if src_bbox:
                        record["metadata"]["source_bbox"] = src_bbox
                    else:
                        print(f"  [{pair.pair_id}] WARNING: OCR could not locate '{text_content}' in source image")
                    if tgt_bbox:
                        record["metadata"]["target_bbox"] = tgt_bbox
                    else:
                        print(f"  [{pair.pair_id}] WARNING: OCR could not locate '{text_content}' in target image")

                if src_result.errors or tgt_result.errors:
                    record["render_errors"] = src_result.errors + tgt_result.errors

                f.write(json.dumps(record) + "\n")
                print(f"  [{pair.pair_id}] {src_result.image_path.name}  {tgt_result.image_path.name}")

    return jsonl_path


def generate_pairs_sync(
    pairs: list[EditPair],
    config: GenerateConfig,
) -> Path:
    """Synchronous wrapper for generate_pairs."""
    return asyncio.run(generate_pairs(pairs, config))


# ---------------------------------------------------------------------------
# Test entry-point  (mirrors render.py's __main__ block)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = GenerateConfig(
        edit_type="color",
        output_root=Path("data"),
        test_run=True,
    )

    print(f"Generating {len(HARDCODED_COLOR_PAIRS)} color-change pairs → data/color/test/")
    jsonl_path = generate_pairs_sync(HARDCODED_COLOR_PAIRS, config)
    print(f"\nDone. JSONL written to: {jsonl_path}")