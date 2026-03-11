"""
l2/content.py — Pre-generated text content pools for Level 2 scenes.

Each layout's role set has a pool of ContentSets. The generation code samples
from the pool (with replacement) to populate scenes.

Constraints enforced here:
  - All text strings within a single ContentSet are distinct.
  - No string is a substring of another string in the same set.
    (Required by OCR-based find_text_bbox, which matches by text content.)
  - Content is generic and varied; semantic coherence between roles is nice
    but not required.
"""

from __future__ import annotations

# A ContentSet maps role name → text string for one scene instance.
ContentSet = dict[str, str]


# ---------------------------------------------------------------------------
# title + subtitle  (12 sets)
# ---------------------------------------------------------------------------
TITLE_SUBTITLE: list[ContentSet] = [
    {"title": "NORTHERN LIGHTS",  "subtitle": "A guide to arctic photography"},
    {"title": "ARCHITECTURE",     "subtitle": "Form follows function"},
    {"title": "VELOCITY",         "subtitle": "Built for speed and precision"},
    {"title": "DEEP OCEAN",       "subtitle": "Exploring the unknown depths"},
    {"title": "MOMENTUM",         "subtitle": "Where science meets design"},
    {"title": "ZENITH",           "subtitle": "Reaching the highest point"},
    {"title": "SOLSTICE",         "subtitle": "Celebrating the longest day"},
    {"title": "REFRACTION",       "subtitle": "Light bends through glass"},
    {"title": "PARALLAX",         "subtitle": "Depth from different perspectives"},
    {"title": "CADENCE",          "subtitle": "Rhythm in every detail"},
    {"title": "FRACTAL",          "subtitle": "Infinite patterns in nature"},
    {"title": "MERIDIAN",         "subtitle": "Where two worlds meet"},
]


# ---------------------------------------------------------------------------
# title + byline  (12 sets)
# Short bylines (issue numbers, edition labels) are intentionally brief
# so they fit in the absolutely positioned corner without overflow.
# ---------------------------------------------------------------------------
TITLE_BYLINE: list[ContentSet] = [
    {"title": "HORIZON",    "byline": "Issue 12"},
    {"title": "SPECTRUM",   "byline": "Vol. 3"},
    {"title": "BLUEPRINT",  "byline": "Draft 07"},
    {"title": "CHRONICLE",  "byline": "Spring Edition"},
    {"title": "MANIFEST",   "byline": "Revised 2024"},
    {"title": "DISPATCH",   "byline": "Final Release"},
    {"title": "CODEX",      "byline": "Chapter 4"},
    {"title": "ATLAS",      "byline": "Map Series"},
    {"title": "EPOCH",      "byline": "Year Two"},
    {"title": "VECTOR",     "byline": "Preview Build"},
    {"title": "NEXUS",      "byline": "Beta Version"},
    {"title": "PRISM",      "byline": "Limited Run"},
]


# ---------------------------------------------------------------------------
# header + body  (12 sets)
# Body text is a single sentence — long enough to be realistic, short enough
# to fit on one line at 24px in an 800px-wide container.
# ---------------------------------------------------------------------------
HEADER_BODY: list[ContentSet] = [
    {"header": "System Status",   "body": "All services operating within normal parameters"},
    {"header": "Project Update",  "body": "Phase two deployment completed successfully"},
    {"header": "Research Notes",  "body": "Preliminary results indicate strong correlation"},
    {"header": "Design Review",   "body": "Revisions approved for production release"},
    {"header": "Field Report",    "body": "Data collection concluded at seventeen sites"},
    {"header": "Weekly Summary",  "body": "Targets met across all major verticals"},
    {"header": "Configuration",   "body": "Default settings restored to factory values"},
    {"header": "Analysis",        "body": "Sample variance within acceptable tolerance range"},
    {"header": "Announcement",    "body": "New policy effective from the first of next month"},
    {"header": "Inspection Log",  "body": "Minor discrepancy flagged for follow-up review"},
    {"header": "Release Notes",   "body": "Bug fixes and performance improvements included"},
    {"header": "Overview",        "body": "Three core modules remain under active development"},
]


# ---------------------------------------------------------------------------
# name + job_title + organization  (12 sets)
# All three strings in each set are distinct and non-overlapping.
# ---------------------------------------------------------------------------
NAME_CARD: list[ContentSet] = [
    {"name": "Elena Vasquez",   "job_title": "Senior Architect",    "organization": "Nordic Design Studio"},
    {"name": "Marcus Chen",     "job_title": "Lead Engineer",       "organization": "Apex Systems"},
    {"name": "Ava Lindqvist",   "job_title": "Creative Director",   "organization": "Studio Forma"},
    {"name": "Dmitri Volkov",   "job_title": "Research Analyst",    "organization": "Beacon Institute"},
    {"name": "Priya Nair",      "job_title": "Product Manager",     "organization": "Clearpath Tech"},
    {"name": "James Okafor",    "job_title": "Data Scientist",      "organization": "Meridian Labs"},
    {"name": "Sofia Reyes",     "job_title": "UX Designer",         "organization": "Tangent Studio"},
    {"name": "Luca Moretti",    "job_title": "Systems Architect",   "organization": "Forge Solutions"},
    {"name": "Ingrid Bergmann", "job_title": "Operations Lead",     "organization": "Polar Ventures"},
    {"name": "Kenji Watanabe",  "job_title": "Frontend Developer",  "organization": "Pixel Craft"},
    {"name": "Amara Osei",      "job_title": "Business Analyst",    "organization": "Summit Advisory"},
    {"name": "Natalia Cruz",    "job_title": "Brand Strategist",    "organization": "Radius Agency"},
]


# ---------------------------------------------------------------------------
# label + descriptor  (12 sets)
# Labels are single ALL-CAPS words. Descriptors are short noun phrases.
# Neither is a substring of the other.
# ---------------------------------------------------------------------------
SPLIT_PANEL: list[ContentSet] = [
    {"label": "RESOLUTION",  "descriptor": "4K Ultra HD at 60fps"},
    {"label": "CAPACITY",    "descriptor": "128 GB internal storage"},
    {"label": "RUNTIME",     "descriptor": "Up to 18 hours on a charge"},
    {"label": "FREQUENCY",   "descriptor": "2.4 GHz dual-band wireless"},
    {"label": "ACCURACY",    "descriptor": "Sub-millimeter touch precision"},
    {"label": "LATENCY",     "descriptor": "12ms measured response time"},
    {"label": "COVERAGE",    "descriptor": "Spans 340 square metres"},
    {"label": "EFFICIENCY",  "descriptor": "92 percent energy conversion"},
    {"label": "BANDWIDTH",   "descriptor": "10 Gbps symmetric throughput"},
    {"label": "PRESSURE",    "descriptor": "Rated to 300 metres depth"},
    {"label": "YIELD",       "descriptor": "3.2 tonnes per hectare annually"},
    {"label": "MAGNITUDE",   "descriptor": "6.4 recorded on Richter scale"},
]


# ---------------------------------------------------------------------------
# Public registry — maps layout name → content pool
# ---------------------------------------------------------------------------

CONTENT_POOLS: dict[str, list[ContentSet]] = {
    "title_subtitle": TITLE_SUBTITLE,
    "title_byline":   TITLE_BYLINE,
    "header_body":    HEADER_BODY,
    "name_card":      NAME_CARD,
    "split_panel":    SPLIT_PANEL,
}
