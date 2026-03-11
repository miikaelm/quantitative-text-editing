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
# headline  (12 sets)
# Single ALL-CAPS or title-case word/phrase. Short enough to render large.
# ---------------------------------------------------------------------------
SOLO_HEADLINE: list[ContentSet] = [
    {"headline": "ILLUMINATE"},
    {"headline": "MOMENTUM"},
    {"headline": "ARISE"},
    {"headline": "THRESHOLD"},
    {"headline": "DIVERGE"},
    {"headline": "RADIANT"},
    {"headline": "CATALYST"},
    {"headline": "OVERTURE"},
    {"headline": "SIGNAL"},
    {"headline": "FRACTURE"},
    {"headline": "ASCENT"},
    {"headline": "GRAVITY"},
]


# ---------------------------------------------------------------------------
# quote + attribution  (12 sets)
# Quote is a short pithy sentence. Attribution is a name (no "by" prefix —
# the HTML builder adds the em dash). Neither is a substring of the other.
# ---------------------------------------------------------------------------
QUOTE_ATTRIBUTION: list[ContentSet] = [
    {"quote": "Simplicity is the ultimate form of sophistication.",
     "attribution": "Leonardo da Vinci"},
    {"quote": "The details are not the details — they make the design.",
     "attribution": "Charles Eames"},
    {"quote": "Good design is as little design as possible.",
     "attribution": "Dieter Rams"},
    {"quote": "Every great design begins with an even better story.",
     "attribution": "Lorinda Mamo"},
    {"quote": "Design is thinking made visual.",
     "attribution": "Saul Bass"},
    {"quote": "Whitespace is to be regarded as an active element.",
     "attribution": "Jan Tschichold"},
    {"quote": "Perfection is achieved when there is nothing left to remove.",
     "attribution": "Antoine de Saint-Exupéry"},
    {"quote": "Color is a power which directly influences the soul.",
     "attribution": "Wassily Kandinsky"},
    {"quote": "Typography is the voice of the written word.",
     "attribution": "Robert Bringhurst"},
    {"quote": "A designer knows success when nothing more can be taken away.",
     "attribution": "Paul Rand"},
    {"quote": "Form ever follows function.",
     "attribution": "Louis Sullivan"},
    {"quote": "Less is more.",
     "attribution": "Ludwig Mies van der Rohe"},
]


# ---------------------------------------------------------------------------
# label + badge  (12 sets)
# Labels are ALL-CAPS nouns. Badges are very short (≤ 10 chars), distinct
# from the label, and not substrings of each other.
# ---------------------------------------------------------------------------
CORNER_BADGE: list[ContentSet] = [
    {"label": "PREMIERE",    "badge": "NEW"},
    {"label": "EDITION",     "badge": "Vol. 4"},
    {"label": "COLLECTION",  "badge": "SOLD OUT"},
    {"label": "SYMPOSIUM",   "badge": "LIVE"},
    {"label": "JOURNAL",     "badge": "Issue 9"},
    {"label": "EXHIBITION",  "badge": "FREE"},
    {"label": "QUARTERLY",   "badge": "Q3"},
    {"label": "WORKSHOP",    "badge": "FULL"},
    {"label": "DISPATCH",    "badge": "DRAFT"},
    {"label": "COMPENDIUM",  "badge": "BETA"},
    {"label": "PORTFOLIO",   "badge": "2025"},
    {"label": "ARCHIVE",     "badge": "v2.1"},
]


# ---------------------------------------------------------------------------
# banner + caption  (12 sets)
# Banner is a short bold phrase (all-caps or title case). Caption is a
# single descriptive sentence. Neither is a substring of the other.
# ---------------------------------------------------------------------------
BANNER_CAPTION: list[ContentSet] = [
    {"banner": "OPENING NIGHT",
     "caption": "Doors open at 7 pm in the east atrium"},
    {"banner": "ANNUAL REPORT",
     "caption": "Fiscal year results across all operating divisions"},
    {"banner": "GRAND OPENING",
     "caption": "Join us for the ribbon cutting on Saturday morning"},
    {"banner": "WORLD TOUR",
     "caption": "Fourteen cities across six continents this season"},
    {"banner": "PRODUCT LAUNCH",
     "caption": "Available in stores and online from the first of March"},
    {"banner": "SUMMER SERIES",
     "caption": "Weekly outdoor screenings begin on the fifteenth"},
    {"banner": "FINAL CALL",
     "caption": "Registration closes at midnight on Friday"},
    {"banner": "DEEP DIVE",
     "caption": "A four-part exploration of modern infrastructure"},
    {"banner": "FIELD NOTES",
     "caption": "Observations collected over three months in the delta"},
    {"banner": "MASTER CLASS",
     "caption": "Six sessions with practitioners from across the field"},
    {"banner": "NIGHT MARKET",
     "caption": "Over forty vendors in the riverside district"},
    {"banner": "YEAR IN REVIEW",
     "caption": "Highlights and milestones from the past twelve months"},
]


# ---------------------------------------------------------------------------
# left_heading + right_heading  (12 sets)
# Both are short ALL-CAPS words or phrases. They must be distinct and
# neither a substring of the other.
# ---------------------------------------------------------------------------
TWO_COLUMN_HEADING: list[ContentSet] = [
    {"left_heading": "LIGHT",    "right_heading": "SHADOW"},
    {"left_heading": "PUSH",     "right_heading": "PULL"},
    {"left_heading": "ORDER",    "right_heading": "CHAOS"},
    {"left_heading": "FORM",     "right_heading": "FUNCTION"},
    {"left_heading": "EAST",     "right_heading": "WEST"},
    {"left_heading": "SIGNAL",   "right_heading": "NOISE"},
    {"left_heading": "MATTER",   "right_heading": "ENERGY"},
    {"left_heading": "ORIGIN",   "right_heading": "TERMINUS"},
    {"left_heading": "DIGITAL",  "right_heading": "ANALOG"},
    {"left_heading": "TENSION",  "right_heading": "RELEASE"},
    {"left_heading": "ABOVE",    "right_heading": "BELOW"},
    {"left_heading": "FORWARD",  "right_heading": "BACK"},
]


# ---------------------------------------------------------------------------
# Public registry — maps layout name → content pool
# ---------------------------------------------------------------------------

CONTENT_POOLS: dict[str, list[ContentSet]] = {
    "title_subtitle":    TITLE_SUBTITLE,
    "title_byline":      TITLE_BYLINE,
    "header_body":       HEADER_BODY,
    "name_card":         NAME_CARD,
    "split_panel":       SPLIT_PANEL,
    "solo_headline":     SOLO_HEADLINE,
    "quote_attribution": QUOTE_ATTRIBUTION,
    "corner_badge":      CORNER_BADGE,
    "banner_caption":    BANNER_CAPTION,
    "two_column_heading": TWO_COLUMN_HEADING,
}
