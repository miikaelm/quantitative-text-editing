"""
l2/content.py — Text content pools for Level 2 scenes, loaded from
data/content_pools.json (produced by generate_content.py).

The generation code samples from the pool (with replacement) to populate
scenes. Constraints enforced during generation:
  - All text strings within a single ContentSet are distinct.
  - No string is a substring of another string in the same set.
    (Required by OCR-based find_text_bbox, which matches by text content.)
  - Content is generic and varied.
"""

from __future__ import annotations

import json
from pathlib import Path

# A ContentSet maps role name → text string for one scene instance.
ContentSet = dict[str, str]

_JSON_PATH = Path(__file__).parents[3] / "data" / "content_pools.json"

if not _JSON_PATH.exists():
    raise FileNotFoundError(
        f"Content pools not found at {_JSON_PATH}. "
        "Run `python src/gen_pipeline/l2/generate_content.py` to generate them."
    )

with _JSON_PATH.open(encoding="utf-8") as _f:
    CONTENT_POOLS: dict[str, list[ContentSet]] = json.load(_f)
