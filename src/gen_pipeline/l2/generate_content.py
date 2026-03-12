#!/usr/bin/env python3
"""
generate_content_pools.py — Generate large-scale content pools for L2 scenes
using the OpenAI API, with automatic constraint validation.

Usage:
    export OPENAI_API_KEY="sk-..."
    python generate_content_pools.py [--target 120] [--output content_pools.json]

Each layout type gets its own generation prompt with few-shot examples and
explicit constraints. Generated sets are validated for:
  1. All strings within a set are distinct.
  2. No string is a substring of another in the same set.
  3. Strings meet length/format requirements for their role.
  4. No duplicate sets across the pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from gen_pipeline.l2.api_tracker import DummyOpenAI, OpenAIClient, TrackedOpenAI

# ---------------------------------------------------------------------------
# Layout type schemas: roles, constraints, and few-shot examples
# ---------------------------------------------------------------------------

LAYOUT_SCHEMAS: dict[str, dict] = {
    "title_subtitle": {
        "roles": {
            "title": "Short ALL-CAPS title (1-2 words, max 20 chars)",
            "subtitle": "Descriptive subtitle sentence (5-8 words, max 50 chars)",
        },
        "examples": [
            {"title": "NORTHERN LIGHTS", "subtitle": "A guide to arctic photography"},
            {"title": "VELOCITY", "subtitle": "Built for speed and precision"},
            {"title": "FRACTAL", "subtitle": "Infinite patterns in nature"},
        ],
    },
    "title_byline": {
        "roles": {
            "title": "Short ALL-CAPS title (1 word, max 12 chars)",
            "byline": "Very short byline like issue/volume/edition (max 18 chars)",
        },
        "examples": [
            {"title": "HORIZON", "byline": "Issue 12"},
            {"title": "SPECTRUM", "byline": "Vol. 3"},
            {"title": "NEXUS", "byline": "Beta Version"},
        ],
    },
    "header_body": {
        "roles": {
            "header": "Title-case header (2-3 words, max 20 chars)",
            "body": "Single sentence body text (7-10 words, max 60 chars). No period at the end.",
        },
        "examples": [
            {"header": "System Status", "body": "All services operating within normal parameters"},
            {"header": "Field Report", "body": "Data collection concluded at seventeen sites"},
            {"header": "Analysis", "body": "Sample variance within acceptable tolerance range"},
        ],
    },
    "name_card": {
        "roles": {
            "name": "Full name (first and last, fictional, max 20 chars)",
            "job_title": "Professional job title (2-3 words, max 22 chars)",
            "organization": "Company/org name (2-3 words, max 24 chars)",
        },
        "examples": [
            {"name": "Elena Vasquez", "job_title": "Senior Architect", "organization": "Nordic Design Studio"},
            {"name": "Priya Nair", "job_title": "Product Manager", "organization": "Clearpath Tech"},
            {"name": "Amara Osei", "job_title": "Business Analyst", "organization": "Summit Advisory"},
        ],
    },
    "split_panel": {
        "roles": {
            "label": "Single ALL-CAPS word (max 12 chars)",
            "descriptor": "Short technical spec phrase (5-7 words, max 40 chars)",
        },
        "examples": [
            {"label": "RESOLUTION", "descriptor": "4K Ultra HD at 60fps"},
            {"label": "LATENCY", "descriptor": "12ms measured response time"},
            {"label": "BANDWIDTH", "descriptor": "10 Gbps symmetric throughput"},
        ],
    },
    "solo_headline": {
        "roles": {
            "headline": "Single ALL-CAPS word (max 12 chars). Abstract/evocative noun or verb.",
        },
        "examples": [
            {"headline": "ILLUMINATE"},
            {"headline": "THRESHOLD"},
            {"headline": "CATALYST"},
        ],
    },
    "quote_attribution": {
        "roles": {
            "quote": "Short pithy sentence (8-12 words, max 65 chars). Should sound like a real aphorism.",
            "attribution": "Full name of a real or plausible person (max 30 chars)",
        },
        "examples": [
            {"quote": "Simplicity is the ultimate form of sophistication.", "attribution": "Leonardo da Vinci"},
            {"quote": "Good design is as little design as possible.", "attribution": "Dieter Rams"},
            {"quote": "Color is a power which directly influences the soul.", "attribution": "Wassily Kandinsky"},
        ],
    },
    "corner_badge": {
        "roles": {
            "label": "Single ALL-CAPS noun (max 12 chars)",
            "badge": "Very short badge text (max 10 chars, e.g. NEW, Vol. 4, Q3, BETA)",
        },
        "examples": [
            {"label": "PREMIERE", "badge": "NEW"},
            {"label": "COLLECTION", "badge": "SOLD OUT"},
            {"label": "PORTFOLIO", "badge": "2025"},
        ],
    },
    "banner_caption": {
        "roles": {
            "banner": "Short ALL-CAPS bold phrase (2-3 words, max 18 chars)",
            "caption": "Single descriptive sentence (8-12 words, max 60 chars)",
        },
        "examples": [
            {"banner": "OPENING NIGHT", "caption": "Doors open at 7 pm in the east atrium"},
            {"banner": "WORLD TOUR", "caption": "Fourteen cities across six continents this season"},
            {"banner": "FINAL CALL", "caption": "Registration closes at midnight on Friday"},
        ],
    },
    "two_column_heading": {
        "roles": {
            "left_heading": "Short ALL-CAPS word (max 10 chars)",
            "right_heading": "Short ALL-CAPS word (max 10 chars). Should contrast/complement left_heading.",
        },
        "examples": [
            {"left_heading": "LIGHT", "right_heading": "SHADOW"},
            {"left_heading": "SIGNAL", "right_heading": "NOISE"},
            {"left_heading": "DIGITAL", "right_heading": "ANALOG"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_set(content_set: dict[str, str], schema: dict) -> list[str]:
    """Return list of constraint violations (empty = valid)."""
    errors = []
    roles = schema["roles"]

    # Check all expected roles are present
    for role in roles:
        if role not in content_set:
            errors.append(f"Missing role: {role}")
            return errors  # Can't validate further

    # Check no extra roles
    for key in content_set:
        if key not in roles:
            errors.append(f"Unexpected role: {key}")

    values = list(content_set.values())

    # All strings must be non-empty
    for role, val in content_set.items():
        if not isinstance(val, str) or not val.strip():
            errors.append(f"Empty or non-string value for {role}")
            return errors

    # All values must be distinct
    if len(set(values)) != len(values):
        errors.append("Duplicate values within set")

    # No value is a substring of another
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            if i != j and a.lower() in b.lower():
                errors.append(f"Substring overlap: '{a}' found in '{b}'")

    # Check length constraints (extracted from role descriptions)
    for role, val in content_set.items():
        desc = roles[role]
        # Extract max char limit if specified
        if "max" in desc:
            import re
            match = re.search(r"max\s+(\d+)\s+chars?", desc)
            if match:
                max_len = int(match.group(1))
                if len(val) > max_len:
                    errors.append(f"{role} too long: {len(val)} > {max_len} chars")

        # Check ALL-CAPS requirement
        if "ALL-CAPS" in desc and val != val.upper():
            errors.append(f"{role} should be ALL-CAPS: '{val}'")

    return errors


def deduplicate_pool(pool: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove exact duplicate content sets."""
    seen = set()
    unique = []
    for cs in pool:
        key = tuple(sorted(cs.items()))
        if key not in seen:
            seen.add(key)
            unique.append(cs)
    return unique


# ---------------------------------------------------------------------------
# Generation via OpenAI API
# ---------------------------------------------------------------------------

# Domain categories for stratified generation.
# Each batch is assigned a domain to keep the pool diverse across topics.
_DOMAINS = [
    "science and technology",
    "creative arts and culture",
    "commerce and finance",
    "nature and environment",
    "sports and fitness",
]


def build_prompt(layout_name: str, schema: dict, batch_size: int = 25, domain: str | None = None) -> str:
    """Build the generation prompt for a layout type, optionally scoped to a domain."""
    roles_desc = "\n".join(
        f'  - "{role}": {desc}' for role, desc in schema["roles"].items()
    )
    examples_json = json.dumps(schema["examples"], indent=2)

    domain_instruction = (
        f"\nDOMAIN FOCUS: All {batch_size} sets in this batch should be themed around "
        f'"{domain}". Stay within this domain while still following all constraints.\n'
        if domain else ""
    )

    return f"""Generate exactly {batch_size} content sets for a layout type called "{layout_name}".

Each content set is a JSON object with these roles:
{roles_desc}
{domain_instruction}
HARD CONSTRAINTS (sets that violate these will be rejected):
1. All string values within a single set must be completely distinct.
2. No string value may be a substring of another string value in the same set (case-insensitive). For example, if one value is "FORM" and another is "FUNCTION", that is NOT allowed because... actually wait, "FORM" is not contained in "FUNCTION". But "ART" would be contained in "ARTIST", so that pair is forbidden.
3. Strings must respect the character limits specified above.
4. ALL-CAPS fields must be ALL-CAPS.
5. Content should be varied and generic — avoid repeating themes across sets.
6. Sets should be diverse: vary topics, domains, and styles.

Here are {len(schema["examples"])} examples of valid sets:
{examples_json}

Generate {batch_size} NEW sets (do not repeat the examples). Respond with ONLY a JSON array of {batch_size} objects. No markdown, no explanation, no backticks."""


def generate_batch(
    client: OpenAIClient,
    layout_name: str,
    schema: dict,
    batch_size: int = 25,
    model: str = "gpt-4.1-mini",
    max_retries: int = 3,
    domain: str | None = None,
) -> list[dict[str, str]]:
    """Generate one batch of content sets via the OpenAI API."""
    prompt = build_prompt(layout_name, schema, batch_size, domain=domain)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data generation assistant. You output only valid JSON arrays with no extra text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=4096,
            )

            text = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[: text.rfind("```")]
            text = text.strip()

            batch = json.loads(text)

            if not isinstance(batch, list):
                print(f"  [!] Response was not a list, retrying... (attempt {attempt+1})")
                continue

            return batch

        except json.JSONDecodeError as e:
            print(f"  [!] JSON parse error: {e} (attempt {attempt+1})")
            time.sleep(1)
        except Exception as e:
            print(f"  [!] API error: {e} (attempt {attempt+1})")
            time.sleep(2)

    return []


def generate_pool(
    client: OpenAIClient,
    layout_name: str,
    schema: dict,
    target: int = 120,
    batch_size: int = 25,
    model: str = "gpt-4.1-mini",
    existing: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Generate a full pool for one layout type, with validation."""
    print(f"\n{'='*60}")
    print(f"Generating: {layout_name} (target: {target})")
    print(f"{'='*60}")

    pool: list[dict[str, str]] = []
    # Start from existing sets if resuming, otherwise seed with examples
    if existing:
        pool.extend(existing)
        print(f"  Resuming from {len(pool)} existing sets")
    else:
        pool.extend(schema["examples"])
    pool = deduplicate_pool(pool)

    total_generated = 0
    total_rejected = 0
    batches = 0

    while len(pool) < target:
        remaining = target - len(pool)
        current_batch_size = min(batch_size, remaining + 10)  # Generate a few extra for rejections

        batches += 1
        # Cycle through domains to ensure even domain coverage across batches.
        domain = _DOMAINS[(batches - 1) % len(_DOMAINS)]
        print(f"  Batch {batches} [{domain}]: requesting {current_batch_size} sets...")

        batch = generate_batch(client, layout_name, schema, current_batch_size, model, domain=domain)
        total_generated += len(batch)

        # Validate each set
        valid_in_batch = 0
        for cs in batch:
            errors = validate_set(cs, schema)
            if errors:
                total_rejected += 1
                if total_rejected <= 5:  # Only show first few rejections per type
                    print(f"    Rejected: {cs}")
                    for e in errors:
                        print(f"      - {e}")
            else:
                pool.append(cs)
                valid_in_batch += 1

        # Deduplicate after each batch
        pool = deduplicate_pool(pool)
        print(f"    Valid: {valid_in_batch} | Pool size: {len(pool)} | Rejected total: {total_rejected}")

        # Rate limit safety
        time.sleep(0.5)

        # Safety valve: don't loop forever
        if batches > 15:
            print(f"  [!] Hit max batches ({batches}), stopping with {len(pool)} sets")
            break

    print(f"  DONE: {len(pool)} sets (generated {total_generated}, rejected {total_rejected})")
    return pool


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate content pools via OpenAI API")
    parser.add_argument("--target", type=int, default=120,
                        help="Target number of content sets per layout type (default: 120)")
    parser.add_argument("--output", type=str, default="content_pools.json",
                        help="Output JSON file (default: content_pools.json)")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model to use (default: gpt-4.1-mini)")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Sets to request per API call (default: 25)")
    parser.add_argument("--layouts", nargs="*", default=None,
                        help="Specific layout types to generate (default: all)")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy client (no API calls, for testing)")
    args = parser.parse_args()

    if args.dummy:
        client: OpenAIClient = DummyOpenAI()
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY environment variable")
            sys.exit(1)
        client = TrackedOpenAI(api_key=api_key, note="generate_content_pools")

    layouts_to_generate = args.layouts or list(LAYOUT_SCHEMAS.keys())
    all_pools: dict[str, list[dict[str, str]]] = {}

    # If output file exists, load it so we can resume / extend
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded existing pools from {args.output}")
        all_pools.update(existing)

    for layout_name in layouts_to_generate:
        if layout_name not in LAYOUT_SCHEMAS:
            print(f"WARNING: Unknown layout '{layout_name}', skipping")
            continue

        schema = LAYOUT_SCHEMAS[layout_name]

        # If we already have enough, skip
        existing_count = len(all_pools.get(layout_name, []))
        if existing_count >= args.target:
            print(f"\nSkipping {layout_name}: already have {existing_count} >= {args.target}")
            continue

        pool = generate_pool(
            client, layout_name, schema,
            target=args.target,
            batch_size=args.batch_size,
            model=args.model,
            existing=all_pools.get(layout_name),
        )
        all_pools[layout_name] = pool

        # Save after each layout type (incremental progress)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_pools, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {args.output}")

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = 0
    for name in sorted(all_pools.keys()):
        count = len(all_pools[name])
        total += count
        print(f"  {name:25s}: {count:4d} sets")
    print(f"  {'TOTAL':25s}: {total:4d} sets")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()