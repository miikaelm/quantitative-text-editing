"""
build_pairs.py — Generic glue between edit specs and rendered HTML pairs.

Takes a list of EditSpec objects and a type-specific HTML builder function,
produces EditPair objects ready for rendering.

To add a new edit type:
    1. Create gen_pipeline/specs/<type>.py
    2. Create gen_pipeline/templates/<type>.py with a build_<type>_html() function
    3. Call build_pairs(specs, build_<type>_html) in generate.py
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

from gen_pipeline.specs.base import EditSpec


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


def build_pairs(
    specs: Sequence[EditSpec],
    html_builder: Callable[..., tuple[str, str]],
) -> list[EditPair]:
    """
    Convert edit specs into EditPair objects using the given HTML builder.

    Args:
        specs:        List of EditSpec (or subclass) objects.
        html_builder: Function that takes a spec and returns (source_html, target_html).

    Returns:
        List of EditPair objects with HTML populated from the builder.
        Metadata is carried through unchanged from each spec.
    """
    pairs = []
    for spec in specs:
        source_html, target_html = html_builder(spec)
        pairs.append(EditPair(
            pair_id=spec.pair_id,
            edit_type=spec.edit_type,
            source_html=source_html,
            target_html=target_html,
            instruction=spec.instruction,
            metadata=spec.metadata,
        ))
    return pairs
