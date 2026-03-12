"""
build_pairs.py — EditPair dataclass shared across the generation pipeline.
"""

from dataclasses import dataclass, field


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
