"""
specs/base.py — Base EditSpec dataclass shared by all edit types.

Fields:
    pair_id:      Unique identifier, e.g. "color_001".
    edit_type:    One of "color", "reposition", "scaling", "content".
    instruction:  Natural-language edit instruction for the model.
    text_content: The text string being edited (used for OCR bbox lookup).
    metadata:     Edit-type-specific fields (old_value, new_value, bbox, …).
"""

from dataclasses import dataclass, field


@dataclass
class EditSpec:
    pair_id: str
    edit_type: str
    instruction: str
    text_content: str
    metadata: dict = field(default_factory=dict)
