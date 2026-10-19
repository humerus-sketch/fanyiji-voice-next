"""Opt-in atom shape without a Chinese vocabulary whitelist.

Only the source parser chooses atom boundaries. A bounded Han span cannot
certify POS, sense or fidelity; the complete result still needs semantic review.
"""
from __future__ import annotations

import hashlib


class AtomGrammar:
    SHAPES = frozenset({"noun", "modifier", "verb", "adjective", "pronoun", "temporal"})
    MAX_CHARACTERS = 12

    def __init__(self):
        # A character limit bounds the decoding language, not Chinese words.
        # All typed atoms use the same rule, independent of source/case/answers.
        self._grammar = 'root ::= [\\u3400-\\u9fff]{1,' + str(self.MAX_CHARACTERS) + '}\n'
        self.metadata = {
            "kind": "bounded_han_atom_shape",
            "dictionary_used": False,
            "vocabulary_whitelist": False,
            "source_or_case_access": False,
            "max_characters": self.MAX_CHARACTERS,
            "grammars": {shape: self.describe(self._grammar) for shape in sorted(self.SHAPES)},
            "limitation": "Character shape does not prove a single word, POS, completeness, word sense or source fidelity; longer/foreign-script content may be unrepresentable.",
        }

    @staticmethod
    def describe(grammar):
        raw = grammar.encode("utf-8")
        return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), "characters": len(grammar)}

    def for_shape(self, shape):
        if shape == "free":
            return None
        if shape not in self.SHAPES:
            raise ValueError(f"未知内容槽生成形状：{shape!r}")
        return self._grammar
