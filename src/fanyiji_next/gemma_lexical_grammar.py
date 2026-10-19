"""Opt-in lexical output shapes from an existing local POS dictionary.

This is a decoding constraint, not a semantic verifier or a bilingual lexicon.
Vocabulary selection is global frequency order and never inspects source cases.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def _prefix_expression(words):
    """Factor a finite word union without changing its accepted strings."""
    tree = {}
    for word in words:
        node = tree
        for character in word:
            node = node.setdefault(character, {})
        node[None] = True

    def emit(node):
        branches = []
        for character in sorted(key for key in node if key is not None):
            child, prefix = node[character], character
            while None not in child and len(child) == 1:
                following = next(iter(child))
                prefix += following
                child = child[following]
            suffix = emit(child)
            branches.append(json.dumps(prefix, ensure_ascii=False) + (" " + suffix if suffix else ""))
        if not branches:
            return ""
        expression = branches[0] if len(branches) == 1 else "(" + " | ".join(branches) + ")"
        return "(" + expression + ")?" if None in node else expression

    return emit(tree)


class LexicalGrammar:
    LIMITS = {"noun": 5000, "verb": 2400, "adjective": 1200, "pronoun": 256, "temporal": 512}
    SHAPES = frozenset({"noun_phrase", "noun", "modifier", "verb", "adjective", "pronoun", "temporal"})

    def __init__(self, dictionary_path):
        path = Path(dictionary_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("内容词典必须是已有的本地文件。")
        raw = path.read_bytes()
        pools = {name: {} for name in self.LIMITS}
        for line in raw.decode("utf-8").splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            word, count, pos = parts
            if not count.isdecimal() or not re.fullmatch(r"[\u3400-\u9fff]{1,8}", word):
                continue
            count = int(count)
            if count <= 0:
                continue
            categories = []
            if pos == "n" or pos == "vn":
                categories.append("noun")
            if pos == "v":
                categories.append("verb")
            if pos in {"a", "ad", "an"}:
                categories.append("adjective")
            if pos.startswith("r"):
                categories.append("pronoun")
            if pos == "t":
                categories.append("temporal")
            for category in categories:
                pools[category][word] = max(count, pools[category].get(word, 0))
        self.words = {name: tuple(word for word, _ in sorted(pool.items(), key=lambda item: (-item[1], item[0]))[:self.LIMITS[name]])
                      for name, pool in pools.items()}
        if any(not words for words in self.words.values()):
            raise ValueError("已有词典缺少候选所需的词性，不能退为无约束生成。")
        rules = {name: name + " ::= " + _prefix_expression(words)
                 for name, words in self.words.items()}
        self._grammars = {
            "noun": "root ::= noun\n" + rules["noun"] + "\n",
            "modifier": "root ::= modifier\nmodifier ::= " + _prefix_expression(
                sorted(set(self.words["noun"]) | set(self.words["verb"]) | set(self.words["adjective"]))) + "\n",
            "verb": "root ::= verb\n" + rules["verb"] + "\n",
            "adjective": "root ::= adjective\n" + rules["adjective"] + "\n",
            "pronoun": "root ::= pronoun\n" + rules["pronoun"] + "\n",
            "temporal": "root ::= temporal\n" + rules["temporal"] + "\n",
            # Head-final nominal shape: a finite lexical head, optional adjective
            # or a verb+de modifier. This cannot certify contextual POS/meaning.
            "noun_phrase": ("root ::= noun-phrase\n"
                            'noun-phrase ::= (adjective "的"? | verb "的")? noun ("的"? noun)?\n'
                            + rules["noun"] + "\n" + rules["verb"] + "\n" + rules["adjective"] + "\n"),
        }
        self.metadata = {"dictionary_path": str(path), "dictionary_sha256": hashlib.sha256(raw).hexdigest(),
                         "dictionary_size_bytes": len(raw), "selection": "Fixed global frequency order, lexical tie-break, no case/source access",
                         "limits": dict(self.LIMITS), "selected_counts": {name: len(words) for name, words in self.words.items()},
                         "grammar_encoding": "Equivalent shared-prefix finite-word trie; unchanged vocabulary and NP productions",
                         "vocabulary_sha256": {name: hashlib.sha256(json.dumps(words, ensure_ascii=False).encode("utf-8")).hexdigest()
                                               for name, words in self.words.items()},
                         "limitation": "Static lexical POS and finite shapes do not prove contextual syntax, word sense, source coverage or semantic correctness.",
                         "grammars": {shape: self.describe(grammar) for shape, grammar in self._grammars.items()}}

    @staticmethod
    def describe(grammar):
        raw = grammar.encode("utf-8")
        return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), "characters": len(grammar)}

    def for_shape(self, shape):
        if shape == "free":
            return None
        if shape not in self.SHAPES:
            raise ValueError(f"未知内容槽生成形状：{shape!r}")
        return self._grammars[shape]
