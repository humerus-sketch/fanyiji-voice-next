"""Opt-in source-only NP atoms; decoding never needs a composite NP grammar.

This finite splitter is not a Japanese parser. Unknown mixed-script syntax is
declined, and lexical accuracy still belongs to the model and separate review.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

from .gemma_fragment_structure import FragmentSlot, analyze_source_v2
from .gemma_source_structure import Analysis, SourceSpan, StructurePlan, StructureRenderError


@dataclass(frozen=True)
class AtomSlot(FragmentSlot):
    pass


@dataclass(frozen=True)
class AtomPlan(StructurePlan):
    delegate: StructurePlan
    bindings: tuple[dict, ...]

    def render(self, values: Mapping[str, str]) -> str:
        if not isinstance(values, Mapping) or set(values) != {s.key for s in self.slots}:
            raise StructureRenderError("内容槽缺失或包含未声明的槽；未交付原子翻译。")
        if any(not isinstance(v, str) or not v.strip() for v in values.values()):
            raise StructureRenderError("内容槽必须是非空文本；未交付原子翻译。")
        # Every raw word, punctuation mark and internal newline is preserved.
        parent_values = {b["parent_slot_key"]: b["joiner"].join(values[k].strip() for k in b["atom_keys"])
                         for b in self.bindings}
        return self.delegate.render(parent_values)

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["structure_version"] = 3
        for row, slot in zip(data["slots"], self.slots):
            row["output_shape"] = slot.output_shape
        data["delegate_kind"] = self.delegate.kind
        data["delegate_plan"] = self.delegate.to_dict()
        data["bindings"] = [{**b, "atom_keys": list(b["atom_keys"]),
                             "source_spans": [list(span) for span in b["source_spans"]]}
                            for b in self.bindings]
        return data


# Japanese form/lemma entries, not bilingual lexical answers.
_ATTRIBUTIVE = {
    "目立った": "目立つ", "整った": "整う",
    "優れた": "優れる", "新しい": "新しい", "古い": "古い", "正しい": "正しい",
    "難しい": "難しい", "易しい": "易しい", "厳しい": "厳しい", "詳しい": "詳しい",
    "大きい": "大きい", "小さい": "小さい", "明るい": "明るい", "暗い": "暗い",
    "美しい": "美しい", "強い": "強い", "弱い": "弱い", "長い": "長い", "短い": "短い",
}
_VN_PREFIXES = frozenset({"存在", "利用", "使用", "研究", "開発", "保存", "確認", "管理",
                          "処理", "設定", "実行", "検証", "作成", "更新", "修正"})
_COMPOUND_HEADS = frozenset({"価値", "方法", "目的", "予定", "状況", "手順", "条件", "結果",
                            "時間", "場所", "記録", "計画", "資料", "情報", "履歴", "対象",
                            "範囲", "期間", "状態"})
_PRONOUNS = frozenset({"私", "僕", "俺", "君", "あなた", "彼", "彼女", "私たち", "僕たち", "あなたたち", "自分"})
# Closed source nouns whose kana/okurigana spelling is outside _BARE_NOUN.
# This does not infer nouns from arbitrary hiragana or reverse verb morphology.
_KANA_NOUNS = frozenset({"おすすめ", "お知らせ", "お願い", "お祝い", "お土産", "たより",
                         "くらし", "こころ", "いのち", "かたち", "しるし", "きっかけ"})
_BARE_NOUN = re.compile(r"(?:[一-龥々]{1,8}|[ァ-ヺー]{1,16}|お[一-龥々]{1,4}|ご[一-龥々]{1,4})")


def _atom_shape(text: str, *, modifier=False) -> str | None:
    if text in _PRONOUNS:
        return "pronoun"
    if text in _KANA_NOUNS or _BARE_NOUN.fullmatch(text):
        return "modifier" if modifier else "noun"
    return None


def _np_parts(text: str):
    """Return (local start, end, lemma, shape, rule) and source grammar spans."""
    if text.count("の") == 1:
        left, right = text.split("の")
        lshape, rshape = _atom_shape(left, modifier=True), _atom_shape(right)
        if lshape and rshape:
            middle = len(left)
            return [(0, middle, left, lshape, "identity"),
                    (middle + 1, len(text), right, rshape, "identity")], [(middle, middle + 1)]
        return None
    if "の" in text:
        return None
    for form, lemma in sorted(_ATTRIBUTIVE.items(), key=lambda x: len(x[0]), reverse=True):
        if text.startswith(form):
            head = text[len(form):]
            shape = _atom_shape(head)
            if shape == "noun":
                return [(0, len(form), lemma, "modifier", "closed_attributive_to_dictionary"),
                        (len(form), len(text), head, shape, "identity")], []
            return None
    for prefix in sorted(_VN_PREFIXES, key=len, reverse=True):
        if text.startswith(prefix) and text[len(prefix):] in _COMPOUND_HEADS:
            return [(0, len(prefix), prefix, "modifier", "source_VN_compound_modifier"),
                    (len(prefix), len(text), text[len(prefix):], "noun", "identity")], []
    shape = _atom_shape(text)
    if shape:
        return [(0, len(text), text, shape, "identity")], []
    return None


def analyze_source_v3(text: str) -> Analysis:
    original = analyze_source_v2(text)
    if original.plan is None:
        return original
    parent = original.plan
    slots, bindings, replacements = [], [], {}
    for old in parent.slots:
        if old.output_shape != "noun_phrase":
            slots.append(AtomSlot(old.key, old.source_text, old.source_spans, old.transformation, old.output_shape))
            bindings.append({"parent_slot_key": old.key, "atom_keys": (old.key,), "joiner": "",
                             "source_spans": old.source_spans, "split_rule": "unchanged_non_NP"})
            continue
        if len(old.source_spans) != 1 or old.transformation != "identity":
            return Analysis(None, "unsupported_np_atom_alignment:" + old.key)
        start, end = old.source_spans[0]
        if parent.source_text[start:end] != old.source_text:
            return Analysis(None, "unsupported_np_atom_alignment:" + old.key)
        split = _np_parts(old.source_text)
        if split is None:
            return Analysis(None, "unsupported_np_atom_structure:" + old.key)
        parts, grammar = split
        keys, new_spans = [], []
        for index, (lo, hi, lemma, shape, rule) in enumerate(parts):
            key = old.key + ("__head" if index == len(parts) - 1 else "__modifier")
            keys.append(key)
            slots.append(AtomSlot(key, lemma, ((start + lo, start + hi),), rule, shape))
            new_spans.append(SourceSpan(start + lo, start + hi, "content", key))
        new_spans.extend(SourceSpan(start + lo, start + hi, "NP_genitive_operator") for lo, hi in grammar)
        replacements[old.key] = sorted(new_spans, key=lambda s: s.start)
        bindings.append({"parent_slot_key": old.key, "atom_keys": tuple(keys),
                         "joiner": "的" if len(keys) == 2 else "", "source_spans": old.source_spans,
                         "split_rule": "source_genitive" if grammar else "source_modifier_head" if len(keys) == 2 else "single_source_noun"})
    spans = tuple(child for span in parent.spans for child in replacements.get(span.slot_key, (span,)))
    mapping = {b["parent_slot_key"]: b["atom_keys"] for b in bindings}
    operators = tuple({**op, "parent_scope_slots": list(op.get("scope_slots", [])),
                       "scope_slots": [key for old in op.get("scope_slots", []) for key in mapping.get(old, (old,))]}
                      for op in parent.operators)
    return Analysis(AtomPlan(parent.kind, parent.source_text, tuple(slots), spans, operators, parent, tuple(bindings)), None)
