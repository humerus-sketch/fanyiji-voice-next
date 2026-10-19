"""Opt-in finite open-fragment plans; v1 remains unchanged.

All lexical output comes from content slots. Operators render only source
grammar, and an open tail never invents its missing predicate or modality.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

from .gemma_source_structure import (Analysis, ContentSlot, SourceSpan, StructurePlan,
                                     StructureRenderError, analyze_source)


@dataclass(frozen=True)
class FragmentSlot(ContentSlot):
    output_shape: str = "free"


@dataclass(frozen=True)
class FragmentPlan(StructurePlan):
    delegate: StructurePlan | None = None

    def render(self, values: Mapping[str, str]) -> str:
        if self.delegate is not None:
            return self.delegate.render(values)
        if not isinstance(values, Mapping) or set(values) != {s.key for s in self.slots}:
            raise StructureRenderError("内容槽缺失或包含未声明的槽；未交付片段翻译。")
        if any(not isinstance(v, str) or not v.strip() for v in values.values()):
            raise StructureRenderError("内容槽必须是非空文本；未交付片段翻译。")
        value = {k: raw.strip() for k, raw in values.items()}
        degree = next((op["form"] for op in self.operators if op["name"] == "degree"), None)
        if self.kind == "open_i_adjective_adverbial":
            time = value["time"] + "，" if "time" in value else ""
            return time + _DEGREE_RENDER.get(degree, "") + value["adjective"] + "地……"
        if self.kind == "open_exemplified_action_nominalization":
            return value["subject"] + _DEGREE_RENDER.get(degree, "") + value["action"] + value["object"] + "之类的事……"
        if self.kind == "open_existential_condition_te_mo":
            return "如果有" + value["condition_noun"] + "，" + value["action"] + "也……"
        if self.kind == "open_topic_whenever":
            return value["topic"] + "，无论什么时候……"
        raise StructureRenderError("未知片段结构；未交付片段翻译。")

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["structure_version"] = 2
        for record, slot in zip(data["slots"], self.slots):
            record["output_shape"] = slot.output_shape
        if self.delegate is not None:
            data["delegate_kind"] = self.delegate.kind
        return data


# Japanese morphology entries only, never Chinese lexical answers.
_I_FORMS = {
    "嬉しく": "嬉しい", "うれしく": "うれしい", "楽しく": "楽しい", "悲しく": "悲しい",
    "寂しく": "寂しい", "優しく": "優しい", "美しく": "美しい", "大きく": "大きい",
    "小さく": "小さい", "強く": "強い", "弱く": "弱い", "明るく": "明るい",
    "暗く": "暗い", "早く": "早い", "速く": "速い", "遅く": "遅い", "長く": "長い", "短く": "短い",
    "詳しく": "詳しい", "新しく": "新しい", "古く": "古い", "正しく": "正しい",
    "難しく": "難しい", "易しく": "易しい", "厳しく": "厳しい",
}
_TARI_FORMS = {
    "書いたり": "書く", "読んだり": "読む", "運んだり": "運ぶ", "待ったり": "待つ",
    "話したり": "話す", "買ったり": "買う", "作ったり": "作る", "飲んだり": "飲む",
    "遊んだり": "遊ぶ", "歌ったり": "歌う", "笑ったり": "笑う", "見たり": "見る",
    "食べたり": "食べる", "覚えたり": "覚える", "忘れたり": "忘れる", "嫌ったり": "嫌う",
}
_TE_MO_FORMS = {
    "聞いても": "聞く", "話しても": "話す", "読んでも": "読む", "書いても": "書く",
    "運んでも": "運ぶ", "買っても": "買う", "待っても": "待つ", "見ても": "見る",
    "食べても": "食べる", "覚えても": "覚える", "忘れても": "忘れる", "言っても": "言う",
    "歌っても": "歌う", "笑っても": "笑う",
}
_TIMES = ("今", "今日", "昨日", "明日", "最近", "今朝", "今夜", "今年")
_PRONOUNS = frozenset({"私", "僕", "俺", "君", "あなた", "彼", "彼女", "私たち", "僕たち", "あなたたち", "自分"})
# These are grammatical degree operators, not noun/verb/adjective translations.
# Zenzen without an expressed negative does not establish inability or NOT.
_DEGREE_RENDER = {"少し": "稍微", "ちょっと": "稍微", "もっと": "更", "全然": "完全", "時々": "有时"}
_JP = r"[一-龥々ぁ-ゖァ-ヺーA-Za-z0-9０-９]+"
_NOUN_SYNTAX = ("から", "けど", "なら", "たら", "ので", "が", "は", "を", "に", "へ", "と", "で", "も", "ない", "こと", "もの")


def _alternatives(forms):
    return "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))


_ADVERBIAL = re.compile(rf"(?P<time>{_alternatives(_TIMES)})?(?P<degree>少し|ちょっと|もっと)?(?P<adjective>{_alternatives(_I_FORMS)})")
_NOMINALIZATION = re.compile(rf"(?P<subject>{_JP})(?P<subject_case>が)(?P<object>{_JP})(?P<object_case>を)"
                             rf"(?P<degree>全然|少し|ちょっと|時々)?(?P<action>{_alternatives(_TARI_FORMS)})(?P<tail>するのが)")
_CONDITION = re.compile(rf"(?P<noun>{_JP})(?P<condition>があれば)(?P<action>{_alternatives(_TE_MO_FORMS)})")
_WHENEVER = re.compile(rf"(?P<topic>{_JP})(?P<topic_case>は)(?P<whenever>いつだって)")


def _simple_noun(text):
    # The opt-in lexical shapes cannot yet preserve ASCII identifiers/digits.
    if re.search(r"[A-Za-z0-9０-９]", text):
        return False
    if text in _PRONOUNS:
        return True
    return (not any(marker in text for marker in _NOUN_SYNTAX)
            and re.search(r"[一-龥々ァ-ヺA-Za-z0-9０-９]$", text) is not None)


def _noun_shape(text):
    return "pronoun" if text in _PRONOUNS else "noun_phrase"


def _wrap_v1(plan):
    slots = tuple(FragmentSlot(s.key, s.source_text, s.source_spans, s.transformation,
                               _noun_shape(s.source_text) if plan.kind == "unfinished_state_topic" and s.key == "topic" else "free")
                  for s in plan.slots)
    return FragmentPlan(plan.kind, plan.source_text, slots, plan.spans, plan.operators, plan)


def _plan(kind, source, slots, labels, operators):
    spans, cursor = [], 0
    for start, end, role, key in sorted(labels):
        if start < cursor or end <= start:
            raise ValueError("Overlapping or empty fragment source ledger")
        if cursor < start:
            spans.append(SourceSpan(cursor, start, "formatting"))
        spans.append(SourceSpan(start, end, role, key))
        cursor = end
    if cursor < len(source):
        spans.append(SourceSpan(cursor, len(source), "formatting"))
    return FragmentPlan(kind, source, tuple(slots), tuple(spans), tuple(operators))


def analyze_source_v2(text: str) -> Analysis:
    original = analyze_source(text)
    if original.plan is not None:
        if original.plan.kind == "unfinished_state_topic" and any(
                s.key == "topic" and re.search(r"[A-Za-z0-9０-９]", s.source_text) for s in original.plan.slots):
            return Analysis(None, "identifier_outside_v2_noun_output_shape")
        return Analysis(_wrap_v1(original.plan), None)
    if original.reason in {"source_is_not_text", "empty_source", "quoted_or_bracketed_source",
                           "question_exclamation_or_multiple_lines"}:
        return original
    core = text.strip()
    body = core[:-1] if core.endswith("。") else core
    offset = len(text) - len(text.lstrip())

    def bounds(match, group):
        start, end = match.span(group)
        return start + offset, end + offset

    def slot(match, group, key, content, shape, rule="identity"):
        return FragmentSlot(key, content, (bounds(match, group),), rule, shape)

    def label(match, group, role, key=None):
        return (*bounds(match, group), role, key)

    def operator(match, group, name, scope, **extra):
        return {"name": name, "source_span": list(bounds(match, group)), "scope_slots": scope, **extra}

    match = _ADVERBIAL.fullmatch(body)
    if match:
        slots = [slot(match, "adjective", "adjective", _I_FORMS[match["adjective"]], "adjective", "closed_i_adjective_ku_to_dictionary")]
        labels = [label(match, "adjective", "content", "adjective")]
        operators = [operator(match, "adjective", "open_adverbial_tail", ["adjective"], limitation="No missing predicate is generated.")]
        if match["time"]:
            slots.insert(0, slot(match, "time", "time", match["time"], "temporal"))
            labels.append(label(match, "time", "content", "time"))
        if match["degree"]:
            labels.append(label(match, "degree", "degree_operator"))
            operators.append(operator(match, "degree", "degree", ["adjective"], form=match["degree"]))
        return Analysis(_plan("open_i_adjective_adverbial", text, slots, labels, operators), None)

    match = _NOMINALIZATION.fullmatch(body)
    if match:
        if not _simple_noun(match["subject"]) or not _simple_noun(match["object"]):
            return Analysis(None, "complex_nominalized_action_arguments")
        slots = [slot(match, "subject", "subject", match["subject"], _noun_shape(match["subject"])),
                 slot(match, "object", "object", match["object"], _noun_shape(match["object"])),
                 slot(match, "action", "action", _TARI_FORMS[match["action"]], "verb", "closed_tari_form_to_dictionary")]
        labels = [label(match, "subject", "content", "subject"), label(match, "object", "content", "object"),
                  label(match, "action", "content", "action"), label(match, "subject_case", "subject_case"),
                  label(match, "object_case", "object_case"), label(match, "tail", "open_nominalization_operator")]
        operators = [operator(match, "tail", "open_nominalized_exemplified_action", ["subject", "action", "object"],
                              limitation="No matrix predicate, ability or inability is supplied.")]
        if match["degree"]:
            labels.append(label(match, "degree", "degree_operator"))
            operators.append(operator(match, "degree", "degree", ["action"], form=match["degree"],
                                      establishes_negative=False))
        return Analysis(_plan("open_exemplified_action_nominalization", text, slots, labels, operators), None)

    match = _CONDITION.fullmatch(body)
    if match:
        if not _simple_noun(match["noun"]):
            return Analysis(None, "complex_existential_condition_noun")
        slots = [slot(match, "noun", "condition_noun", match["noun"], _noun_shape(match["noun"])),
                 slot(match, "action", "action", _TE_MO_FORMS[match["action"]], "verb", "closed_te_mo_form_to_dictionary")]
        labels = [label(match, "noun", "content", "condition_noun"), label(match, "condition", "existential_condition_operator"),
                  label(match, "action", "content_and_open_te_mo", "action")]
        operators = [operator(match, "condition", "existential_condition", ["condition_noun", "action"]),
                     operator(match, "action", "open_te_mo_tail", ["action"],
                              limitation="Following permission/concession/conclusion is unexpressed and remains unexpressed.")]
        return Analysis(_plan("open_existential_condition_te_mo", text, slots, labels, operators), None)

    match = _WHENEVER.fullmatch(body)
    if match:
        if not _simple_noun(match["topic"]):
            return Analysis(None, "complex_whenever_topic")
        slots = [slot(match, "topic", "topic", match["topic"], _noun_shape(match["topic"]))]
        labels = [label(match, "topic", "content", "topic"), label(match, "topic_case", "topic_operator"),
                  label(match, "whenever", "open_whenever_operator")]
        operators = [operator(match, "whenever", "open_whenever_tail", ["topic"], limitation="No habitual predicate or conclusion is added.")]
        return Analysis(_plan("open_topic_whenever", text, slots, labels, operators), None)
    return Analysis(None, "unsupported_fragment_structure")
