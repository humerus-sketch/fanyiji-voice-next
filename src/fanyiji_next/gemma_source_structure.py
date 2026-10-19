"""Finite source-only Japanese transduction; unknown structures are declined.

This module neither generates translations nor judges lexical slot accuracy.
Offsets refer to Unicode code points in the complete, unchanged source string.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re


class StructureRenderError(ValueError):
    pass


@dataclass(frozen=True)
class ContentSlot:
    key: str
    source_text: str
    source_spans: tuple[tuple[int, int], ...]
    transformation: str = "identity"


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    role: str
    slot_key: str | None = None


@dataclass(frozen=True)
class StructurePlan:
    kind: str
    source_text: str
    slots: tuple[ContentSlot, ...]
    spans: tuple[SourceSpan, ...]
    operators: tuple[dict, ...]

    def render(self, values: Mapping[str, str]) -> str:
        expected = {slot.key for slot in self.slots}
        if not isinstance(values, Mapping) or set(values) != expected:
            raise StructureRenderError("内容槽缺失或包含未声明的槽；未交付结构翻译。")
        clean = {}
        for key, raw in values.items():
            if not isinstance(raw, str) or not raw.strip():
                raise StructureRenderError("内容槽必须是非空文本；未交付结构翻译。")
            # Never remove punctuation, words or model-generated explanations.
            # Their semantic correctness belongs to the independent evaluation.
            clean[key] = raw.strip()
        if self.kind == "deferred_prohibition":
            topic = clean["object"] + "，" if "object" in clean else ""
            return topic + "请暂时不要" + clean["action"] + "。"
        if self.kind == "denied_reason":
            return "并不是因为" + clean["reason"] + "才" + clean["negative_event"] + "。"
        if self.kind == "unfinished_state_topic":
            return "还没有" + clean["event"] + "，不过" + clean["topic"] + "嘛……"
        raise StructureRenderError("未知结构计划；未交付结构翻译。")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "source_text": self.source_text,
            "offset_unit": "unicode_code_points",
            "slots": [{"key": slot.key, "source_text": slot.source_text,
                       "source_spans": [list(span) for span in slot.source_spans],
                       "transformation": slot.transformation} for slot in self.slots],
            "source_spans": [{"start": span.start, "end": span.end,
                              "text": self.source_text[span.start:span.end],
                              "role": span.role, "slot_key": span.slot_key} for span in self.spans],
            "operators": [dict(operator) for operator in self.operators],
        }


@dataclass(frozen=True)
class Analysis:
    plan: StructurePlan | None
    reason: str | None


# Closed Japanese morphology vocabulary, not a bilingual answer dictionary.
# A-row restoration is accepted only when the derived lemma is in this set.
_GODAN_LEMMAS = frozenset({
    "支払う", "払う", "送る", "歌う", "書く", "読む", "運ぶ", "待つ", "話す", "買う",
    "戻る", "開く", "切る", "作る", "飲む", "遊ぶ", "急ぐ", "消す", "触る", "取る",
    "使う", "受け取る", "返す", "選ぶ", "呼ぶ", "持つ", "置く",
})
_A_TO_U = {"わ": "う", "か": "く", "が": "ぐ", "さ": "す", "た": "つ",
           "な": "ぬ", "ば": "ぶ", "ま": "む", "ら": "る"}
_ICHIDAN_STEMS = {"見": "見る", "食べ": "食べる", "閉め": "閉める", "借り": "借りる", "届け": "届ける"}
# Contracted te-inai cannot generally be reversed from its spelling. Only these
# explicitly recorded Japanese form/lemma pairs are recognized in this version.
_TE_FORMS = {"終わって": "終わる", "始まって": "始まる", "決まって": "決まる",
             "集まって": "集まる", "届いて": "届く", "乾いて": "乾く"}
_JP = r"[一-龥々ぁ-ゖァ-ヺーA-Za-z0-9０-９]+"
_OBJECT = r"(?:これ|それ|あれ|(?:この|その|あの)[一-龥々ァ-ヺA-Za-z0-9０-９]{1,16})"
_DEFER = re.compile(rf"(?:(?P<object>{_OBJECT})(?P<topic>は))?(?P<defer>まだ)"
                    rf"(?P<stem>{_JP})(?P<negative>ない)(?P<request>でください)")
_REASON = re.compile(rf"(?P<reason>{_JP})(?P<because>から)(?P<event>{_JP}ない)(?P<outer>んじゃない)")
_STATE = re.compile(rf"(?P<form>{'|'.join(map(re.escape, _TE_FORMS))})(?P<negative>ない)"
                    rf"(?P<contrast>けど)(?P<topic>{_JP})(?P<tail>がね)")
_QUOTING = frozenset('「」『』“”\"\'()（）[]{}〈〉《》【】')
_NESTED = ("なら", "たら", "ても", "ので", "わけ", "とは", "では", "じゃ", "ません", "なかった", "なく")
_TOPIC_SYNTAX = ("から", "けど", "が", "は", "を", "に", "へ", "と", "で", "も", "こと", "もの")


def _lemma(stem: str) -> tuple[str, str] | None:
    if stem in _ICHIDAN_STEMS:
        return _ICHIDAN_STEMS[stem], "closed_ichidan_stem_to_dictionary"
    if stem and stem[-1] in _A_TO_U:
        lemma = stem[:-1] + _A_TO_U[stem[-1]]
        if lemma in _GODAN_LEMMAS:
            return lemma, "closed_godan_a_to_u"
    return None


def _make_plan(kind, source, slots, labels, operators):
    spans, cursor = [], 0
    for start, end, role, key in sorted(labels):
        if start < cursor or end <= start:
            raise ValueError("Overlapping or empty source ledger span")
        if start > cursor:
            spans.append(SourceSpan(cursor, start, "formatting"))
        spans.append(SourceSpan(start, end, role, key))
        cursor = end
    if cursor < len(source):
        spans.append(SourceSpan(cursor, len(source), "formatting"))
    return StructurePlan(kind, source, tuple(slots), tuple(spans), tuple(operators))


def analyze_source(text: str) -> Analysis:
    if not isinstance(text, str):
        return Analysis(None, "source_is_not_text")
    core = text.strip()
    if not core:
        return Analysis(None, "empty_source")
    if any(char in _QUOTING for char in core):
        return Analysis(None, "quoted_or_bracketed_source")
    if any(char in core for char in "?？!！\n\r"):
        return Analysis(None, "question_exclamation_or_multiple_lines")
    offset = len(text) - len(text.lstrip())
    body = core[:-1] if core.endswith("。") else core

    def bounds(match, name):
        start, end = match.span(name)
        return start + offset, end + offset

    def label(match, name, role, key=None):
        return (*bounds(match, name), role, key)

    match = _DEFER.fullmatch(body)
    if match:
        if body.count("ない") != 1:
            return Analysis(None, "nested_or_repeated_negation")
        restored = _lemma(match["stem"])
        if restored is None:
            return Analysis(None, "unknown_or_irregular_negative_verb")
        lemma, rule = restored
        slots = [ContentSlot("action", lemma, (bounds(match, "stem"),), rule)]
        labels = [label(match, "defer", "temporal_operator"), label(match, "stem", "content", "action"),
                  label(match, "negative", "negation_operator"), label(match, "request", "directive_operator")]
        if match["object"]:
            slots.insert(0, ContentSlot("object", match["object"], (bounds(match, "object"),)))
            labels += [label(match, "object", "content", "object"), label(match, "topic", "topic_operator")]
        operators = [{"name": "deferred_prohibition", "scope_slots": [slot.key for slot in slots],
                      "source_span": list(bounds(match, "defer")), "realization": "renderer"},
                     {"name": "negative_directive", "source_span": [bounds(match, "negative")[0], bounds(match, "request")[1]],
                      "scope_slots": ["action"], "realization": "renderer"}]
        return Analysis(_make_plan("deferred_prohibition", text, slots, labels, operators), None)

    match = _REASON.fullmatch(body)
    if match:
        # A written period is a conservative boundary, not proof of spoken
        # intonation. Unpunctuated conversational n-ja-nai remains unsupported.
        if not core.endswith("。"):
            return Analysis(None, "unpunctuated_reason_mood_is_ambiguous")
        if body.count("ない") != 2 or body.count("から") != 1 or any(
                marker in match["reason"] or marker in match["event"] for marker in _NESTED):
            return Analysis(None, "nested_or_ambiguous_reason_negation")
        if _lemma(match["event"][:-2]) is None:
            return Analysis(None, "unknown_inner_negative_verb")
        slots = [ContentSlot("reason", match["reason"], (bounds(match, "reason"),)),
                 ContentSlot("negative_event", match["event"], (bounds(match, "event"),))]
        labels = [label(match, "reason", "content", "reason"), label(match, "because", "causal_operator"),
                  label(match, "event", "content", "negative_event"), label(match, "outer", "outer_negation_operator")]
        operators = [{"name": "deny_reason_attribution", "source_span": list(bounds(match, "outer")),
                      "scope_slots": ["reason", "negative_event"], "realization": "renderer",
                      "limitation": "Explicit written period required; spoken rhetorical or conjectural intonation is not resolved."},
                     {"name": "inner_event_negation", "source_span": [bounds(match, "event")[1] - 2, bounds(match, "event")[1]],
                      "scope_slots": ["negative_event"], "realization": "model_content_slot",
                      "limitation": "Inner negation remains dependent on the Q4 slot translation."}]
        return Analysis(_make_plan("denied_reason", text, slots, labels, operators), None)

    match = _STATE.fullmatch(body)
    if match:
        if body.count("ない") != 1 or body.count("けど") != 1 or any(marker in match["topic"] for marker in _NESTED):
            return Analysis(None, "nested_or_ambiguous_unfinished_topic")
        # This is only a conservative noun-like surface shape, not a POS or
        # dependency parser. Decline particles and predicate-like endings.
        if (any(marker in match["topic"] for marker in _TOPIC_SYNTAX)
                or not re.search(r"[一-龥々ァ-ヺA-Za-z0-9０-９]$", match["topic"])):
            return Analysis(None, "topic_not_a_simple_noun_like_surface")
        slots = [ContentSlot("event", _TE_FORMS[match["form"]], (bounds(match, "form"),), "closed_te_inai_form_to_dictionary"),
                 ContentSlot("topic", match["topic"], (bounds(match, "topic"),))]
        labels = [label(match, "form", "content", "event"), label(match, "negative", "incomplete_state_operator"),
                  label(match, "contrast", "contrast_operator"), label(match, "topic", "content", "topic"),
                  label(match, "tail", "unfinished_topic_operator")]
        operators = [{"name": "not_completed", "source_span": list(bounds(match, "negative")),
                      "scope_slots": ["event"], "realization": "renderer"},
                     {"name": "contrast", "source_span": list(bounds(match, "contrast")),
                      "scope_slots": ["event", "topic"], "realization": "renderer"},
                     {"name": "unfinished_topic", "source_span": list(bounds(match, "tail")),
                      "scope_slots": ["topic"], "realization": "renderer",
                      "limitation": "Conservative noun-like spelling only; no general Japanese POS or dependency analysis."}]
        return Analysis(_make_plan("unfinished_state_topic", text, slots, labels, operators), None)
    return Analysis(None, "unsupported_source_structure")
