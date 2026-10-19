"""Opt-in source-syntax additions with explicit unresolved scope.

The closed Japanese form tables are morphology, not lexical translations.
Unrecognized sources retain the unchanged v3 analysis and current fallback.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

from .gemma_fragment_atoms import AtomSlot, analyze_source_v3
from .gemma_source_structure import Analysis, SourceSpan, StructurePlan, StructureRenderError


@dataclass(frozen=True)
class UncertainPlan(StructurePlan):
    delegate: StructurePlan | None = None
    unresolved_scope: tuple[dict, ...] = ()

    def render(self, values: Mapping[str, str]) -> str:
        if not isinstance(values, Mapping) or set(values) != {s.key for s in self.slots}:
            raise StructureRenderError("内容槽缺失或包含未声明的槽；未交付来源翻译。")
        if any(not isinstance(v, str) or not v.strip() for v in values.values()):
            raise StructureRenderError("内容槽必须是非空文本；未交付来源翻译。")
        raw = {key: value.strip() for key, value in values.items()}
        if self.kind == "open_exemplified_event_unresolved_degree":
            parent = {b["parent_slot_key"]: b["joiner"].join(raw[k] for k in b["atom_keys"])
                      for b in self.delegate.bindings}
            event = parent["subject"] + parent["action"] + parent["object"]
            return "像" + event + "这样的事，完全……"
        if self.kind == "pending_event_postposed_time":
            particle = "呢" if any(op["name"] == "interaction_particle" for op in self.operators) else ""
            return "不过，" + raw["time"] + "还没有" + raw["event"] + particle + "。"
        if self.kind == "time_topic_positive_continuative":
            filler = "呃，" if any(op["name"] == "discourse_filler" for op in self.operators) else ""
            already = "已经" if any(op["name"] == "already" for op in self.operators) else ""
            return raw["time"] + "，" + filler + already + "在" + raw["event"] + "呢。"
        raise StructureRenderError("未知来源结构；未交付来源翻译。")

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["structure_version"] = 5
        for item, slot in zip(data["slots"], self.slots):
            item["output_shape"] = slot.output_shape
        data["unresolved_scope"] = [dict(item) for item in self.unresolved_scope]
        if self.delegate is not None:
            data["delegate_plan"] = self.delegate.to_dict()
        return data


_PENDING_TE_FORMS = {
    "終わって": "終わる", "始まって": "始まる", "決まって": "決まる",
    "集まって": "集まる", "届いて": "届く", "乾いて": "乾く",
}
# Only bounded action verbs: result states such as shitte-iru or kekkon-shite-iru
# cannot safely be rendered as the same action-continuative construction.
_ACTION_CONTINUATIVES = {
    "選んでる": "選ぶ", "読んでる": "読む", "書いてる": "書く", "聞いてる": "聞く",
    "待ってる": "待つ", "話してる": "話す", "運んでる": "運ぶ",
}
_PENDING_TIMES = ("今", "今日", "昨日", "最近", "今朝", "昨夜", "今年", "去年", "今週", "先週", "今月", "先月")
_CONTINUATIVE_TIMES = _PENDING_TIMES + ("毎日", "毎朝", "毎晩")


def _alternatives(values):
    return "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))


_PENDING = re.compile(rf"(?P<defer>まだ)(?P<form>{_alternatives(_PENDING_TE_FORMS)})(?P<negative>ない)"
                      rf"(?P<contrast>けど)(?P<particle>ね)?(?P<comma>、)(?P<time>{_alternatives(_PENDING_TIMES)})(?P<period>。)")
_CONTINUATIVE = re.compile(rf"(?P<time>{_alternatives(_CONTINUATIVE_TIMES)})(?P<topic>は)(?P<already>もう)?(?P<filler>なんか)?"
                           rf"(?P<form>{_alternatives(_ACTION_CONTINUATIVES)})(?P<explanation>の)(?P<period>。)?")


def _make_plan(kind, text, slots, labels, operators, unresolved):
    spans, cursor = [], 0
    for start, end, role, key in sorted(labels):
        if start < cursor or end <= start:
            raise ValueError("Invalid or overlapping source ledger")
        if cursor < start:
            spans.append(SourceSpan(cursor, start, "formatting"))
        spans.append(SourceSpan(start, end, role, key))
        cursor = end
    if cursor < len(text):
        spans.append(SourceSpan(cursor, len(text), "formatting"))
    return UncertainPlan(kind, text, tuple(slots), tuple(spans), tuple(operators), None, tuple(unresolved))


def _open_degree(parent):
    degree = next((op for op in parent.operators if op["name"] == "degree" and op.get("form") == "全然"), None)
    if parent.kind != "open_exemplified_action_nominalization" or degree is None:
        return None
    event_keys = [key for binding in parent.bindings if binding["parent_slot_key"] in {"subject", "action", "object"}
                  for key in binding["atom_keys"]]
    operators = []
    for op in parent.operators:
        if op is degree:
            operators.append({**op, "name": "unresolved_degree_tail", "scope_slots": event_keys,
                              "attachment": "unresolved_event_or_following_predicate", "polarity": "unexpressed"})
        else:
            operators.append(dict(op))
    action = next(slot for slot in parent.slots if slot.key == "action")
    start, end = action.source_spans[0]
    operators.append({"name": "whole_event_exemplification", "source_span": [end - 2, end],
                      "source_form": parent.source_text[end - 2:end], "scope_slots": event_keys})
    unresolved = ({"kind": "degree_attachment_and_polarity_unresolved", "source_span": degree["source_span"],
                   "polarity_inferred": False, "missing_predicate_generated": False},)
    return UncertainPlan("open_exemplified_event_unresolved_degree", parent.source_text, parent.slots,
                         parent.spans, tuple(operators), parent, unresolved)


def analyze_source_v5(text: str) -> Analysis:
    original = analyze_source_v3(text)
    if original.plan is not None:
        changed = _open_degree(original.plan)
        return Analysis(changed, None) if changed is not None else original
    if not isinstance(text, str) or original.reason in {
            "empty_source", "quoted_or_bracketed_source", "question_exclamation_or_multiple_lines"}:
        return original
    core = text.strip()
    offset = len(text) - len(text.lstrip())

    def bounds(match, group):
        start, end = match.span(group)
        return start + offset, end + offset

    def slot(match, group, key, lemma, shape, rule="identity"):
        return AtomSlot(key, lemma, (bounds(match, group),), rule, shape)

    def label(match, group, role, key=None):
        return (*bounds(match, group), role, key)

    def operator(match, group, name, scope, **kwargs):
        return {"name": name, "source_span": list(bounds(match, group)), "scope_slots": scope, **kwargs}

    match = _PENDING.fullmatch(core)
    if match:
        slots = [slot(match, "form", "event", _PENDING_TE_FORMS[match["form"]], "verb", "closed_te_form_to_dictionary"),
                 slot(match, "time", "time", match["time"], "temporal")]
        labels = [label(match, "defer", "pending_operator"), label(match, "form", "content", "event"),
                  label(match, "negative", "negative_operator"), label(match, "contrast", "contrast_operator"),
                  label(match, "comma", "postposition_boundary"), label(match, "time", "content", "time"),
                  label(match, "period", "statement_boundary")]
        operators = [operator(match, "defer", "not_yet", ["event"]), operator(match, "negative", "negative", ["event"]),
                     operator(match, "contrast", "contrast", ["event", "time"]),
                     operator(match, "time", "postposed_time_anchor", ["time", "event"], omitted_arguments_generated=False)]
        if match["particle"]:
            labels.append(label(match, "particle", "interaction_particle"))
            operators.append(operator(match, "particle", "interaction_particle", ["event"]))
        unresolved = [{"kind": "unexpressed_event_arguments", "arguments_generated": False}]
        return Analysis(_make_plan("pending_event_postposed_time", text, slots, labels, operators, unresolved), None)

    match = _CONTINUATIVE.fullmatch(core)
    if match:
        slots = [slot(match, "time", "time", match["time"], "temporal"),
                 slot(match, "form", "event", _ACTION_CONTINUATIVES[match["form"]], "verb", "closed_action_continuative_to_dictionary")]
        labels = [label(match, "time", "content", "time"), label(match, "topic", "topic_operator"),
                  label(match, "form", "content_and_continuative", "event"), label(match, "explanation", "explanatory_particle")]
        operators = [operator(match, "topic", "time_topic", ["time", "event"]),
                     operator(match, "form", "positive_action_continuative", ["event"],
                              polarity="positive", instantaneous_present_claimed=False),
                     operator(match, "explanation", "explanatory_particle", ["event"])]
        for group, name in (("already", "already"), ("filler", "discourse_filler")):
            if match[group]:
                labels.append(label(match, group, name))
                interpretation = {"interpretation": "candidate_assumption_with_unresolved_alternative"} if group == "filler" else {}
                operators.append(operator(match, group, name, ["event"], **interpretation))
        if match["period"]:
            labels.append(label(match, "period", "statement_boundary"))
        unresolved = [{"kind": "habitual_vs_in_progress", "resolved": False},
                      {"kind": "spoken_explanation_vs_question_intonation", "audio_verified": False}]
        if match["filler"]:
            unresolved.append({"kind": "filler_or_indefinite_argument", "resolved": False})
        return Analysis(_make_plan("time_topic_positive_continuative", text, slots, labels, operators, unresolved), None)
    return original
