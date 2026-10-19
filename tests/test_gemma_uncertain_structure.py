"""Generic source syntax contracts; no corpus, answers, model or runtime."""
import json

import pytest

from fanyiji_next.gemma_fragment_atoms import analyze_source_v3
from fanyiji_next.gemma_uncertain_structure import analyze_source_v5
from fanyiji_next.gemma_source_structure import StructureRenderError


def test_zenzen_tari_scopes_the_whole_event_without_completing_polarity():
    source = "僕が書類を全然読んだりするのが"
    plan = analyze_source_v5(source).plan
    assert plan.kind == "open_exemplified_event_unresolved_degree"
    assert plan.delegate == analyze_source_v3(source).plan
    values = {s.key: {"subject": "他", "action": "读", "object__head": "资料"}[s.key] for s in plan.slots}
    assert plan.render(values) == "像他读资料这样的事，完全……"
    scope = next(op for op in plan.operators if op["name"] == "whole_event_exemplification")
    assert set(scope["scope_slots"]) == set(values)
    degree = next(op for op in plan.operators if op["name"] == "unresolved_degree_tail")
    assert degree["polarity"] == "unexpressed" and degree["attachment"] == "unresolved_event_or_following_predicate"
    assert not plan.unresolved_scope[0]["polarity_inferred"]


@pytest.mark.parametrize("source", [
    "彼が本を時々読んだりするのが", "まだ書かないでください。", "雨だから待たないんじゃない。",
    "君はいつだって", "乾いてないけど整った設備がね。", "今少し悲しく",
])
def test_existing_nonzenzen_plans_keep_exact_v3_analysis_and_render(source):
    old, new = analyze_source_v3(source), analyze_source_v5(source)
    assert new == old
    values = {s.key: " 原样。\n内容！ " for s in old.plan.slots}
    assert new.plan.render(values) == old.plan.render(values)


@pytest.mark.parametrize("source,event,time,particle", [
    ("まだ乾いてないけどね、今朝。", "乾く", "今朝", True),
    ("まだ届いてないけど、今日。", "届く", "今日", False),
    ("まだ集まってないけどね、今週。", "集まる", "今週", True),
])
def test_pending_event_and_postposed_time_keep_negation_contrast_and_particle(source, event, time, particle):
    plan = analyze_source_v5(source).plan
    assert plan.kind == "pending_event_postposed_time"
    assert [(s.key, s.source_text, s.output_shape) for s in plan.slots] == [("event", event, "verb"), ("time", time, "temporal")]
    assert plan.render({"event": "干燥", "time": "今晨"}) == "不过，今晨还没有干燥" + ("呢" if particle else "") + "。"
    assert not next(op for op in plan.operators if op["name"] == "postposed_time_anchor")["omitted_arguments_generated"]


@pytest.mark.parametrize("source,lemma,already,filler", [
    ("今朝は書いてるの", "書く", False, False),
    ("毎日はもう読んでるの。", "読む", True, False),
    ("今月はなんか待ってるの", "待つ", False, True),
    ("今週はもうなんか聞いてるの", "聞く", True, True),
])
def test_affirmative_action_continuative_does_not_become_willingness_or_instant_claim(source, lemma, already, filler):
    plan = analyze_source_v5(source).plan
    assert plan.kind == "time_topic_positive_continuative"
    assert next(s.source_text for s in plan.slots if s.key == "event") == lemma
    assert plan.render({"time": "今晨", "event": "书写"}) == "今晨，" + ("呃，" if filler else "") + ("已经" if already else "") + "在书写呢。"
    aspect = next(op for op in plan.operators if op["name"] == "positive_action_continuative")
    assert aspect["polarity"] == "positive" and not aspect["instantaneous_present_claimed"]
    assert plan.unresolved_scope[0] == {"kind": "habitual_vs_in_progress", "resolved": False}


def test_nanka_filler_is_an_explicit_assumption_not_a_resolved_semantic_pass():
    plan = analyze_source_v5("今はなんか読んでるの").plan
    data = plan.to_dict()
    assert {"kind": "filler_or_indefinite_argument", "resolved": False} in data["unresolved_scope"]
    filler = next(op for op in data["operators"] if op["name"] == "discourse_filler")
    assert filler["interpretation"] == "candidate_assumption_with_unresolved_alternative"
    assert plan.render({"time": "今时", "event": "阅读"}) == "今时，呃，在阅读呢。"
    assert "semantic_pass" not in data and "semantic_acceptance" not in data


@pytest.mark.parametrize("source", [
    "まだ動いてないけどね、今朝。", "まだ続いてないけど、今日。", "まだ乾いてるけどね、今朝。",
    "まだ乾いてないけどね、何時。", "まだ乾いてないけどね、今朝", "まだ乾いてないけどね今朝。",
    "まだ乾いてないけどね、今朝だよ。", "まだ乾いてないけどね、今朝？",
    "今朝は知ってるの", "今朝は結婚してるの", "今朝は住んでるの", "今朝は持ってるの",
    "今朝は読んでないの", "今朝は読んでたの", "今朝は読んでるの？", "今朝は読んでるから",
    "私は読んでるの", "今朝はもうなんか図書館で読んでるの", "「今朝は読んでるの」",
    "今朝は読んでるの\n別の句", "なにか", "カタカナにしよう",
    "明日は読んでるの", "来週は書いてるの", "来月は聞いてるの", "来年は待ってるの", "今夜は読んでるの",
    "まだ乾いてないけどね、明日。", "まだ集まってないけど、来週。", "まだ終わってないけどね、来月。",
    "まだ始まってないけど、来年。", "まだ決まってないけどね、今夜。",
    "まだ乾いてないけどね、毎日。", "まだ届いてないけど、毎朝。", "まだ集まってないけどね、毎晩。",
])
def test_unknown_states_scopes_or_missing_evidence_retain_original_fallback(source):
    assert analyze_source_v5(source) == analyze_source_v3(source)
    assert analyze_source_v5(source).plan is None


@pytest.mark.parametrize("source", [
    " 僕が書類を全然読んだりするのが。 \n", "まだ乾いてないけどね、今朝。", "今週はもうなんか聞いてるの",
])
def test_full_span_ledger_and_serialized_diagnostics_are_lossless(source):
    plan = analyze_source_v5(source).plan
    data = plan.to_dict()
    assert data["structure_version"] == 5
    cursor = 0
    for span in data["source_spans"]:
        assert span["start"] == cursor and span["end"] > span["start"]
        assert span["text"] == source[span["start"]:span["end"]]
        cursor = span["end"]
    assert cursor == len(source) and "".join(s["text"] for s in data["source_spans"]) == source
    for op in data["operators"]:
        if "source_span" in op:
            start, end = op["source_span"]
            assert 0 <= start < end <= len(source)
    assert data["unresolved_scope"] and json.loads(json.dumps(data, ensure_ascii=False)) == data


def test_raw_words_punctuation_and_internal_newlines_are_never_filtered():
    plan = analyze_source_v5("まだ乾いてないけどね、今朝。").plan
    raw = {"event": " 不要。\n解释！ ", "time": " 某时。 "}
    assert plan.render(raw) == "不过，某时。还没有不要。\n解释！呢。"


@pytest.mark.parametrize("source", [None, 4, "", " \n "])
def test_invalid_source_keeps_original_analysis(source):
    assert analyze_source_v5(source) == analyze_source_v3(source)


@pytest.mark.parametrize("values", [None, {}, {"event": "字", "time": ""}, {"event": None, "time": "字"}, {"event": "字", "time": "时", "extra": "错"}])
def test_invalid_raw_slots_fail_without_content_substitution(values):
    with pytest.raises(StructureRenderError, match="未交付"):
        analyze_source_v5("今朝は書いてるの").plan.render(values)
