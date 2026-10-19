"""Finite grammar contracts using independent vocabulary, without a model."""
import json

import pytest

from fanyiji_next.gemma_source_structure import analyze_source, StructureRenderError


@pytest.mark.parametrize("source,expected", [
    ("まだ書かないでください。", "書く"), ("まだ読まないでください。", "読む"),
    ("まだ運ばないでください。", "運ぶ"), ("まだ待たないでください。", "待つ"),
    ("まだ話さないでください。", "話す"), ("まだ買わないでください。", "買う"),
    ("まだ急がないでください。", "急ぐ"), ("まだ戻らないでください。", "戻る"),
    ("まだ閉めないでください。", "閉める"), ("まだ見ないでください。", "見る"),
])
def test_deferred_directive_closed_morphology(source, expected):
    analysis = analyze_source(source)
    assert analysis.reason is None and analysis.plan.kind == "deferred_prohibition"
    assert [(slot.key, slot.source_text) for slot in analysis.plan.slots] == [("action", expected)]
    assert analysis.plan.render({"action": " 内容动作。 "}) == "请暂时不要内容动作。。"


def test_demonstrative_object_remains_an_independent_full_source_slot():
    plan = analyze_source("その箱はまだ開かないでください。").plan
    assert [(slot.key, slot.source_text) for slot in plan.slots] == [("object", "その箱"), ("action", "開く")]
    assert plan.render({"object": "那个箱子", "action": "打开"}) == "那个箱子，请暂时不要打开。"
    assert plan.operators[0]["scope_slots"] == ["object", "action"]


@pytest.mark.parametrize("source,reason,event", [
    ("雨だから待たないんじゃない。", "雨だ", "待たない"),
    ("忙しいから読まないんじゃない。", "忙しい", "読まない"),
])
def test_reason_denial_scopes_the_relation_and_preserves_negative_event(source, reason, event):
    plan = analyze_source(source).plan
    assert plan.kind == "denied_reason"
    assert [(slot.key, slot.source_text) for slot in plan.slots] == [("reason", reason), ("negative_event", event)]
    outer, inner = plan.operators
    assert outer["name"] == "deny_reason_attribution" and outer["scope_slots"] == ["reason", "negative_event"]
    assert inner["realization"] == "model_content_slot" and "Q4" in inner["limitation"]
    assert plan.render({"reason": "下雨", "negative_event": "不等待"}) == "并不是因为下雨才不等待。"


@pytest.mark.parametrize("source,lemma,topic", [
    ("乾いてないけど白いシャツがね。", "乾く", "白いシャツ"),
    ("届いてないけど新しい荷物がね。", "届く", "新しい荷物"),
    ("始まってないけど面白い大会がね。", "始まる", "面白い大会"),
])
def test_closed_te_form_and_incomplete_topic_are_separate_source_operators(source, lemma, topic):
    plan = analyze_source(source).plan
    assert plan.kind == "unfinished_state_topic"
    assert [(slot.key, slot.source_text) for slot in plan.slots] == [("event", lemma), ("topic", topic)]
    assert [operator["name"] for operator in plan.operators] == ["not_completed", "contrast", "unfinished_topic"]
    assert plan.render({"event": "变干", "topic": "白衬衫"}) == "还没有变干，不过白衬衫嘛……"


@pytest.mark.parametrize("source", [
    "まだ読まない。", "まだ読まないで待ってください。", "まだ読まないでくださいと頼みました。",
    "「まだ読まないでください。」", "『まだ書かないでください。』", "まだ読まないでください？",
    "まだ読まないでください!", "(まだ読まないでください。)", "まだ読まないでください。次の文。",
    "まだしないでください。", "まだ来ないでください。", "まだ知られないでください。",
    "まだカニらないでください。", "まだ切ないでください。", "まだ足りないでください。",
    "私はまだ読まないでください。", "荷物はまだ読まないでください。",
    "雨だから読まないんじゃないか。", "読まないから待たないんじゃない。",
    "雨だから待たないんじゃない",
    "雨なら読まないから待たないんじゃない。", "暇じゃないから読まないんじゃない。",
    "雨だから来ないんじゃない。", "雨だから大丈夫じゃないんじゃない。",
    "走ってないけど新しい靴がね。", "行ってないけど遠い場所がね。",
    "乾いてないけど濡れてないシャツがね。", "乾いてないけど新しいシャツですね。",
    "動いてないけど新しい機械がね。", "続いてないけど長い大会がね。",
    "乾いてないけど明日来るがね。", "乾いてないけど会社で働く人がね。",
    "乾いてないけど資料は大事がね。", "乾いてないけど暑いから休む人がね。",
    "乾いてないけど休むけど別の人がね。", "乾いてないけど白いがね。",
    "乾いてないけどシャツがね。\n次の文。", "まだ全く知らないでください。",
])
def test_unsupported_or_ambiguous_source_is_declined_without_guessing(source):
    analysis = analyze_source(source)
    assert analysis.plan is None and analysis.reason


@pytest.mark.parametrize("source", [None, 1, "", " \n "])
def test_invalid_or_empty_source_has_explicit_unsupported_reason(source):
    analysis = analyze_source(source)
    assert analysis.plan is None and analysis.reason


@pytest.mark.parametrize("source", [
    " その箱はまだ開かないでください。 \n", "雨だから待たないんじゃない。", "乾いてないけど白いシャツがね。",
])
def test_lossless_source_ledger_including_outer_whitespace_and_punctuation(source):
    plan = analyze_source(source).plan
    assert plan is not None
    cursor = 0
    for span in plan.spans:
        assert span.start == cursor and span.end > span.start
        cursor = span.end
    assert cursor == len(source)
    data = plan.to_dict()
    assert "".join(span["text"] for span in data["source_spans"]) == source
    assert data["source_text"] == source and data["offset_unit"] == "unicode_code_points"
    for slot in plan.slots:
        assert slot.source_text
        assert all(0 <= start < end <= len(source) for start, end in slot.source_spans)
        assert all(any(span.slot_key == slot.key and (span.start, span.end) == original
                       for span in plan.spans) for original in slot.source_spans)
    assert json.loads(json.dumps(data, ensure_ascii=False)) == data


def test_render_preserves_every_model_word_and_punctuation_even_when_wrong():
    plan = analyze_source("その箱はまだ開かないでください。").plan
    values = {"object": "  那个箱子。\n ", "action": " 不要打开！\n解释\n "+"重复。" * 2}
    result = plan.render(values)
    assert result == values["object"].strip() + "，请暂时不要" + values["action"].strip() + "。"
    assert "不要打开！\n解释\n 重复。重复。" in result
    assert values["object"].strip() in result and values["action"].strip() in result


@pytest.mark.parametrize("values", [None, [], {}, {"action": ""}, {"action": " \n "},
                                    {"action": None}, {"action": 2}, {"action": "动作", "extra": "值"}])
def test_render_missing_empty_nontext_or_extra_slots_fail_without_substitution(values):
    plan = analyze_source("まだ書かないでください。").plan
    with pytest.raises(StructureRenderError, match="未交付"):
        plan.render(values)
