"""Generic v2 fragment contracts; no runtime, corpus or answer file is used."""
import json

import pytest

from fanyiji_next.gemma_fragment_structure import analyze_source_v2
from fanyiji_next.gemma_source_structure import analyze_source, StructureRenderError


@pytest.mark.parametrize("source", ["まだ読まないでください。", "雨だから待たないんじゃない。", "乾いてないけど白いシャツがね。"])
def test_v1_plans_render_identically_and_only_topic_shape_changes(source):
    old, new = analyze_source(source).plan, analyze_source_v2(source).plan
    assert old is not None and new is not None and new.delegate == old
    assert new.kind == old.kind
    assert [(s.key, s.source_text, s.source_spans, s.transformation) for s in old.slots] == [
        (s.key, s.source_text, s.source_spans, s.transformation) for s in new.slots]
    for slot in new.slots:
        assert slot.output_shape == ("noun_phrase" if old.kind == "unfinished_state_topic" and slot.key == "topic" else "free")
    raw = {slot.key: " 原样内容。\n重复！ " for slot in old.slots}
    assert new.render(raw) == old.render(raw)
    assert new.to_dict()["structure_version"] == 2


@pytest.mark.parametrize("source,adjective,time,degree", [
    ("今日少し悲しく", "悲しい", "今日", "少し"),
    ("もっと美しく。", "美しい", None, "もっと"),
    ("今うれしく", "うれしい", "今", None),
    ("ちょっと優しく", "優しい", None, "ちょっと"),
    ("楽しく", "楽しい", None, None),
    ("昨日詳しく", "詳しい", "昨日", None),
    ("新しく", "新しい", None, None),
    ("古く", "古い", None, None),
    ("正しく", "正しい", None, None),
    ("難しく", "難しい", None, None),
    ("易しく", "易しい", None, None),
    ("厳しく", "厳しい", None, None),
])
def test_i_adjective_renyou_is_an_open_adverbial_without_new_predicate(source, adjective, time, degree):
    plan = analyze_source_v2(source).plan
    assert plan.kind == "open_i_adjective_adverbial"
    slots = {s.key: s for s in plan.slots}
    assert slots["adjective"].source_text == adjective and slots["adjective"].output_shape == "adjective"
    assert (slots["time"].source_text if "time" in slots else None) == time
    if time:
        assert slots["time"].output_shape == "temporal"
    assert next((op["form"] for op in plan.operators if op["name"] == "degree"), None) == degree
    values = {key: "悲伤" if key == "adjective" else "今天" for key in slots}
    result = plan.render(values)
    assert result.endswith("悲伤地……")
    assert "开心" not in result and "变得" not in result and "会" not in result


@pytest.mark.parametrize("source,subject,object_text,lemma,subject_shape", [
    ("彼が本を時々読んだりするのが", "彼", "本", "読む", "pronoun"),
    ("作業員が箱を運んだりするのが", "作業員", "箱", "運ぶ", "noun_phrase"),
    ("私が手順を覚えたりするのが。", "私", "手順", "覚える", "pronoun"),
])
def test_tari_nominalization_uses_closed_godan_and_ichidan_forms(source, subject, object_text, lemma, subject_shape):
    plan = analyze_source_v2(source).plan
    assert plan.kind == "open_exemplified_action_nominalization"
    slots = {s.key: s for s in plan.slots}
    assert slots["subject"].source_text == subject and slots["subject"].output_shape == subject_shape
    assert slots["object"].source_text == object_text and slots["object"].output_shape == "noun_phrase"
    assert slots["action"].source_text == lemma and slots["action"].output_shape == "verb"
    result = plan.render({"subject": "他", "object": "书", "action": "读"})
    assert result.endswith("读书之类的事……")
    assert "可以" not in result and "不能" not in result


def test_zenzen_does_not_create_unexpressed_negative_or_inability():
    plan = analyze_source_v2("君が詩を全然書いたりするのが").plan
    assert plan is not None
    degree = next(op for op in plan.operators if op["name"] == "degree")
    assert degree["form"] == "全然" and degree["establishes_negative"] is False
    assert plan.render({"subject": "你", "object": "诗", "action": "写"}) == "你完全写诗之类的事……"


@pytest.mark.parametrize("source,noun,action", [
    ("説明があれば聞いても", "説明", "聞く"),
    ("お金があれば買っても。", "お金", "買う"),
    ("時間があれば読んでも", "時間", "読む"),
])
def test_existential_condition_te_mo_leaves_following_modality_unexpressed(source, noun, action):
    plan = analyze_source_v2(source).plan
    assert plan.kind == "open_existential_condition_te_mo"
    assert [(s.key, s.source_text, s.output_shape) for s in plan.slots] == [
        ("condition_noun", noun, "noun_phrase"), ("action", action, "verb")]
    result = plan.render({"condition_noun": "说明", "action": "听"})
    assert result == "如果有说明，听也……"
    assert all(word not in result for word in ["我", "请", "可以", "应该", "即使"])


@pytest.mark.parametrize("source,topic,shape", [("絵本はいつだって", "絵本", "noun_phrase"), ("君はいつだって。", "君", "pronoun")])
def test_whenever_topic_has_no_invented_habitual_conclusion(source, topic, shape):
    plan = analyze_source_v2(source).plan
    assert plan.kind == "open_topic_whenever"
    assert [(s.key, s.source_text, s.output_shape) for s in plan.slots] == [("topic", topic, shape)]
    assert plan.render({"topic": "绘本"}) == "绘本，无论什么时候……"


@pytest.mark.parametrize("source,key", [
    ("私が彼女を嫌ったりするのが", "object"),
    ("彼があれば聞いても", "condition_noun"),
    ("乾いてないけど彼女がね。", "topic"),
])
def test_pronouns_use_pronoun_shape_in_every_noun_argument(source, key):
    plan = analyze_source_v2(source).plan
    assert plan is not None
    assert next(s for s in plan.slots if s.key == key).output_shape == "pronoun"


@pytest.mark.parametrize("source", [
    "RTX5090はいつだって", "資料2はいつだって", "資料２はいつだって",
    "私がRTX5090を買ったりするのが", "GPUがあれば買っても",
    "乾いてないけどRTX5090がね。",
])
def test_identifiers_decline_without_forcing_a_lossy_noun_shape(source):
    if source.startswith("乾いて"):
        assert analyze_source(source).plan is not None
    result = analyze_source_v2(source)
    assert result.plan is None and result.reason


@pytest.mark.parametrize("source", [
    "今ちょっと静かに", "今ちょっと書く", "今ちょっとなく", "もっとよく", "今日も楽しく",
    "私が箱をバズったりするのが", "私が箱を読めたりするのが", "私が本を読むのが",
    "私が本を読んだりするのが楽しい", "私が本を読まなかったりするのが",
    "会社で働く人が箱を運んだりするのが", "私が会社で使う箱を運んだりするのが",
    "説明があれば聞いてもいい", "説明があれば聞いても構わない", "説明があれば聞かなくても",
    "説明さえあれば聞いても", "説明があれば聞く", "会社で使う説明があれば聞いても",
    "絵本はいつだって面白い", "絵本がいつだって", "会社で読む本はいつだって",
    "「今少し悲しく」", "説明があれば聞いても？", "君はいつだって!",
    "今少し悲しく\n別の文", "私が本を読んだりするのが。別の文。",
])
def test_unknown_morphology_quotes_nested_scope_and_completed_continuations_decline(source):
    result = analyze_source_v2(source)
    assert result.plan is None and result.reason


@pytest.mark.parametrize("source", [None, 7, "", " \n "])
def test_invalid_sources_return_explicit_decline(source):
    result = analyze_source_v2(source)
    assert result.plan is None and result.reason


@pytest.mark.parametrize("source", [
    " 今日少し悲しく。 \n", "作業員が箱を運んだりするのが", "説明があれば聞いても", "絵本はいつだって",
])
def test_every_source_code_point_has_a_lossless_span_and_shapes_serialize(source):
    plan = analyze_source_v2(source).plan
    data = plan.to_dict()
    cursor = 0
    for span in data["source_spans"]:
        assert span["start"] == cursor and span["end"] > span["start"]
        assert span["text"] == source[span["start"]:span["end"]]
        cursor = span["end"]
    assert cursor == len(source) and "".join(s["text"] for s in data["source_spans"]) == source
    assert data["source_text"] == source
    for record, slot in zip(data["slots"], plan.slots):
        assert record["output_shape"] == slot.output_shape
        assert all(0 <= start < end <= len(source) for start, end in slot.source_spans)
    assert json.loads(json.dumps(data, ensure_ascii=False)) == data


def test_render_preserves_wrong_words_and_all_model_punctuation():
    plan = analyze_source_v2("彼が本を読んだりするのが").plan
    raw = {"subject": " 他。 ", "object": " 不要这本书！ ", "action": " 不能读\n解释。 "}
    rendered = plan.render(raw)
    assert rendered == raw["subject"].strip() + raw["action"].strip() + raw["object"].strip() + "之类的事……"
    assert all(value.strip() in rendered for value in raw.values())


@pytest.mark.parametrize("values", [None, [], {}, {"topic": ""}, {"topic": " \n "}, {"topic": None}, {"topic": 1}, {"topic": "本", "extra": "字"}])
def test_invalid_slots_fail_without_substituting_any_content(values):
    plan = analyze_source_v2("絵本はいつだって").plan
    with pytest.raises(StructureRenderError, match="未交付"):
        plan.render(values)
