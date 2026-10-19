"""Source-only NP atom contracts; no corpus, runtime or model is accessed."""
import json

import pytest

from fanyiji_next.gemma_fragment_atoms import analyze_source_v3
from fanyiji_next.gemma_fragment_structure import analyze_source_v2
from fanyiji_next.gemma_source_structure import StructureRenderError


@pytest.mark.parametrize("noun", ["図書館", "ネコ", "資料", "お金"])
def test_simple_noun_has_one_lexical_atom_without_added_de(noun):
    plan = analyze_source_v3(noun + "はいつだって").plan
    assert plan is not None
    assert [(s.source_text, s.output_shape) for s in plan.slots] == [(noun, "noun")]
    assert plan.render({plan.slots[0].key: " 内容。 "}) == "内容。，无论什么时候……"
    assert plan.bindings[0]["joiner"] == ""


@pytest.mark.parametrize("noun,left,right,lshape", [
    ("鳥の巣", "鳥", "巣", "modifier"),
    ("学校の図書館", "学校", "図書館", "modifier"),
    ("彼女の資料", "彼女", "資料", "pronoun"),
    ("お知らせの内容", "お知らせ", "内容", "modifier"),
    ("くらしの記録", "くらし", "記録", "modifier"),
    ("お祝いの写真", "お祝い", "写真", "modifier"),
    ("おすすめの道具", "おすすめ", "道具", "modifier"),
])
def test_source_genitive_splits_exact_source_and_preserves_pronouns(noun, left, right, lshape):
    plan = analyze_source_v3(noun + "はいつだって").plan
    assert [(s.source_text, s.output_shape) for s in plan.slots] == [(left, lshape), (right, "noun")]
    keys = [s.key for s in plan.slots]
    assert plan.render({keys[0]: "左", keys[1]: "右"}) == "左的右，无论什么时候……"
    assert plan.bindings[0]["split_rule"] == "source_genitive"


@pytest.mark.parametrize("form,lemma,head", [
    ("整った", "整う", "設備"), ("優れた", "優れる", "技術"),
    ("正しい", "正しい", "手順"), ("新しい", "新しい", "道具"),
    ("目立った", "目立つ", "傷"),
])
def test_closed_attributive_forms_split_without_guessing_unknown_lemmas(form, lemma, head):
    plan = analyze_source_v3("乾いてないけど" + form + head + "がね。").plan
    assert plan is not None
    atoms = [s for s in plan.slots if s.key.startswith("topic__")]
    assert [(s.source_text, s.output_shape) for s in atoms] == [(lemma, "modifier"), (head, "noun")]
    assert atoms[0].transformation == "closed_attributive_to_dictionary"


@pytest.mark.parametrize("prefix,head", [("管理", "方法"), ("保存", "場所"), ("開発", "予定"), ("検証", "結果")])
def test_VN_compounds_split_from_general_source_classes(prefix, head):
    plan = analyze_source_v3(prefix + head + "はいつだって").plan
    assert [(s.source_text, s.output_shape) for s in plan.slots] == [(prefix, "modifier"), (head, "noun")]
    assert plan.render({plan.slots[0].key: "修饰", plan.slots[1].key: "中心"}) == "修饰的中心，无论什么时候……"


@pytest.mark.parametrize("source", [
    "まだ読まないでください。", "雨だから待たないんじゃない。", "今少し悲しく",
    "君はいつだって", "私が彼女を嫌ったりするのが",
])
def test_non_NP_slots_keep_v2_source_shape_and_render(source):
    parent = analyze_source_v2(source).plan
    plan = analyze_source_v3(source).plan
    assert plan is not None and plan.delegate == parent
    assert [(s.key, s.source_text, s.output_shape) for s in plan.slots] == [
        (s.key, s.source_text, s.output_shape) for s in parent.slots]
    raw = {s.key: " 原样。\n内容！ " for s in plan.slots}
    assert plan.render(raw) == parent.render(raw)


@pytest.mark.parametrize("source", [
    "鳥の巣はいつだって", "乾いてないけど整った設備がね。", "保存場所はいつだって",
    "  学校の図書館はいつだって。 \n", "私が道具を運んだりするのが",
])
def test_expanded_spans_cover_every_source_codepoint_and_bindings_serialize(source):
    plan = analyze_source_v3(source).plan
    assert plan is not None
    data = plan.to_dict()
    assert data["structure_version"] == 3
    assert data["delegate_plan"] == analyze_source_v2(source).plan.to_dict()
    cursor = 0
    for span in data["source_spans"]:
        assert span["start"] == cursor and span["end"] > span["start"]
        assert span["text"] == source[span["start"]:span["end"]]
        cursor = span["end"]
    assert cursor == len(source)
    assert "".join(s["text"] for s in data["source_spans"]) == source
    assert all(s.output_shape != "noun_phrase" for s in plan.slots)
    assert {k for b in plan.bindings for k in b["atom_keys"]} == {s.key for s in plan.slots}
    assert json.loads(json.dumps(data, ensure_ascii=False)) == data


@pytest.mark.parametrize("noun", [
    "白くない紙", "走った少年", "鳥の巣の写真", "学校の新しい設備", "長く続く話", "静かな場所",
    "変わった人", "困った人", "変わった職員", "困った道具", "おどろきの資料",
])
def test_unknown_complex_NP_declines_instead_of_dropping_expressed_content(noun):
    source = "乾いてないけど" + noun + "がね。"
    result = analyze_source_v3(source)
    assert result.plan is None and result.reason


@pytest.mark.parametrize("source", [None, "", "RTX5090はいつだって", "説明があれば聞いてもいい", "「鳥の巣はいつだって」"])
def test_v2_unsupported_stays_unsupported(source):
    assert analyze_source_v3(source) == analyze_source_v2(source)


def test_complete_raw_contents_are_bound_without_punctuation_or_word_filtering():
    plan = analyze_source_v3("乾いてないけど鳥の巣がね。").plan
    raw = {s.key: " 不要。\n解释！ " for s in plan.slots}
    parent_raw = {b["parent_slot_key"]: b["joiner"].join(raw[k].strip() for k in b["atom_keys"]) for b in plan.bindings}
    assert plan.render(raw) == plan.delegate.render(parent_raw)
    assert all(value.strip() in plan.render(raw) for value in raw.values())


@pytest.mark.parametrize("values", [None, {}, {"topic__head": ""}, {"topic__head": None}, {"topic__head": "内容", "extra": "字"}])
def test_invalid_raw_atom_sets_fail_without_substitution(values):
    plan = analyze_source_v3("資料はいつだって").plan
    with pytest.raises(StructureRenderError, match="未交付"):
        plan.render(values)
