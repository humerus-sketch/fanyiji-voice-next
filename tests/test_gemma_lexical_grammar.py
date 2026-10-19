"""Corpus-independent lexical constraints; no model or downloaded resources."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fanyiji_next.gemma_lexical_grammar import LexicalGrammar, _prefix_expression
from fanyiji_next.gemma_translation import GemmaTranslationEngine


@pytest.fixture
def dictionary(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("湖泊 20 n\n池塘 20 n\n研究 30 vn\n观察 17 v\n没有 15 v\n宽广 10 a\n确切 9 ad\n我 7 r\n他们 8 rr\n往昔 4 t\n目前 9 t\n私名 100 nr\n英文abc 99 n\n坏频率 nope n\n", encoding="utf-8")
    return path


def test_frequency_selection_is_source_independent_and_stable(dictionary):
    lexicon = LexicalGrammar(dictionary)
    assert lexicon.words == {"noun": ("研究", "池塘", "湖泊"), "verb": ("观察", "没有"),
                             "adjective": ("宽广", "确切"), "pronoun": ("他们", "我"), "temporal": ("目前", "往昔")}
    assert lexicon.metadata["dictionary_sha256"] == hashlib.sha256(dictionary.read_bytes()).hexdigest()
    assert lexicon.metadata["selected_counts"] == {"noun": 3, "verb": 2, "adjective": 2, "pronoun": 2, "temporal": 2}
    first = {s: lexicon.for_shape(s) for s in lexicon.SHAPES}
    dictionary.write_text("\n".join(reversed(dictionary.read_text(encoding="utf-8").splitlines())), encoding="utf-8")
    second = LexicalGrammar(dictionary)
    assert first == {s: second.for_shape(s) for s in second.SHAPES}


def test_shapes_have_no_free_unicode_or_sentence_escape(dictionary):
    lexicon = LexicalGrammar(dictionary)
    assert lexicon.for_shape("free") is None
    for shape in lexicon.SHAPES:
        grammar = lexicon.for_shape(shape)
        assert "[" not in grammar and ".*" not in grammar
        assert "<" not in grammar and "source" not in grammar
        assert lexicon.describe(grammar)["size_bytes"] == len(grammar.encode("utf-8"))
    assert 'verb "的"' in lexicon.for_shape("noun_phrase")
    assert "没有" in lexicon.words["verb"]  # No hidden negation blacklist.
    with pytest.raises(ValueError, match="未知"):
        lexicon.for_shape("arbitrary_sentence")


def test_global_truncation_is_by_frequency_not_input_or_reference(dictionary, monkeypatch):
    monkeypatch.setattr(LexicalGrammar, "LIMITS", {"noun": 1, "verb": 1, "adjective": 1, "pronoun": 1, "temporal": 1})
    lexicon = LexicalGrammar(dictionary)
    assert lexicon.words == {"noun": ("研究",), "verb": ("观察",), "adjective": ("宽广",), "pronoun": ("他们",), "temporal": ("目前",)}


def test_missing_local_dictionary_and_incomplete_pos_fail_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        LexicalGrammar(tmp_path / "absent.txt")
    empty = tmp_path / "incomplete.txt"
    empty.write_text("山川 1 n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不能退"):
        LexicalGrammar(empty)


def test_v2_requires_explicit_dictionary_before_runtime_creation():
    factory = Mock()
    with pytest.raises(ValueError, match="已有本地"):
        GemmaTranslationEngine(SimpleNamespace(), runtime_factory=factory,
                               repair_policy="source_structure_v2", prompt_transport="token_ids")
    factory.assert_not_called()


def test_v2_bad_dictionary_never_starts_runtime(tmp_path):
    factory = Mock()
    with pytest.raises(FileNotFoundError):
        GemmaTranslationEngine(SimpleNamespace(), runtime_factory=factory,
                               repair_policy="source_structure_v2", prompt_transport="token_ids",
                               structure_lexicon_path=tmp_path / "missing.txt")
    factory.assert_not_called()


def test_none_ignores_optional_dictionary_path_and_keeps_backend_schema():
    runtime = Mock()
    settings = SimpleNamespace(hymt_server_exe="unused", gemma_model_file="unused", translation_device="cuda")
    engine = GemmaTranslationEngine(settings, runtime_factory=lambda *a, **kw: runtime,
                                   structure_lexicon_path="nonexistent")
    try:
        assert engine._structure_grammar is None
        assert "structure_lexicon" not in engine.backend_info
        assert engine.repair_policy == "none" and engine.prompt_transport == "string"
    finally:
        engine.close()


def test_grammar_metadata_and_json_are_serializable(dictionary):
    lexicon = LexicalGrammar(dictionary)
    for shape, detail in lexicon.metadata["grammars"].items():
        assert len(detail["sha256"]) == 64
        assert detail["sha256"] == hashlib.sha256(lexicon.for_shape(shape).encode()).hexdigest()
    assert json.loads(json.dumps(lexicon.metadata, ensure_ascii=False)) == lexicon.metadata


def test_atomic_shapes_do_not_reference_combinatorial_phrase_rules(dictionary):
    lexicon = LexicalGrammar(dictionary)
    for shape in ("noun", "modifier"):
        lines = lexicon.for_shape(shape).splitlines()
        assert len(lines) == 2
        assert lines[0] == f"root ::= {shape}"
        assert "noun-phrase" not in lines[1] and '"的"' not in lines[1]
        assert "?" not in lines[1]  # This fixture has no prefix words.


def test_modifier_is_existing_global_content_union_without_new_bilingual_entries(dictionary):
    lexicon = LexicalGrammar(dictionary)
    combined = sorted(set(lexicon.words["noun"]) | set(lexicon.words["verb"]) | set(lexicon.words["adjective"]))
    assert lexicon.for_shape("modifier") == "root ::= modifier\nmodifier ::= " + _prefix_expression(combined) + "\n"


@pytest.mark.parametrize("words,expected", [
    (["河", "河流", "河岸"], '"河" (("岸" | "流"))?'),
    (["远方"], '"远方"'),
    (["知识", "知觉", "地方"], '("地方" | "知" ("觉" | "识"))'),
])
def test_prefix_factoring_retains_complete_words_and_prefix_terminal(words, expected):
    assert _prefix_expression(words) == expected
    assert _prefix_expression(reversed(words)) == expected
