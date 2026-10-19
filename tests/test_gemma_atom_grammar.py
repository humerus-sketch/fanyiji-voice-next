"""Generic opt-in bounds and engine isolation; no real model or case answers."""
from collections import deque
import re
from types import SimpleNamespace

import pytest

from fanyiji_next.gemma_atom_grammar import AtomGrammar
from fanyiji_next.gemma_translation import GemmaTranslationEngine, translation_prompt


@pytest.mark.parametrize("shape", sorted(AtomGrammar.SHAPES))
def test_all_typed_shapes_share_source_independent_character_language(shape):
    grammar = AtomGrammar()
    rule = grammar.for_shape(shape)
    assert rule == 'root ::= [\\u3400-\\u9fff]{1,12}\n'
    pattern = re.compile(rule.removeprefix("root ::= ").strip())
    assert pattern.fullmatch("山川")
    assert pattern.fullmatch("没有去做任何事")  # No semantic/POS guarantee.
    assert not pattern.fullmatch("山" * 13)
    assert not pattern.fullmatch("ABC")
    assert not pattern.fullmatch("山川。")
    assert not pattern.fullmatch("")
    assert grammar.metadata["dictionary_used"] is False
    assert grammar.metadata["vocabulary_whitelist"] is False
    assert grammar.describe(rule) == grammar.metadata["grammars"][shape]


def test_unknown_and_composite_shapes_are_not_silently_accepted():
    grammar = AtomGrammar()
    assert grammar.for_shape("free") is None
    for shape in ("noun_phrase", "unknown"):
        with pytest.raises(ValueError, match="未知"):
            grammar.for_shape(shape)


class Runtime:
    def __init__(self):
        self.calls = []
        self.outputs = deque(["新的", "资料", "阅读"])

    def start(self):
        pass

    def close(self):
        pass

    def post_json(self, route, body):
        self.calls.append((route, body))
        if route == "/tokenize":
            return {"tokens": [2, 19, 105]}
        value = self.outputs.popleft()
        if isinstance(value, Exception):
            raise value
        return {"content": value, "stopped_eos": True}


def create():
    runtime = Runtime()
    settings = SimpleNamespace(hymt_server_exe="unused", gemma_model_file="unused", translation_device="cuda")
    engine = GemmaTranslationEngine(settings, runtime_factory=lambda *a, **k: runtime,
                                   repair_policy="source_structure_v4", prompt_transport="token_ids")
    return engine, runtime


def test_v4_constructs_without_dictionary_and_uses_unchanged_source_prompt_per_atom():
    engine, runtime = create()
    try:
        refs = ["明日は山へ行きます。", "荷物を準備します。"]
        engine.translate("新しい資料があれば読んでも", "ja_to_zh", source_language="ja", reference_texts=refs)
        trace = engine.last_structure_trace
        assert trace["supported"] and not trace["fallback_used"]
        assert [s["source_text"] for s in trace["slots"]] == ["新しい", "資料", "読む"]
        tokenizes = [body for route, body in runtime.calls if route == "/tokenize"]
        completions = [body for route, body in runtime.calls if route == "/completion"]
        assert [b["content"] for b in tokenizes] == [translation_prompt(s["source_text"], "ja", "zh", refs) for s in trace["slots"]]
        assert len(completions) == 3
        assert all(b["grammar"] == AtomGrammar().for_shape("noun") for b in completions)
        assert all(b["prompt"] == [2, 19, 105] for b in completions)
        assert engine.backend_info["structure_lexicon"]["dictionary_used"] is False
    finally:
        engine.close()


def test_v4_does_not_remove_wrong_content_or_punctuation_from_raw_slots():
    engine, runtime = create()
    runtime.outputs = deque(["没有新的", "错误。\n内容", "不阅读"])
    try:
        out = engine.translate("新しい資料があれば読んでも", "ja_to_zh", source_language="ja", reference_texts=[])
        assert out == "如果有没有新的的错误。\n内容，不阅读也……"
        assert [s["raw"] for s in engine.last_structure_trace["slots"]] == ["没有新的", "错误。\n内容", "不阅读"]
    finally:
        engine.close()


def test_v4_slot_failure_has_no_second_attempt_unconstrained_retry_or_partial_output():
    engine, runtime = create()
    runtime.outputs = deque(["新的", RuntimeError("request failed")])
    try:
        with pytest.raises(RuntimeError, match="request failed"):
            engine.translate("新しい資料があれば読んでも", "ja_to_zh", source_language="ja", reference_texts=[])
        assert len([1 for route, _ in runtime.calls if route == "/completion"]) == 2
        assert engine.last_structure_trace["status"] == "error"
        assert not any(engine._history.values())
    finally:
        engine.close()
