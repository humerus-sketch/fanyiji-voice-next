"""V2 grammar is sent before generation; raw output is never repaired afterward."""
from collections import deque
from types import SimpleNamespace
import hashlib

import pytest

from fanyiji_next.gemma_translation import GemmaTranslationEngine, translation_prompt


class Runtime:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.calls = []

    def start(self):
        pass

    def close(self):
        pass

    def post_json(self, route, body):
        self.calls.append((route, body))
        if route == "/tokenize":
            return {"tokens": [2, 19, 105]}
        assert route == "/completion"
        result = self.outputs.popleft()
        if isinstance(result, Exception):
            raise result
        return {"content": result, "stopped_eos": True}


@pytest.fixture
def create(tmp_path):
    dictionary = tmp_path / "lexical.txt"
    dictionary.write_text("文献 5 n\n读 5 v\n清晰 5 a\n我 5 r\n今日 5 t\n", encoding="utf-8")
    engines = []

    def factory(outputs, policy="source_structure_v2"):
        runtime = Runtime(outputs)
        settings = SimpleNamespace(hymt_server_exe="unused", gemma_model_file="unused", translation_device="cuda")
        engine = GemmaTranslationEngine(settings, runtime_factory=lambda *a, **kw: runtime,
                                        prompt_transport="token_ids", repair_policy=policy,
                                        structure_lexicon_path=dictionary)
        engines.append(engine)
        return engine, runtime

    yield factory
    for engine in engines:
        engine.close()


def test_every_shaped_slot_sends_frozen_grammar_and_same_references(create):
    engine, runtime = create(["我", "文献", "读"])
    source, refs = "僕が本を読んだりするのが", ["別の原文", "一つ前の原文"]
    result = engine.translate(source, "ja_to_zh", source_language="ja", reference_texts=refs)
    assert result == "我读文献之类的事……"
    trace = engine.last_structure_trace
    assert trace["supported"] and not trace["fallback_used"]
    assert len(runtime.calls) == 6
    assert [s["output_shape"] for s in trace["slots"]] == ["pronoun", "noun_phrase", "verb"]
    for index, slot in enumerate(trace["slots"]):
        tokenized, generated = runtime.calls[index * 2:index * 2 + 2]
        assert tokenized == ("/tokenize", {"content": translation_prompt(slot["source_text"], "ja", "zh", refs),
                                            "add_special": False, "parse_special": True})
        grammar = generated[1]["grammar"]
        assert slot["grammar_sha256"] == hashlib.sha256(grammar.encode("utf-8")).hexdigest()
        assert slot["grammar_size"] == len(grammar.encode("utf-8"))
        assert generated[1]["prompt"] == [2, 19, 105]
        assert generated[1]["temperature"] == 0.0
    assert list(engine._history[("default", "ja", "zh")]) == [source]


def test_raw_semantic_error_is_not_deleted_or_hidden_by_slot_processing(create):
    engine, runtime = create([" 我。 ", " 文献。 ", " 不会，不会。\n"])
    result = engine.translate("僕が本を読んだりするのが", "ja_to_zh", source_language="ja")
    assert result == "我。不会，不会。文献。之类的事……"
    assert [s["raw"] for s in engine.last_structure_trace["slots"]] == [" 我。 ", " 文献。 ", " 不会，不会。\n"]
    assert len(runtime.calls) == 6


def test_grammar_request_error_is_not_retried_or_changed_to_unconstrained(create):
    engine, runtime = create([RuntimeError("grammar rejected")])
    with pytest.raises(RuntimeError, match="grammar rejected"):
        engine.translate("僕が本を読んだりするのが", "ja_to_zh", source_language="ja")
    assert len(runtime.calls) == 2 and "grammar" in runtime.calls[1][1]
    assert engine.last_structure_trace["status"] == "error"
    assert list(engine._history[("default", "ja", "zh")]) == []


def test_v2_unsupported_keeps_current_full_prompt_and_no_grammar(create):
    engine, runtime = create(["原样整句输出。"])
    source = "知らない構造です。"
    assert engine.translate(source, "ja_to_zh", source_language="ja", reference_texts=[]) == "原样整句输出。"
    assert len(runtime.calls) == 2 and "grammar" not in runtime.calls[1][1]
    assert runtime.calls[0][1]["content"] == translation_prompt(source, "ja", "zh", [])
    assert engine.last_structure_trace["supported"] is False and engine.last_structure_trace["fallback_used"] is True


def test_v1_still_sends_no_grammar_and_preserves_delegate_render(create):
    engine, runtime = create(["付款。"], policy="source_structure_v1")
    assert engine.translate("まだ払わないでください。", "ja_to_zh", source_language="ja") == "请暂时不要付款。。"
    assert len(runtime.calls) == 2 and "grammar" not in runtime.calls[1][1]
    assert engine._structure_grammar is None
