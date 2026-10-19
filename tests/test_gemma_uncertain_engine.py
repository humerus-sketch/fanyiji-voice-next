"""Candidate-only source routing, unchanged prompts and no output rewriting."""
from collections import deque
from types import SimpleNamespace

import pytest

from fanyiji_next.gemma_atom_grammar import AtomGrammar
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
        value = self.outputs.popleft()
        if isinstance(value, Exception):
            raise value
        return {"content": value, "stopped_eos": True}


def create(outputs, policy="source_structure_v5", transport="token_ids"):
    runtime = Runtime(outputs)
    settings = SimpleNamespace(hymt_server_exe="unused", gemma_model_file="unused", translation_device="cuda")
    engine = GemmaTranslationEngine(settings, runtime_factory=lambda *a, **k: runtime,
                                   repair_policy=policy, prompt_transport=transport)
    return engine, runtime


@pytest.mark.parametrize("source,outputs,expected", [
    ("まだ乾いてないけどね、今朝。", ["干燥", "今晨"], "不过，今晨还没有干燥呢。"),
    ("今週はもうなんか聞いてるの", ["本周", "听"], "本周，呃，已经在听呢。"),
    ("僕が書類を全然読んだりするのが", ["他", "资料", "读"], "像他读资料这样的事，完全……"),
])
def test_v5_uses_each_source_slot_once_with_same_current_prompt_and_two_references(source, outputs, expected):
    engine, runtime = create(outputs)
    history = ["明日は図書館に行きます。", "荷物を準備します。"]
    try:
        assert engine.translate(source, "ja_to_zh", source_language="ja", reference_texts=history) == expected
        trace = engine.last_structure_trace
        assert trace["supported"] and not trace["fallback_used"]
        tokenizes = [body for route, body in runtime.calls if route == "/tokenize"]
        completions = [body for route, body in runtime.calls if route == "/completion"]
        assert len(completions) == len(outputs) == len(trace["slots"])
        assert [b["content"] for b in tokenizes] == [translation_prompt(s["source_text"], "ja", "zh", history) for s in trace["slots"]]
        assert [s["raw"] for s in trace["slots"]] == outputs
        for slot, body in zip(trace["slots"], completions):
            assert body["prompt"] == [2, 19, 105]
            assert body["grammar"] == AtomGrammar().for_shape(slot["output_shape"])
            assert body["temperature"] == 0.0 and body["n_predict"] == 256
        assert engine.backend_info["structure_lexicon"]["dictionary_used"] is False
        assert trace["plan"]["unresolved_scope"]  # Diagnostic, never a pass flag.
        assert list(engine._history.values())[0] == deque([source], maxlen=2)
    finally:
        engine.close()


def test_v5_keeps_wrong_raw_content_including_negation_and_punctuation():
    engine, runtime = create(["不该。\n删除", "不确定"])
    try:
        value = engine.translate("まだ乾いてないけどね、今朝。", "ja_to_zh", source_language="ja", reference_texts=[])
        assert value == "不过，不确定还没有不该。\n删除呢。"
        assert engine.last_structure_trace["slots"][0]["raw"] == "不该。\n删除"
    finally:
        engine.close()


def test_v5_slot_error_does_not_retry_correct_or_return_partial_translation():
    engine, runtime = create(["干燥", RuntimeError("fixture timeout")])
    try:
        with pytest.raises(RuntimeError, match="fixture timeout"):
            engine.translate("まだ乾いてないけどね、今朝。", "ja_to_zh", source_language="ja", reference_texts=[])
        assert len([1 for route, _ in runtime.calls if route == "/completion"]) == 2
        assert engine.last_structure_trace["status"] == "error"
        assert not any(engine._history.values())
    finally:
        engine.close()


@pytest.mark.parametrize("source", ["未知の文字列", "ナミダにしよう", "今週は知ってるの", "まだ乾いてないけどね、来年。"])
def test_unsupported_inputs_really_translate_original_full_text_without_claiming_repair(source):
    engine, runtime = create(["未修饰的完整原始输出。"])
    history = ["先の発言。", "次の発言。"]
    try:
        assert engine.translate(source, "ja_to_zh", source_language="ja", reference_texts=history) == "未修饰的完整原始输出。"
        assert engine.last_structure_trace["supported"] is False
        assert engine.last_structure_trace["fallback_used"] is True
        assert runtime.calls[0][1]["content"] == translation_prompt(source, "ja", "zh", history)
        assert len(runtime.calls) == 2 and "grammar" not in runtime.calls[1][1]
    finally:
        engine.close()


def test_default_none_never_invokes_v5_parser(monkeypatch):
    import fanyiji_next.gemma_uncertain_structure as structures

    def forbidden(_):
        raise AssertionError("Default must not parse candidate structures")

    monkeypatch.setattr(structures, "analyze_source_v5", forbidden)
    engine, runtime = create(["完整原始输出。"], policy="none", transport="string")
    try:
        assert engine.translate("まだ乾いてないけどね、今朝。", "ja_to_zh", source_language="ja") == "完整原始输出。"
        assert engine.last_structure_trace is None
        assert len(runtime.calls) == 2 and "grammar" not in runtime.calls[1][1]
        assert engine.prompt_variant == "current" and engine.repair_policy == "none"
    finally:
        engine.close()
