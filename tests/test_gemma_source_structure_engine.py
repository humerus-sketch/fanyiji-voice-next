"""Candidate isolation, lossless content transport and failure contracts."""
from collections import deque
from types import SimpleNamespace
import threading
from unittest.mock import Mock

import pytest

from fanyiji_next.gemma_translation import GemmaTranslationEngine, translation_prompt
from fanyiji_next.translation_engines import TranslationCancelled


class Runtime:
    def __init__(self):
        self.calls = []
        self.outputs = deque()
        self.after_call = None
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True

    def post_json(self, route, body):
        self.calls.append((route, body))
        if self.after_call:
            self.after_call(len(self.calls))
        if route == "/tokenize":
            return {"tokens": [2, 19, 105]}
        assert route == "/completion"
        value = self.outputs.popleft() if self.outputs else "完整的原始内容。"
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, dict) else {"content": value, "stopped_eos": True}


class Plan:
    def __init__(self):
        self.slots = [SimpleNamespace(key="a", source_text="内容甲"),
                      SimpleNamespace(key="b", source_text="内容乙")]
        self.received = None

    def to_dict(self):
        return {"kind": "test_plan"}

    def render(self, values):
        self.received = dict(values)
        return "关系甲[" + values["a"].strip() + "]关系乙[" + values["b"].strip() + "]"


@pytest.fixture
def make_engine():
    owned = []

    def create(*, policy="source_structure_v1", transport="token_ids", stop_event=None):
        runtime = Runtime()
        settings = SimpleNamespace(hymt_server_exe="unused", gemma_model_file="unused",
                                   translation_device="cuda")
        engine = GemmaTranslationEngine(settings, runtime_factory=lambda *a, **kw: runtime,
                                        repair_policy=policy, prompt_transport=transport,
                                        stop_event=stop_event)
        owned.append(engine)
        return engine, runtime

    yield create
    for engine in owned:
        engine.close()


@pytest.mark.parametrize("variant,transport", [("current", "string"), ("native_single_ja_zh", "token_ids"),
                                              ("role_task_ja_zh_v3", "token_ids")])
def test_candidate_invalid_config_fails_before_runtime_creation(variant, transport):
    factory = Mock()
    with pytest.raises(ValueError, match="current.*token_ids"):
        GemmaTranslationEngine(SimpleNamespace(), runtime_factory=factory,
                               prompt_variant=variant, prompt_transport=transport,
                               repair_policy="source_structure_v1")
    factory.assert_not_called()


@pytest.mark.parametrize("source,direction", [(None, "ja_to_zh"), ("en", "ja_to_zh"),
                                             ("zh", "ja_to_zh"), ("ja", "zh_to_ja")])
def test_candidate_language_rejected_without_requests(make_engine, source, direction):
    engine, runtime = make_engine()
    with pytest.raises(ValueError, match="仅支持明确"):
        engine.translate("原文", direction, source_language=source)
    assert runtime.calls == [] and engine._history == {}


def patch_analysis(monkeypatch, plan=None, reason=None):
    import fanyiji_next.gemma_source_structure as structures
    analyze = Mock(return_value=SimpleNamespace(plan=plan, reason=reason))
    monkeypatch.setattr(structures, "analyze_source", analyze)
    return analyze


def test_unsupported_uses_exact_current_prompt_and_preserves_raw(make_engine, monkeypatch):
    engine, runtime = make_engine()
    patch_analysis(monkeypatch, reason="ambiguous_scope")
    runtime.outputs.append("  未过滤的不、不、暂时。 \n")
    source, refs = "未知结构", ["参考甲", "参考乙"]
    value = engine.translate(source, "ja_to_zh", source_language="ja", reference_texts=refs)
    assert value == "未过滤的不、不、暂时。"
    assert runtime.calls[0] == ("/tokenize", {"content": translation_prompt(source, "ja", "zh", refs),
                                             "add_special": False, "parse_special": True})
    assert len(runtime.calls) == 2
    assert engine.last_structure_trace["supported"] is False
    assert engine.last_structure_trace["fallback_used"] is True
    assert engine.last_structure_trace["unsupported_reason"] == "ambiguous_scope"
    assert list(engine._history[("default", "ja", "zh")]) == [source]


def test_supported_has_no_full_sentence_draft_or_self_review(make_engine, monkeypatch):
    engine, runtime = make_engine()
    plan = Plan()
    analyze = patch_analysis(monkeypatch, plan=plan)
    first, second = " \n不、不、不会。 ", " 先不要，先不要！\n"
    runtime.outputs.extend([first, second])
    refs = ["过旧参考", "参考甲", "参考乙"]
    result = engine.translate("整句原文", "ja_to_zh", source_language="ja", reference_texts=refs)
    assert result == "关系甲[不、不、不会。]关系乙[先不要，先不要！]"
    analyze.assert_called_once_with("整句原文")
    assert plan.received == {"a": first, "b": second}
    assert len(runtime.calls) == 4
    assert [runtime.calls[i][1]["content"] for i in (0, 2)] == [
        translation_prompt(slot.source_text, "ja", "zh", refs[-2:]) for slot in plan.slots]
    assert [runtime.calls[i][1]["prompt"] for i in (1, 3)] == [[2, 19, 105], [2, 19, 105]]
    assert [slot["raw"] for slot in engine.last_structure_trace["slots"]] == [first, second]
    assert engine.last_structure_trace["fallback_used"] is False
    assert engine.last_structure_trace["status"] == "rendered"
    assert list(engine._history[("default", "ja", "zh")]) == ["整句原文"]


@pytest.mark.parametrize("bad", [RuntimeError("server error"), {"content": "", "stopped_eos": True},
                                 {"content": "被截断", "stopped_limit": True},
                                 {"content": "被截断", "truncated": True}])
def test_slot_failure_never_returns_partial_or_appends_history(make_engine, monkeypatch, bad):
    engine, runtime = make_engine()
    patch_analysis(monkeypatch, plan=Plan())
    runtime.outputs.extend(["已完成第一槽", bad])
    with pytest.raises(RuntimeError):
        engine.translate("整句原文", "ja_to_zh", source_language="ja", reference_texts=[])
    assert len(runtime.calls) == 4
    assert list(engine._history[("default", "ja", "zh")]) == []
    assert engine.last_structure_trace["status"] == "error"
    assert engine.last_structure_trace["fallback_used"] is False


def test_cancellation_between_slots_does_not_render_or_append(make_engine, monkeypatch):
    stop = threading.Event()
    engine, runtime = make_engine(stop_event=stop)
    plan = Plan()
    patch_analysis(monkeypatch, plan=plan)
    runtime.after_call = lambda count: stop.set() if count == 2 else None
    with pytest.raises(TranslationCancelled):
        engine.translate("整句原文", "ja_to_zh", source_language="ja")
    assert len(runtime.calls) == 2 and plan.received is None
    assert list(engine._history[("default", "ja", "zh")]) == []


def test_none_never_invokes_structure_parser(make_engine, monkeypatch):
    engine, runtime = make_engine(policy="none", transport="string")
    analyze = patch_analysis(monkeypatch, plan=Plan())
    assert engine.translate("任何原文", "ja_to_zh", source_language="ja") == "完整的原始内容。"
    analyze.assert_not_called()
    assert len(runtime.calls) == 2
    assert engine.last_structure_trace is None and "repair_policy" not in engine.backend_info


def test_trace_resets_between_calls_and_explicit_history_remains_source_only(make_engine, monkeypatch):
    engine, runtime = make_engine()
    patch_analysis(monkeypatch, plan=Plan())
    engine.translate("第一整句", "ja_to_zh", source_language="ja", reference_texts=[])
    first_trace = engine.last_structure_trace
    patch_analysis(monkeypatch, reason="unsupported")
    engine.translate("第二整句", "ja_to_zh", source_language="ja", reference_texts=["明确参考"])
    assert first_trace["status"] == "rendered" and len(first_trace["slots"]) == 2
    assert engine.last_structure_trace is not first_trace
    assert engine.last_structure_trace["supported"] is False
    assert list(engine._history[("default", "ja", "zh")]) == ["第一整句", "第二整句"]
