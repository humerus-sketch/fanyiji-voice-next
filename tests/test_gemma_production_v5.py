"""Production factory routing contracts; all inference is replaced by a fake runtime."""
from collections import deque
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest

import fanyiji_next.gemma_translation as gemma
from fanyiji_next.gemma_translation import GemmaTranslationEngine, translation_prompt
from fanyiji_next.hymt_runtime import HyRuntimeCancelled
from fanyiji_next.models import AppSettings
from fanyiji_next.translation_engines import TranslationCancelled, create_translation_engine


class Runtime:
    def __init__(self, outputs, *, start_error=None, on_request=None, token_counts=()):
        self.outputs = deque(outputs)
        self.start_error = start_error
        self.on_request = on_request
        self.token_counts = deque(token_counts)
        self.calls = []
        self.starts = 0
        self.closes = 0

    def start(self):
        self.starts += 1
        if self.start_error is not None:
            raise self.start_error

    def close(self):
        self.closes += 1

    def post_json(self, route, body):
        self.calls.append((route, body))
        if self.on_request is not None:
            self.on_request(route, body)
        if route == "/tokenize":
            count = self.token_counts.popleft() if self.token_counts else 3
            return {"tokens": [2, 19, 105][:count] if count <= 3 else [2] * count}
        value = self.outputs.popleft()
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, dict) else {"content": value, "stopped_eos": True}


@pytest.fixture
def production(monkeypatch):
    instances = []

    def create(outputs=(), *, stop_event=None, settings=None, **runtime_options):
        runtime = Runtime(outputs, **runtime_options)
        instances.append(runtime)
        if settings is None:
            settings = SimpleNamespace(translation_backend="translategemma", hymt_server_exe="existing-server",
                                       gemma_model_file="existing-Q4_K_M.gguf", translation_device="cuda",
                                       model_combination="B")
        original_settings = deepcopy(vars(settings))
        constructions = []

        def runtime_factory(*args, **kwargs):
            constructions.append((args, kwargs))
            return runtime

        # The real factory still supplies every production option to the real
        # engine. Only its external process boundary is replaced here.
        def injected_engine(settings, **kwargs):
            return GemmaTranslationEngine(settings, runtime_factory=runtime_factory, **kwargs)

        monkeypatch.setattr(gemma, "GemmaTranslationEngine", injected_engine)
        engine = create_translation_engine(settings, stop_event=stop_event)
        assert vars(settings) == original_settings
        return engine, runtime, constructions, settings

    create.instances = instances
    return create


def forbid_post_edit(*_args, **_kwargs):
    raise AssertionError("Production must not send a translation back for self-correction")


@pytest.mark.parametrize("combination,recognition", [("A", "qwen3_asr"), ("B", "fun_asr")])
def test_factory_accepts_existing_ab_settings_without_new_persisted_fields(production, combination, recognition):
    settings = AppSettings(translation_backend="translategemma", recognition_backend=recognition,
                           model_combination=combination, gemma_model_file="shared/existing-Q4_K_M.gguf",
                           hymt_server_exe="existing-server", translation_device="cuda")
    before = deepcopy(vars(settings))
    engine, runtime, constructions, actual = production(settings=settings)
    try:
        assert actual is settings and vars(settings) == before
        assert engine.application_mode and engine.repair_policy == "source_structure_v5"
        assert len(constructions) == runtime.starts == 1
        assert constructions[0][0][1] == "shared/existing-Q4_K_M.gguf"
        assert "repair_policy" not in vars(settings) and "prompt_transport" not in vars(settings)
    finally:
        engine.close()


def test_real_factory_enables_v5_on_one_existing_q4_runtime(production, monkeypatch):
    monkeypatch.setattr(GemmaTranslationEngine, "_post_edit_once", forbid_post_edit)
    stop = threading.Event()
    engine, runtime, constructions, settings = production(["干燥", "今晨"], stop_event=stop)
    try:
        assert engine.application_mode is True
        assert (engine.prompt_variant, engine.prompt_transport, engine.repair_policy) == (
            "current", "token_ids", "source_structure_v5")
        assert len(constructions) == runtime.starts == 1
        assert constructions[0] == ((settings.hymt_server_exe, settings.gemma_model_file), {
            "device": "cuda", "stop_event": stop, "ctx_size": 2048})
        references = ["最初の発言。", "次の発言。"]
        text = "まだ乾いてないけどね、今朝。"
        assert engine.translate(text, "ja_to_zh", source_language="ja", reference_texts=references) == (
            "不过，今晨还没有干燥呢。")
        trace = engine.last_structure_trace
        assert trace["supported"] and trace["status"] == "rendered"
        tokenizes = [body for route, body in runtime.calls if route == "/tokenize"]
        completions = [body for route, body in runtime.calls if route == "/completion"]
        assert [body["content"] for body in tokenizes] == [
            translation_prompt(slot["source_text"], "ja", "zh", references) for slot in trace["slots"]]
        assert all(body["parse_special"] is True and body["add_special"] is False for body in tokenizes)
        assert len(completions) == len(trace["slots"]) == 2
        assert all(body["prompt"] == [2, 19, 105] and "grammar" in body for body in completions)
        assert [slot["raw"] for slot in trace["slots"]] == ["干燥", "今晨"]
        assert list(engine._history[("default", "ja", "zh")]) == [text]
        assert len(constructions) == runtime.starts == 1
    finally:
        engine.close()
        engine.close()
    assert runtime.closes == 1 and not engine._history


def test_unsupported_japanese_preserves_full_current_prompt_with_token_ids(production, monkeypatch):
    monkeypatch.setattr(GemmaTranslationEngine, "_post_edit_once", forbid_post_edit)
    engine, runtime, _, _ = production(["完整原始译文。"])
    try:
        source = "未知の文字列"
        refs = ["以前の原文。", "直前の原文。"]
        assert engine.translate(source, "ja_to_zh", source_language="ja", reference_texts=refs) == "完整原始译文。"
        assert runtime.calls[0] == ("/tokenize", {
            "content": translation_prompt(source, "ja", "zh", refs), "add_special": False, "parse_special": True})
        assert runtime.calls[1][1]["prompt"] == [2, 19, 105]
        assert "grammar" not in runtime.calls[1][1]
        assert len(runtime.calls) == 2
        assert engine.last_structure_trace["status"] == "unsupported_current_output"
        assert engine.last_structure_trace["fallback_used"] is True
    finally:
        engine.close()


@pytest.mark.parametrize("source_language,direction,target", [
    (None, "ja_to_zh", "zh"), (None, "zh_to_ja", "ja"),
    ("en", "ja_to_zh", "zh"), ("en", "zh_to_ja", "ja"),
    ("zh", "zh_to_ja", "ja"), ("ko", "ja_to_zh", "zh"),
])
def test_other_or_unknown_languages_keep_legacy_string_without_guessing_japanese(
        production, monkeypatch, source_language, direction, target):
    import fanyiji_next.gemma_uncertain_structure as structures

    def no_parser(_text):
        raise AssertionError("Only explicitly recognized Japanese may use the Japanese parser")

    monkeypatch.setattr(structures, "analyze_source_v5", no_parser)
    monkeypatch.setattr(GemmaTranslationEngine, "_post_edit_once", forbid_post_edit)
    engine, runtime, _, _ = production(["  未清洗的完整输出！  "])
    source = "まだ乾いてないけどね、今朝。"
    refs = ["prior source", "previous source"]
    try:
        assert engine.translate(source, direction, source_language=source_language, reference_texts=refs) == "未清洗的完整输出！"
        prompt = translation_prompt(source, source_language, target, refs)
        assert runtime.calls[0] == ("/tokenize", {"content": prompt, "add_special": False})
        assert runtime.calls[1][1]["prompt"] == prompt
        assert "grammar" not in runtime.calls[1][1] and len(runtime.calls) == 2
        assert engine.prompt_transport == "token_ids"
        assert engine.last_structure_trace["status"] == "language_pair_current_output"
    finally:
        engine.close()


@pytest.mark.parametrize("language,direction", [("zh", "ja_to_zh"), ("ja", "zh_to_ja")])
def test_same_language_target_passes_original_without_model_requests(production, language, direction):
    engine, runtime, _, _ = production()
    try:
        source = "  原文。ご視聴ありがとうございました。  "
        assert engine.translate(source, direction, source_language=language) == source
        assert runtime.calls == [] and engine.last_structure_trace is None
        assert list(engine._history.values())[0] == deque([source], maxlen=2)
    finally:
        engine.close()


def test_language_target_and_channel_switches_keep_source_only_history_isolated(production):
    engine, runtime, _, _ = production(["译文一", "译文二", "译文三", "译文四", "译文五", "译文六"])
    try:
        steps = [
            ("English one", "en", "ja_to_zh", "system", "zh", []),
            ("English two", "en", "ja_to_zh", "system", "zh", ["English one"]),
            ("未知の文字列", "ja", "ja_to_zh", "system", "zh", []),
            ("Another channel", "en", "ja_to_zh", "microphone", "zh", []),
            ("Switch target", "en", "zh_to_ja", "system", "ja", []),
            ("Switch back", "en", "ja_to_zh", "system", "zh", []),
        ]
        for source, language, direction, channel, target, refs in steps:
            engine.translate(source, direction, context_id=channel, source_language=language)
            assert runtime.calls[-2][1]["content"] == translation_prompt(source, language, target, refs)
            assert isinstance(runtime.calls[-1][1]["prompt"], list) is (language == "ja" and target == "zh")
        assert engine.prompt_transport == "token_ids"
        assert set(engine._history) == {("system", "en", "zh"), ("microphone", "en", "zh")}
        assert list(engine._history[("system", "en", "zh")]) == ["Switch back"]
    finally:
        engine.close()


def test_legacy_fallback_context_trimming_preserves_entire_current_sentence_and_string(production):
    engine, runtime, _, _ = production(["完整译文"], token_counts=[1751, 3])
    try:
        source = "Current source must stay complete."
        assert engine.translate(source, "ja_to_zh", source_language=None,
                                reference_texts=["old one", "old two"]) == "完整译文"
        assert runtime.calls[0][1] == {
            "content": translation_prompt(source, None, "zh", ["old one", "old two"]), "add_special": False}
        prompt = translation_prompt(source, None, "zh")
        assert runtime.calls[1][1] == {"content": prompt, "add_special": False}
        assert runtime.calls[2][1]["prompt"] == prompt
    finally:
        engine.close()


@pytest.mark.parametrize("source_language,source", [(None, "original"), ("ja", "未知の文字列"),
                                                       ("ja", "まだ乾いてないけどね、今朝。")])
@pytest.mark.parametrize("response,error", [
    (RuntimeError("fixture timeout"), RuntimeError),
    (HyRuntimeCancelled("fixture cancelled"), TranslationCancelled),
    ({"content": "", "stopped_eos": True}, RuntimeError),
    ({"content": "partial", "stopped_limit": True}, RuntimeError),
])
def test_failures_never_retry_self_correct_or_commit_history(production, monkeypatch,
                                                           source_language, source, response, error):
    monkeypatch.setattr(GemmaTranslationEngine, "_post_edit_once", forbid_post_edit)
    engine, runtime, constructions, _ = production([response])
    try:
        with pytest.raises(error):
            engine.translate(source, "ja_to_zh", source_language=source_language)
        assert len([1 for route, _ in runtime.calls if route == "/completion"]) == 1
        assert not any(engine._history.values())
        assert runtime.starts == len(constructions) == 1
    finally:
        engine.close()
    assert runtime.closes == 1


def test_second_slot_failure_does_not_return_partial_or_fallback_to_another_generation(production):
    engine, runtime, _, _ = production(["干燥", RuntimeError("second slot failed")])
    try:
        with pytest.raises(RuntimeError, match="second slot failed"):
            engine.translate("まだ乾いてないけどね、今朝。", "ja_to_zh", source_language="ja")
        assert len([1 for route, _ in runtime.calls if route == "/completion"]) == 2
        assert engine.last_structure_trace["status"] == "error"
        assert not any(engine._history.values())
    finally:
        engine.close()


@pytest.mark.parametrize("source_language", [None, "ja"])
@pytest.mark.parametrize("cancel_route", ["/tokenize", "/completion"])
def test_cancellation_during_request_preserves_history_and_stops_further_requests(
        production, source_language, cancel_route):
    stop = threading.Event()

    def cancel(route, _body):
        if route == cancel_route:
            stop.set()

    engine, runtime, _, _ = production(["干燥"], stop_event=stop, on_request=cancel)
    try:
        with pytest.raises(TranslationCancelled):
            engine.translate("まだ乾いてないけどね、今朝。", "ja_to_zh", source_language=source_language)
        assert len(runtime.calls) == (1 if cancel_route == "/tokenize" else 2)
        assert not any(engine._history.values())
    finally:
        engine.close()


def test_pre_cancel_does_not_construct_runtime(monkeypatch):
    stop = threading.Event()
    stop.set()
    created = []
    settings = SimpleNamespace()
    with pytest.raises(TranslationCancelled):
        GemmaTranslationEngine(settings, stop_event=stop,
                               runtime_factory=lambda *a, **k: created.append((a, k)),
                               repair_policy="source_structure_v5", prompt_transport="token_ids",
                               application_mode=True)
    assert created == []


@pytest.mark.parametrize("start_error,error", [(HyRuntimeCancelled("startup cancelled"), TranslationCancelled),
                                              (RuntimeError("startup failed"), RuntimeError)])
def test_factory_failed_start_closes_owned_single_runtime(production, start_error, error):
    with pytest.raises(error):
        production(start_error=start_error)
    assert len(production.instances) == 1
    runtime = production.instances[0]
    assert runtime.starts == runtime.closes == 1


def test_closed_factory_engine_is_idempotent_and_cannot_translate(production):
    engine, runtime, _, _ = production()
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="已关闭"):
        engine.translate("original", "ja_to_zh", source_language=None)
    assert runtime.closes == 1 and runtime.calls == []


@pytest.mark.parametrize("options", [
    {"application_mode": 1}, {"application_mode": "true"}, {"application_mode": None},
    {"application_mode": True},
    {"application_mode": True, "repair_policy": "source_structure_v4", "prompt_transport": "token_ids"},
])
def test_invalid_production_options_fail_before_runtime_creation(options):
    created = []
    with pytest.raises(ValueError):
        GemmaTranslationEngine(SimpleNamespace(), runtime_factory=lambda *a, **k: created.append((a, k)), **options)
    assert created == []


def test_direct_experimental_defaults_and_strict_v5_language_guard_are_unchanged():
    settings = SimpleNamespace(hymt_server_exe="unused", gemma_model_file="unused", translation_device="cuda")
    runtime = Runtime(["legacy output"])
    engine = GemmaTranslationEngine(settings, runtime_factory=lambda *a, **k: runtime)
    try:
        assert (engine.prompt_variant, engine.prompt_transport, engine.repair_policy, engine.application_mode) == (
            "current", "string", "none", False)
        assert engine.translate("unknown source", "ja_to_zh", source_language=None) == "legacy output"
        assert runtime.calls[1][1]["prompt"] == translation_prompt("unknown source", None, "zh")
        assert "repair_policy" not in engine.backend_info and "language_routing" not in engine.backend_info
    finally:
        engine.close()
    strict_runtime = Runtime([])
    strict = GemmaTranslationEngine(settings, runtime_factory=lambda *a, **k: strict_runtime,
                                   repair_policy="source_structure_v5", prompt_transport="token_ids")
    try:
        with pytest.raises(ValueError, match="明确的日语"):
            strict.translate("unknown source", "ja_to_zh", source_language=None)
        assert strict_runtime.calls == []
    finally:
        strict.close()
