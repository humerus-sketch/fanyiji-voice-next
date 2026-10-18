"""Contract tests for local A/B adapters; no Torch or model weights required."""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fanyiji_next.ab_recognition import ABRecognitionEngine, MAX_NEW_TOKENS, normalize_language
from fanyiji_next.gemma_translation import GemmaTranslationEngine, translation_prompt
from fanyiji_next.hymt_runtime import HyRuntimeCancelled
from fanyiji_next.recognition_engines import RecognitionCancelled
from fanyiji_next.translation_engines import TranslationCancelled


@pytest.fixture
def recognition(monkeypatch, tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    for relative in ("model.pt", "config.yaml", "multilingual.tiktoken", "Qwen3-0.6B/config.json",
                     "Qwen3-0.6B/tokenizer.json", "Qwen3-0.6B/tokenizer_config.json"):
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    state = SimpleNamespace(folder=folder, generated=2, eos=False,
                            parsed=[{"transcription": "  原始字幕  ", "language": "Japanese"}],
                            fun_result=[{"text": "  原始字幕  "}], on_generate=None,
                            cuda_available=True, fun_steps=2)
    torch = ModuleType("torch")
    torch.float32, torch.bfloat16 = "float32", "bfloat16"
    torch.inference_mode = nullcontext
    torch.from_numpy = Mock(side_effect=lambda array: SimpleNamespace(array=array, device="cpu", dtype="float32"))
    torch.cuda = SimpleNamespace(is_available=lambda: state.cuda_available, empty_cache=Mock())
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Inputs(dict):
        def to(self, *args):
            state.input_to = args
            return self

    class Processor:
        def apply_transcription_request(self, **kwargs):
            state.processor_audio = kwargs
            return Inputs(input_ids=np.array([[8, 9]]))

        def decode(self, ids, **kwargs):
            state.decode = (ids, kwargs)
            return state.parsed

    class Model:
        device, dtype = "cuda:0", "bfloat16"
        config = SimpleNamespace(eos_token_id=99)
        generation_config = SimpleNamespace(eos_token_id=[99, 100])

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            state.qwen_generate = kwargs
            if state.on_generate:
                state.on_generate(kwargs["stopping_criteria"])
            tokens = [4] * state.generated
            if state.eos and tokens:
                tokens[-1] = 99
            return np.array([[8, 9] + tokens])

    processor, model = Processor(), Model()
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
    transformers.AutoModelForMultimodalLM = SimpleNamespace(from_pretrained=Mock(return_value=model))
    transformers.StoppingCriteria = type("StoppingCriteria", (), {})
    transformers.StoppingCriteriaList = list
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class FunModel:
        model = SimpleNamespace(llm=SimpleNamespace(config=SimpleNamespace(eos_token_id=99)))

        def generate(self, **kwargs):
            state.fun_generate = kwargs
            criteria = kwargs["llm_kwargs"]["stopping_criteria"]
            for step in range(state.fun_steps):
                last = 99 if state.eos and step == state.fun_steps-1 else 4
                criteria[0](np.array([[last]]), None)
            if state.on_generate:
                state.on_generate(criteria)
            return state.fun_result

    funasr = ModuleType("funasr")
    funasr.AutoModel = Mock(return_value=FunModel())
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    state.settings = SimpleNamespace(recognition_backend="qwen3_asr", recognition_device="cuda",
                                     qwen_asr_model_dir=str(folder), fun_asr_model_dir=str(folder))
    state.torch, state.transformers, state.funasr = torch, transformers, funasr
    yield state


def test_qwen_native_processor_and_parse_contract_stays_local(recognition):
    state = recognition
    engine = ABRecognitionEngine(state.settings)
    try:
        result = engine.recognize_result([.1, -.2])
        assert (result.text, result.language) == ("原始字幕", "ja")
        assert state.transformers.AutoProcessor.from_pretrained.call_args.kwargs == {
            "local_files_only": True, "trust_remote_code": False}
        kwargs = state.transformers.AutoModelForMultimodalLM.from_pretrained.call_args.kwargs
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        assert kwargs["dtype"] == "bfloat16"
        assert state.processor_audio.keys() == {"audio"}
        assert state.processor_audio["audio"].dtype == np.float32
        assert state.input_to == ("cuda", "bfloat16")
        assert state.decode[0].shape == (1, 2)
        assert state.decode[1] == {"return_format": "parsed"}
        assert state.qwen_generate["do_sample"] is False
    finally:
        engine.close()


def test_fun_uses_same_repository_local_paths_and_automatic_language(recognition):
    state = recognition
    state.settings.recognition_backend = "fun_asr"
    state.settings.recognition_device = "cpu"
    engine = ABRecognitionEngine(state.settings)
    try:
        result = engine.recognize_result([.2])
        assert (result.text, result.language) == ("原始字幕", None)
        kwargs = state.funasr.AutoModel.call_args.kwargs
        assert kwargs["model"] == str(state.folder)
        assert kwargs["init_param"] == str(state.folder / "model.pt")
        assert kwargs["llm_conf"]["init_param_path"] == str(state.folder / "Qwen3-0.6B")
        assert kwargs["tokenizer_conf"]["init_param_path"] == kwargs["llm_conf"]["init_param_path"]
        assert kwargs["ctc_tokenizer_conf"]["vocab_path"] == str(state.folder / "multilingual.tiktoken")
        assert kwargs["llm_conf"]["llm_dtype"] == "fp32"
        assert kwargs["trust_remote_code"] is False and kwargs["check_latest"] is False
        assert kwargs["local_files_only"] is True and kwargs["disable_update"] is True
        assert state.fun_generate["language"] is None
        assert "fs" not in state.fun_generate
        assert state.fun_generate["llm_kwargs"]["stopping_criteria"]
        state.torch.from_numpy.assert_called_once()
        assert state.fun_generate["input"].device == "cpu"
        assert state.fun_generate["input"].dtype == "float32"
        assert state.fun_generate["input"].array.tolist() == pytest.approx([.2])
    finally:
        engine.close()


def test_fun_missing_local_config_fails_before_model_factory(recognition):
    recognition.settings.recognition_backend = "fun_asr"
    (recognition.folder / "Qwen3-0.6B/config.json").unlink()
    with pytest.raises(RuntimeError, match="不完整"):
        ABRecognitionEngine(recognition.settings)
    recognition.funasr.AutoModel.assert_not_called()


def test_empty_model_setting_never_uses_working_directory(recognition):
    recognition.settings.qwen_asr_model_dir = ""
    with pytest.raises(RuntimeError, match="尚未选择"):
        ABRecognitionEngine(recognition.settings)
    recognition.transformers.AutoProcessor.from_pretrained.assert_not_called()


@pytest.mark.parametrize("backend", ["qwen3_asr", "fun_asr"])
def test_token_cancellation_discards_result_and_is_forwarded(recognition, backend):
    state = recognition
    state.settings.recognition_backend = backend
    stop = threading.Event()
    engine = ABRecognitionEngine(state.settings, stop_event=stop)

    def cancel(criteria):
        stop.set()
        assert criteria[0](np.array([[1]]), None) is True

    state.on_generate = cancel
    try:
        with pytest.raises(RecognitionCancelled):
            engine.recognize_result([.1])
    finally:
        engine.close()


@pytest.mark.parametrize("backend", ["qwen3_asr", "fun_asr"])
def test_token_limit_is_error_unless_last_token_is_eos(recognition, backend):
    state = recognition
    state.settings.recognition_backend = backend
    state.generated = state.fun_steps = MAX_NEW_TOKENS
    engine = ABRecognitionEngine(state.settings)
    try:
        with pytest.raises(RuntimeError, match="长度上限"):
            engine.recognize_result([.1])
        state.eos = True
        assert engine.recognize_result([.1]).text == "原始字幕"
    finally:
        engine.close()


@pytest.mark.parametrize("samples", [[[.1]], [float("nan")], [float("inf")]])
def test_invalid_waveforms_do_not_reach_generation(recognition, samples):
    engine = ABRecognitionEngine(recognition.settings)
    try:
        with pytest.raises(ValueError):
            engine.recognize_result(samples)
        assert not hasattr(recognition, "qwen_generate")
    finally:
        engine.close()


def test_recognition_empty_and_closed_boundaries(recognition):
    engine = ABRecognitionEngine(recognition.settings)
    assert engine.recognize_result([]).text == ""
    assert not hasattr(recognition, "qwen_generate")
    engine.close()
    with pytest.raises(RuntimeError, match="关闭"):
        engine.recognize_result([.1])


def test_precancelled_load_and_unavailable_cuda_do_not_load_models(recognition):
    stop = threading.Event()
    stop.set()
    with pytest.raises(RecognitionCancelled):
        ABRecognitionEngine(recognition.settings, stop_event=stop)
    recognition.transformers.AutoProcessor.from_pretrained.assert_not_called()
    recognition.cuda_available = False
    with pytest.raises(RuntimeError, match="CUDA"):
        ABRecognitionEngine(recognition.settings)
    recognition.transformers.AutoModelForMultimodalLM.from_pretrained.assert_not_called()


@pytest.mark.parametrize("payload", [None, {}, ["invalid"], [{"transcription": 7}]])
def test_malformed_qwen_output_is_not_invented_or_stringified(recognition, payload):
    recognition.parsed = payload
    engine = ABRecognitionEngine(recognition.settings)
    try:
        with pytest.raises(RuntimeError):
            engine.recognize_result([.1])
    finally:
        engine.close()


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), ("auto", None),
    ("Japanese", "ja"), ("Chinese", "zh"), ("zh-Hant", "zh"), ("en_US", "en")])
def test_language_metadata_is_normalized_without_assuming_japanese(value, expected):
    assert normalize_language(value) == expected


class FakeRuntime:
    def __init__(self, *_args, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.responses = deque()
        self.token_counts = deque()
        self.starts = self.closes = 0
        self.on_request = None
        self.start_error = None

    def start(self):
        self.starts += 1
        if self.start_error:
            raise self.start_error

    def close(self):
        self.closes += 1

    def post_json(self, route, body):
        self.calls.append((route, body))
        if self.on_request:
            self.on_request(route)
        if route == "/tokenize":
            return {"tokens": [1] * (self.token_counts.popleft() if self.token_counts else 20)}
        assert route == "/completion"
        value = self.responses.popleft() if self.responses else {"content": "译文", "stopped_eos": True}
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.fixture
def gemma():
    settings = SimpleNamespace(hymt_server_exe="fake/llama-server.exe", gemma_model_file="fake/gemma.gguf",
                               translation_device="cuda")
    runtime = FakeRuntime()
    engine = GemmaTranslationEngine(settings, runtime_factory=lambda *_args, **_kwargs: runtime)
    yield engine, runtime, settings
    engine.close()


def prompts(runtime):
    return [body["prompt"] for route, body in runtime.calls if route == "/completion"]


def test_gemma_native_template_has_separate_source_context_and_current_text():
    value = translation_prompt("本句未完…", "ja", "zh", ["很旧一句", "前两句", "前一句"])
    assert value.startswith("<bos><start_of_turn>user\nYou are a professional Japanese (ja) to Chinese (zh) translator.")
    assert "nuances of the original Japanese text" in value
    assert 'PREVIOUS SOURCE TEXT (reference only; do not translate or repeat):\n["前两句", "前一句"]' in value
    assert "很旧一句" not in value
    assert value.count("本句未完…") == 1
    assert value.endswith("本句未完…<end_of_turn>\n<start_of_turn>model\n")


def test_unknown_language_is_never_forced_to_japanese_and_same_language_is_passed_through(gemma):
    engine, runtime, _settings = gemma
    assert engine.translate("原句", "ja_to_zh", source_language=None) == "译文"
    assert "professional original-language to Chinese (zh)" in prompts(runtime)[0]
    assert "Japanese" not in prompts(runtime)[0]
    count = len(runtime.calls)
    assert engine.translate(" 中文原句 ", "ja_to_zh", source_language="zh") == " 中文原句 "
    assert len(runtime.calls) == count


def test_context_is_last_two_sources_and_isolated_by_source_language_and_target(gemma):
    engine, runtime, _settings = gemma
    for text in ("s1", "s2", "s3", "s4"):
        engine.translate(text, "ja_to_zh", context_id="audio1", source_language="ja")
    assert '["s2", "s3"]' in prompts(runtime)[-1]
    assert '"s1"' not in prompts(runtime)[-1]
    assert "译文" not in prompts(runtime)[-1]
    engine.translate("new source", "ja_to_zh", context_id="audio2", source_language="ja")
    assert "PREVIOUS SOURCE TEXT" not in prompts(runtime)[-1]
    engine.translate("english switch", "ja_to_zh", context_id="audio1", source_language="en")
    assert "PREVIOUS SOURCE TEXT" not in prompts(runtime)[-1]
    engine.translate("target switch", "zh_to_ja", context_id="audio1", source_language="en")
    assert "PREVIOUS SOURCE TEXT" not in prompts(runtime)[-1]
    assert all(len(history) <= 2 for history in engine._history.values())


@pytest.mark.parametrize("response", [{"content": "可能残句", "stopped_limit": True},
    {"content": "可能残句", "truncated": True}, {"content": "可能残句", "tokens_predicted": 256},
    {"content": "可能残句", "stop_type": "length"}, {"content": " "}, {"content": 7}])
def test_partial_or_invalid_generation_is_error_and_never_enters_history(gemma, response):
    engine, runtime, _settings = gemma
    runtime.responses.append(response)
    with pytest.raises(RuntimeError):
        engine.translate("source", "ja_to_zh", source_language="ja")
    assert not any(engine._history.values())


def test_exact_output_budget_with_eos_is_complete(gemma):
    engine, runtime, _settings = gemma
    runtime.responses.append({"content": "完整译文", "tokens_predicted": 256, "stopped_eos": True})
    assert engine.translate("source", "ja_to_zh", source_language="ja") == "完整译文"


def test_overlong_reference_is_removed_but_current_text_is_never_truncated(gemma):
    engine, runtime, _settings = gemma
    engine.translate("old context", "ja_to_zh", source_language="ja")
    runtime.token_counts.extend([1751, 20])
    current = "当前句" * 30
    engine.translate(current, "ja_to_zh", source_language="ja")
    assert "PREVIOUS SOURCE TEXT" not in prompts(runtime)[-1]
    assert current in prompts(runtime)[-1]
    runtime.token_counts.extend([1751, 1751])
    count = len(prompts(runtime))
    with pytest.raises(RuntimeError, match="上下文长度"):
        engine.translate("too long", "ja_to_zh", source_language="ja")
    assert len(prompts(runtime)) == count
    assert "too long" not in list(engine._history[("default", "ja", "zh")])


def test_runtime_cancellation_maps_to_session_exception_and_does_not_add_context(gemma):
    engine, runtime, _settings = gemma
    runtime.responses.append(HyRuntimeCancelled("cancelled"))
    with pytest.raises(TranslationCancelled):
        engine.translate("source", "ja_to_zh", source_language="ja")
    assert not any(engine._history.values())


def test_stop_after_tokenization_prevents_completion_and_late_stop_discards_output(gemma):
    engine, runtime, _settings = gemma
    stop = threading.Event()
    engine._stop_event = stop
    runtime.on_request = lambda _route: stop.set()
    with pytest.raises(TranslationCancelled):
        engine.translate("source", "ja_to_zh", source_language="ja")
    assert not prompts(runtime)
    stop.clear()
    runtime.on_request = lambda route: stop.set() if route == "/completion" else None
    with pytest.raises(TranslationCancelled):
        engine.translate("late source", "ja_to_zh", source_language="ja")
    assert not any(engine._history.values())


def test_cancelled_start_is_closed_and_uses_translation_cancelled(gemma):
    _engine, _runtime, settings = gemma
    runtime = FakeRuntime()
    runtime.start_error = HyRuntimeCancelled("cancelled start")
    with pytest.raises(TranslationCancelled):
        GemmaTranslationEngine(settings, runtime_factory=lambda *_a, **_k: runtime)
    assert runtime.closes == 1


def test_pre_cancelled_translation_does_not_construct_runtime(gemma):
    _engine, _runtime, settings = gemma
    stop = threading.Event()
    stop.set()
    factory = Mock()
    with pytest.raises(TranslationCancelled):
        GemmaTranslationEngine(settings, stop_event=stop, runtime_factory=factory)
    factory.assert_not_called()


def test_closed_engine_and_invalid_direction_fail_without_model_call(gemma):
    engine, runtime, _settings = gemma
    with pytest.raises(ValueError):
        engine.translate("source", "invalid")
    assert not runtime.calls
    engine.close()
    engine.close()
    assert runtime.closes == 1
    with pytest.raises(RuntimeError, match="关闭"):
        engine.translate("source", "ja_to_zh")
