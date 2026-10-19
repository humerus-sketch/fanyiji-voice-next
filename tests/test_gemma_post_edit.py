"""Opt-in one-pass semantic correction contracts; no model inference."""
from collections import deque
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fanyiji_next.gemma_translation import GemmaTranslationEngine, translation_prompt
from fanyiji_next.hymt_runtime import HyRuntimeCancelled
from fanyiji_next.translation_engines import TranslationCancelled


CORRECTION_INSTRUCTION = (
    "Correct the Chinese translation against the Japanese source. "
    "Make only necessary semantic corrections; otherwise keep the draft. "
    "Return only the complete corrected Chinese translation.\n"
)
PARAMETERS = {"n_predict": 256, "temperature": 0.0,
              "stop": ["<end_of_turn>"], "cache_prompt": True}


class RecordingRuntime:
    def __init__(self):
        self.calls = []
        self.token_payloads = deque()
        self.responses = deque()
        self.errors = {}
        self.after_call = None

    def start(self):
        pass

    def close(self):
        pass

    def post_json(self, route, body):
        self.calls.append((route, body))
        if len(self.calls) in self.errors:
            raise self.errors[len(self.calls)]
        if route == "/tokenize":
            result = {"tokens": self.token_payloads.popleft() if self.token_payloads else [2, 19, 105]}
        else:
            assert route == "/completion"
            result = self.responses.popleft() if self.responses else {"content": "完整译文。", "stop_type": "eos"}
        if self.after_call:
            self.after_call(len(self.calls))
        return result


@pytest.fixture
def make_engine():
    engines = []

    def create(*, policy="post_edit_once", transport="token_ids", stop_event=None):
        runtime = RecordingRuntime()
        settings = SimpleNamespace(hymt_server_exe="unused", gemma_model_file="unused",
                                   translation_device="cuda")
        engine = GemmaTranslationEngine(settings, runtime_factory=lambda *_a, **_kw: runtime,
                                        prompt_transport=transport, repair_policy=policy, stop_event=stop_event)
        engines.append(engine)
        return engine, runtime

    yield create
    for engine in engines:
        engine.close()


@pytest.mark.parametrize("policy", ["unknown", None, [], 1])
def test_unknown_repair_policy_rejected_before_runtime_creation(policy):
    factory = Mock()
    with pytest.raises(ValueError, match="校对策略"):
        GemmaTranslationEngine(SimpleNamespace(), runtime_factory=factory, repair_policy=policy)
    factory.assert_not_called()


@pytest.mark.parametrize("variant,transport", [
    ("current", "string"), ("native_single_ja_zh", "token_ids"),
    ("structured_ja_zh_v1", "token_ids"), ("faithful_ja_zh_v2", "token_ids"),
    ("role_task_ja_zh_v3", "token_ids"),
])
def test_post_edit_invalid_combinations_rejected_before_runtime_creation(variant, transport):
    factory = Mock()
    with pytest.raises(ValueError, match="current.*token_ids"):
        GemmaTranslationEngine(SimpleNamespace(), runtime_factory=factory,
                               prompt_variant=variant, prompt_transport=transport, repair_policy="post_edit_once")
    factory.assert_not_called()


@pytest.mark.parametrize("source,direction", [(None, "ja_to_zh"), ("en", "ja_to_zh"),
                                             ("zh", "ja_to_zh"), ("ja", "zh_to_ja")])
def test_post_edit_rejects_language_before_request_or_history(make_engine, source, direction):
    engine, runtime = make_engine()
    with pytest.raises(ValueError, match="仅支持明确"):
        engine.translate("source", direction, source_language=source)
    assert runtime.calls == [] and engine._history == {}


def test_none_default_keeps_single_generation_and_original_current_request():
    runtime = RecordingRuntime()
    settings = SimpleNamespace(hymt_server_exe="unused", gemma_model_file="unused", translation_device="cuda")
    engine = GemmaTranslationEngine(settings, runtime_factory=lambda *_a, **_kw: runtime)
    try:
        runtime.responses.append({"content": " 原样译文、原样译文。 \n", "stop_type": "eos"})
        assert engine.translate("  今日は晴れです。 ", "ja_to_zh", source_language="ja",
                                reference_texts=["前文"]) == "原样译文、原样译文。"
        prompt = translation_prompt("  今日は晴れです。 ", "ja", "zh", ["前文"])
        assert runtime.calls == [("/tokenize", {"content": prompt, "add_special": False}),
                                 ("/completion", {"prompt": prompt, **PARAMETERS})]
        assert engine.repair_policy == "none" and engine.prompt_variant == "current"
        assert engine.prompt_transport == "string" and "repair_policy" not in engine.backend_info
    finally:
        engine.close()


def test_explicit_none_token_transport_still_makes_only_one_generation(make_engine):
    engine, runtime = make_engine(policy="none")
    assert engine.translate("日語", "ja_to_zh", source_language="ja") == "完整译文。"
    assert [route for route, _body in runtime.calls] == ["/tokenize", "/completion"]
    assert engine.backend_info["prompt_transport"] == "token_ids"


def test_two_stages_exact_prompts_ids_parameters_and_history_append_once(make_engine):
    engine, runtime = make_engine()
    first_ids, second_ids = [2, 105, 11, 106], [2, 105, 17, 29, 106]
    runtime.token_payloads.extend([first_ids, second_ids])
    draft = "  今天三点在车站见面。\n "
    corrected = " 今天下午三点在车站见面。 \n"
    runtime.responses.extend([{"content": draft, "stop_type": "eos"},
                              {"content": corrected, "stop_type": "eos"}])
    source = '  今日は午後三時に"駅"で会います。\n '
    refs = ["除去する古い文", '時刻は"午後"です。\n前半', "場所を教えてください。"]
    assert engine.translate(source, "ja_to_zh", source_language="ja", reference_texts=refs) == corrected.strip()
    expected_first = translation_prompt(source, "ja", "zh", refs[-2:])
    expected_second = (
        "<bos><start_of_turn>user\n" + CORRECTION_INSTRUCTION
        + '{"reference_only": ["時刻は\\\"午後\\\"です。\\n前半", "場所を教えてください。"], '
        '"source": "  今日は午後三時に\\\"駅\\\"で会います。\\n ", '
        '"draft": "  今天三点在车站见面。\\n "}'
        + "<end_of_turn>\n<start_of_turn>model\n"
    )
    assert runtime.calls == [
        ("/tokenize", {"content": expected_first, "add_special": False, "parse_special": True}),
        ("/completion", {"prompt": first_ids, **PARAMETERS}),
        ("/tokenize", {"content": expected_second, "add_special": False, "parse_special": True}),
        ("/completion", {"prompt": second_ids, **PARAMETERS}),
    ]
    assert runtime.calls[1][1]["prompt"] is first_ids and runtime.calls[3][1]["prompt"] is second_ids
    assert first_ids.count(2) == second_ids.count(2) == 1
    assert list(engine._history[("default", "ja", "zh")]) == [source]
    assert engine.backend_info["repair_policy"] == "post_edit_once"


@pytest.mark.parametrize("source,refs,draft,answer", [
    ("帰らない、帰らない。", ["前文", "帰らない、帰らない。"], "不回去、不回去。", "不回去、不回去。"),
    ("識別子 control_04 は七番です。", [], "control_04是七号。", "control_04是七号。"),
    ("仮の原文\n引用\"記号\"", ["一つ前"], "\n 完整中文草稿、重复、重复。 ", "校正版、重复、重复。"),
])
def test_post_edit_generic_data_only_no_answer_reference_filter_or_case_branch(make_engine, source, refs, draft, answer):
    engine, runtime = make_engine()
    runtime.responses.extend([{"content": draft, "stop_type": "eos"}, {"content": answer, "stop_type": "eos"}])
    assert engine.translate(source, "ja_to_zh", source_language="ja", reference_texts=refs) == answer
    correction = runtime.calls[2][1]["content"]
    prefix = "<bos><start_of_turn>user\n" + CORRECTION_INSTRUCTION
    assert correction.startswith(prefix)
    assert json.loads(correction[len(prefix):-len("<end_of_turn>\n<start_of_turn>model\n")]) == {
        "reference_only": refs, "source": source, "draft": draft}
    assert len(runtime.calls) == 4


@pytest.mark.parametrize("response", [
    {"content": "译文", "stopped_limit": True}, {"content": "译文", "truncated": True},
    {"content": "译文", "stop_type": "limit"}, {"content": "译文", "stop_type": "length"},
    {"content": "译文", "stop_type": "max_tokens"}, {"content": "译文", "tokens_predicted": 256},
    {"content": ""}, {"content": " \n "}, {"content": None}, {"content": 123}, {},
])
def test_second_stage_failure_never_returns_draft_retries_or_appends_history(make_engine, response):
    engine, runtime = make_engine()
    runtime.responses.extend([{"content": "有效的完整草稿", "stop_type": "eos"}, response])
    with pytest.raises(RuntimeError, match="原文已保留"):
        engine.translate("明日は行きます。", "ja_to_zh", source_language="ja", reference_texts=["前文"])
    assert [route for route, _body in runtime.calls] == ["/tokenize", "/completion", "/tokenize", "/completion"]
    assert list(engine._history[("default", "ja", "zh")]) == []


def test_first_stage_failure_does_not_start_correction(make_engine):
    engine, runtime = make_engine()
    runtime.responses.append({"content": "未完成", "truncated": True})
    with pytest.raises(RuntimeError, match="原文已保留"):
        engine.translate("原文。", "ja_to_zh", source_language="ja")
    assert [route for route, _body in runtime.calls] == ["/tokenize", "/completion"]
    assert list(engine._history[("default", "ja", "zh")]) == []


@pytest.mark.parametrize("second_ids", [None, {}, "2,17", [2, True], [2, "17"], [-1], []])
def test_second_stage_invalid_ids_stop_without_generation_or_history(make_engine, second_ids):
    engine, runtime = make_engine()
    runtime.token_payloads.extend([[2, 11], second_ids])
    with pytest.raises(RuntimeError, match="token"):
        engine.translate("原文。", "ja_to_zh", source_language="ja")
    assert len(runtime.calls) == 3
    assert list(engine._history[("default", "ja", "zh")]) == []


def test_second_stage_budget_failure_keeps_all_data_without_trimming_retry(make_engine):
    engine, runtime = make_engine()
    runtime.token_payloads.extend([[2, 11], [2] + [11] * 1750])
    runtime.responses.append({"content": "完整草稿" * 20, "stop_type": "eos"})
    source, refs = "当前原文。" * 10, ["前第二句。", "前一句。"]
    with pytest.raises(RuntimeError, match="校对输入超过"):
        engine.translate(source, "ja_to_zh", source_language="ja", reference_texts=refs)
    assert len(runtime.calls) == 3
    assert json.dumps(source, ensure_ascii=False) in runtime.calls[2][1]["content"]
    assert json.dumps(refs, ensure_ascii=False) in runtime.calls[2][1]["content"]
    assert "完整草稿" * 20 in runtime.calls[2][1]["content"]
    assert list(engine._history[("default", "ja", "zh")]) == []


@pytest.mark.parametrize("cancel_after", [2, 3, 4])
def test_cancellation_never_returns_draft_or_remembers_source(make_engine, cancel_after):
    stop = threading.Event()
    engine, runtime = make_engine(stop_event=stop)
    runtime.after_call = lambda count: stop.set() if count == cancel_after else None
    with pytest.raises(TranslationCancelled):
        engine.translate("原文。", "ja_to_zh", source_language="ja")
    assert len(runtime.calls) == cancel_after
    assert list(engine._history[("default", "ja", "zh")]) == []


@pytest.mark.parametrize("error_at", [3, 4])
def test_runtime_cancellation_in_second_stage_preserves_cancellation_type(make_engine, error_at):
    engine, runtime = make_engine()
    runtime.errors[error_at] = HyRuntimeCancelled("取消")
    with pytest.raises(TranslationCancelled):
        engine.translate("原文。", "ja_to_zh", source_language="ja")
    assert len(runtime.calls) == error_at
    assert list(engine._history[("default", "ja", "zh")]) == []


def test_second_stage_error_does_not_replace_existing_successful_history(make_engine):
    engine, runtime = make_engine()
    engine.translate("前の成功文。", "ja_to_zh", source_language="ja")
    runtime.responses.extend([{"content": "新草稿", "stop_type": "eos"}, {"content": "", "stop_type": "eos"}])
    with pytest.raises(RuntimeError, match="原文已保留"):
        engine.translate("失敗した文。", "ja_to_zh", source_language="ja")
    assert list(engine._history[("default", "ja", "zh")]) == ["前の成功文。"]
    assert '"reference_only": ["前の成功文。"]' in runtime.calls[6][1]["content"]


@pytest.mark.parametrize("complete_flag", ["stopped_eos", "stopped_word"])
def test_second_stage_keeps_existing_complete_stop_limit_semantics(make_engine, complete_flag):
    engine, runtime = make_engine()
    runtime.responses.extend([{"content": "草稿", "stop_type": "eos"},
                              {"content": "完整校正版", "tokens_predicted": 256, complete_flag: True}])
    assert engine.translate("原文。", "ja_to_zh", source_language="ja") == "完整校正版"
    assert len(runtime.calls) == 4
