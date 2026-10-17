"""Hy adapter contracts with fake transport and guard, without native engines."""

from collections import deque
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import fanyiji_next
from fanyiji_next import translation_engines
from fanyiji_next.hymt_runtime import HyRuntimeCancelled, HyRuntimeError


@pytest.fixture
def fake(monkeypatch):
    # Load an isolated adapter module: the guard is developed independently and
    # these tests must neither rely on its rules nor replace it for other tests.
    guard = ModuleType("fanyiji_next.translation_guard")
    guard.prepare_source = Mock(side_effect=lambda text: text)
    guard.candidate_issues = Mock(return_value=[])
    guard.context_grounding_issues = Mock(return_value=())
    guard.source_fragment_issue = Mock(return_value=None)
    name = "fanyiji_next.hymt_translation"
    path = Path(translation_engines.__file__).with_name("hymt_translation.py")
    spec = importlib.util.spec_from_file_location(name, path)
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, guard.__name__, guard)
    monkeypatch.setitem(sys.modules, name, adapter)
    monkeypatch.setattr(fanyiji_next, "translation_guard", guard, raising=False)
    monkeypatch.setattr(fanyiji_next, "hymt_translation", adapter, raising=False)
    spec.loader.exec_module(adapter)
    state = SimpleNamespace(module=adapter, instances=[], start_error=None, close_errors=deque(), engines=[])

    class FakeRuntime:
        def __init__(self, executable, model_path, **kwargs):
            self.executable, self.model_path, self.kwargs = executable, model_path, kwargs
            self.backend_info = {"runtime": "fake", "device": kwargs["device"]}
            self.starts = 0
            self.closes = 0
            self.rendered = []
            self.generations = []
            self.stages = []
            self.outputs = deque()
            self.token_count = lambda messages: 20
            state.instances.append(self)

        def start(self):
            self.starts += 1
            if state.start_error is not None:
                raise state.start_error

        def close(self):
            self.closes += 1
            if state.close_errors:
                raise state.close_errors.popleft()

        def post_json(self, route, body):
            self.stages.append(route)
            if route == "/apply-template":
                assert body["add_generation_prompt"] is True
                self.rendered.append(deepcopy(body["messages"]))
                return {"prompt": f"rendered:{len(self.rendered)-1}"}
            if route == "/tokenize":
                assert body["add_special"] is True and body["parse_special"] is True
                index = int(body["content"].split(":")[-1])
                return {"tokens": [1] * self.token_count(self.rendered[index])}
            raise AssertionError(f"Unexpected local API: {route}")

        def iter_completion(self, body):
            self.stages.append("/completion")
            index = int(body["prompt"].split(":")[-1])
            self.generations.append({"messages": deepcopy(self.rendered[index]), "body": deepcopy(body)})
            planned = self.outputs.popleft() if self.outputs else f"target-{len(self.generations)}"
            if isinstance(planned, BaseException):
                raise planned
            if isinstance(planned, list):
                yield from planned
                return
            settings = planned if isinstance(planned, dict) else {"output": planned}
            output = settings.get("output", "")
            messages = self.rendered[index]
            if (len(messages) == 1 and messages[0]["content"].startswith(adapter.PARAGRAPH_RULE)
                    and not settings.get("raw_paragraph_output")):
                content = messages[0]["content"][len(adapter.PARAGRAPH_RULE):]
                delimiter = next(item for item in adapter.PARAGRAPH_DELIMITERS if "\n" + item + "\n" in content)
                count = content.count(delimiter)
                output = ("\n" + delimiter + "\n").join(["previous-target"] * count + [output])
            yield {"content": output, "stop": False}
            if not settings.get("no_final"):
                yield {"content": "", "stop": True,
                       "stop_type": settings.get("stop_type", "eos"),
                       "truncated": settings.get("truncated", False), "timings": {"cache_n": 0}}

    monkeypatch.setattr(adapter, "LlamaServerRuntime", FakeRuntime)
    settings = SimpleNamespace(hymt_server_exe="selected-server.exe", hymt_model_file="selected-q8.gguf",
                               translation_backend="hymt")
    state.settings = settings

    def make(*, stop_event=None, factory=False):
        engine = (translation_engines.create_translation_engine(settings, stop_event=stop_event) if factory
                  else adapter.HyTranslationEngine(settings, stop_event=stop_event))
        state.engines.append(engine)
        return engine

    state.make = make
    yield state
    for engine in state.engines:
        engine.close()
    # Constructor failures may have no engine reference, but fake resources can
    # still be retired independently without interacting with a real process.
    for runtime in state.instances:
        runtime.close()


def only_user(messages):
    users = [message["content"] for message in messages if message["role"] == "user"]
    assert len(users) == 1
    return users[0]


def system_text(messages):
    return next(message["content"] for message in messages if message["role"] == "system")


def test_loads_selected_runtime_explicitly_and_keeps_q8_metadata(fake):
    stop = threading.Event()
    engine = fake.make(stop_event=stop)
    runtime = engine.runtime
    assert runtime.starts == 1
    assert runtime.executable == fake.settings.hymt_server_exe
    assert runtime.model_path == fake.settings.hymt_model_file
    assert runtime.kwargs == {"device": "cuda", "stop_event": stop, "ctx_size": 2048}
    assert engine.backend_info["backend"] == "hymt"
    assert engine.backend_info["quantization"] == "Q8_0"
    assert engine.backend_info["local_files_only"] is True


def test_first_sentence_is_one_context_free_draft(fake):
    engine = fake.make()
    engine.runtime.outputs.append("当前句译文")
    assert engine.translate("当前源文", "zh_to_ja", context_id="mic") == "当前句译文"
    assert len(engine.runtime.generations) == 1
    messages = engine.runtime.generations[0]["messages"]
    assert messages == [{"role": "user", "content": only_user(messages)}]
    assert only_user(messages).endswith("当前源文")
    assert "日语" in only_user(messages)
    assert engine.last_diagnostics["history_available"] == 0
    assert engine.last_diagnostics["history_used"] == 0


def test_source_context_contains_only_last_two_utterances_not_previous_targets(fake):
    engine = fake.make()
    engine.runtime.outputs.extend(["target-one", "draft-two", "target-two", "draft-three", "target-three",
                                   "draft-four", "target-four"])
    for source in ("source-one", "source-two", "source-three", "source-four"):
        engine.translate(source, "zh_to_ja", context_id="mic")
    last = engine.runtime.generations[-1]["messages"]
    context = system_text(last)
    assert "source-two\nsource-three" in context
    assert "source-one" not in context
    assert "target-one" not in context and "target-two" not in context and "target-three" not in context
    assert "draft-four" in context
    assert only_user(last).endswith("source-four")
    assert "source-four" not in context
    assert engine.last_diagnostics["history_available"] == 2


def test_channels_and_directions_have_independent_source_histories(fake):
    engine = fake.make()
    engine.translate("mic-zh-old", "zh_to_ja", context_id="mic")
    engine.translate("system-zh-old", "zh_to_ja", context_id="system")
    engine.translate("mic-ja-old", "ja_to_zh", context_id="mic")
    assert all(len(call["messages"]) == 1 for call in engine.runtime.generations)
    engine.translate("mic-zh-current", "zh_to_ja", context_id="mic")
    last = engine.runtime.generations[-1]["messages"]
    assert "mic-zh-old" in system_text(last)
    assert "system-zh-old" not in system_text(last) and "mic-ja-old" not in system_text(last)
    engine.translate("system-zh-current", "zh_to_ja", context_id="system")
    assert "system-zh-old" in system_text(engine.runtime.generations[-1]["messages"])
    assert "mic-zh-current" not in system_text(engine.runtime.generations[-1]["messages"])


def test_current_draft_precedes_context_pass_and_current_source_stays_in_user(fake):
    engine = fake.make()
    engine.translate("previous-source", "zh_to_ja", context_id="mic")
    engine.runtime.outputs.extend(["current-independent-draft", "current-context-result"])
    before = len(engine.runtime.generations)
    stage_before = len(engine.runtime.stages)
    assert engine.translate("current-source", "zh_to_ja", context_id="mic") == "current-context-result"
    draft, contextual = engine.runtime.generations[before:]
    assert len(draft["messages"]) == 1
    assert only_user(draft["messages"]) == only_user(contextual["messages"])
    assert only_user(contextual["messages"]).endswith("current-source")
    assert "current-source" not in system_text(contextual["messages"])
    assert "current-independent-draft" in system_text(contextual["messages"])
    assert "日语" in only_user(contextual["messages"])
    assert engine.runtime.stages[stage_before:] == ["/apply-template", "/tokenize", "/completion"] * 2
    for call in (draft, contextual):
        assert call["body"]["cache_prompt"] is False
        assert call["body"]["n_predict"] == 256
        assert call["body"]["temperature"] == 0


def test_source_preparation_applies_to_current_and_source_history(fake):
    engine = fake.make()
    fake.module.prepare_source.side_effect = lambda text: f"NORMALIZED<{text}>"
    engine.translate("previous", "zh_to_ja")
    engine.translate("current", "zh_to_ja")
    last = engine.runtime.generations[-1]["messages"]
    assert "NORMALIZED<previous>" in system_text(last)
    assert only_user(last).endswith("NORMALIZED<current>")
    assert engine.last_diagnostics["source_normalized"] is True
    assert fake.module.candidate_issues.call_args.args[0] == "NORMALIZED<current>"


def test_bad_context_returns_same_q8_safe_current_draft(fake):
    engine = fake.make()
    engine.translate("prior-source", "zh_to_ja")
    engine.runtime.outputs.extend(["safe-current-draft", "unsafe-context"])
    fake.module.candidate_issues.side_effect = lambda source, output, direction: ["negation"] if output == "unsafe-context" else []
    assert engine.translate("do-not-pay", "zh_to_ja") == "safe-current-draft"
    assert len(fake.instances) == 1
    assert engine.last_diagnostics["fallback"] == "current_sentence_draft"
    assert engine.last_diagnostics["issues"] == ["negation"]
    assert engine.last_diagnostics["calls"] == 2


def test_two_bad_candidates_raise_but_keep_recognized_source_for_next_context(fake):
    engine = fake.make()
    engine.translate("prior-source", "zh_to_ja")
    engine.runtime.outputs.extend(["unsafe-draft", "unsafe-context"])
    fake.module.candidate_issues.side_effect = lambda source, output, direction: ["numbers"] if output.startswith("unsafe-") else []
    with pytest.raises(RuntimeError, match="保留原文"):
        engine.translate("recognized-source-failed-translation", "zh_to_ja")
    engine.runtime.outputs.extend(["next-safe-draft", "next-safe-context"])
    engine.translate("next-source", "zh_to_ja")
    context = system_text(engine.runtime.generations[-1]["messages"])
    assert "recognized-source-failed-translation" in context
    assert "unsafe-draft" not in context and "unsafe-context" not in context


def test_bad_first_draft_raises_without_faking_context(fake):
    engine = fake.make()
    fake.module.candidate_issues.return_value = ["negation"]
    with pytest.raises(RuntimeError, match="保留原文"):
        engine.translate("first-source", "zh_to_ja")
    assert len(engine.runtime.generations) == 1
    assert len(engine.runtime.generations[0]["messages"]) == 1


def test_cancellation_after_draft_does_not_begin_context_request(fake):
    stop = threading.Event()
    engine = fake.make(stop_event=stop)
    engine.translate("prior-source", "zh_to_ja")
    engine.runtime.outputs.append("draft-before-stop")

    def check(source, translated, direction):
        stop.set()
        return []

    fake.module.candidate_issues.side_effect = check
    before = len(engine.runtime.generations)
    with pytest.raises(translation_engines.TranslationCancelled):
        engine.translate("current-cancelled-source", "zh_to_ja")
    assert len(engine.runtime.generations) == before + 1
    assert len(engine.runtime.generations[-1]["messages"]) == 1


def test_runtime_cancellation_converts_to_translation_contract(fake):
    engine = fake.make()
    engine.runtime.outputs.append(HyRuntimeCancelled("fixture cancelled"))
    with pytest.raises(translation_engines.TranslationCancelled) as error:
        engine.translate("current-source", "ja_to_zh")
    assert isinstance(error.value.__cause__, HyRuntimeCancelled)


def test_stop_already_set_does_not_construct_runtime(fake):
    stop = threading.Event()
    stop.set()
    with pytest.raises(translation_engines.TranslationCancelled):
        fake.make(stop_event=stop)
    assert fake.instances == []


def test_context_budget_drops_whole_oldest_utterance_and_preserves_current(fake):
    engine = fake.make()
    engine.translate("older-complete-source", "zh_to_ja")
    engine.translate("newer-complete-source", "zh_to_ja")
    runtime = engine.runtime
    runtime.outputs.extend(["safe-current-draft", "safe-context-result"])
    runtime.token_count = lambda messages: fake.module.MAX_INPUT + 1 if len(messages) == 2 and "older-complete-source" in system_text(messages) else 20
    before_render, before_generate = len(runtime.rendered), len(runtime.generations)
    current = "current-complete-source-unchanged"
    assert engine.translate(current, "zh_to_ja") == "safe-context-result"
    draft, too_long, trimmed = runtime.rendered[before_render:]
    assert "older-complete-source\nnewer-complete-source" in system_text(too_long)
    assert "older-complete-source" not in system_text(trimmed)
    assert "newer-complete-source" in system_text(trimmed)
    assert all(only_user(messages).endswith(current) for messages in (draft, too_long, trimmed))
    assert len(runtime.generations) == before_generate + 2
    assert engine.last_diagnostics["history_used"] == 1


def test_when_all_history_exceeds_budget_returns_safe_current_draft(fake):
    engine = fake.make()
    engine.translate("first-prior-source", "zh_to_ja")
    engine.translate("second-prior-source", "zh_to_ja")
    runtime = engine.runtime
    runtime.token_count = lambda messages: fake.module.MAX_INPUT + 1 if len(messages) == 2 else 20
    runtime.outputs.append("safe-current-draft")
    before = len(runtime.generations)
    assert engine.translate("current-source", "zh_to_ja") == "safe-current-draft"
    assert len(runtime.generations) == before + 1
    assert engine.last_diagnostics["fallback"] == "context_over_budget"
    assert engine.last_diagnostics["history_used"] == 0


@pytest.mark.parametrize("over_budget", [False, True])
def test_current_input_limit_boundary_never_silently_truncates(fake, over_budget):
    engine = fake.make()
    engine.runtime.token_count = lambda messages: fake.module.MAX_INPUT + int(over_budget)
    current = "full-current-source-which-must-be-preserved"
    if over_budget:
        with pytest.raises(RuntimeError, match="长度上限"):
            engine.translate(current, "zh_to_ja")
        assert engine.runtime.generations == []
    else:
        engine.translate(current, "zh_to_ja")
        assert engine.last_diagnostics["draft_info"]["input_tokens"] == fake.module.MAX_INPUT
    assert only_user(engine.runtime.rendered[0]).endswith(current)


@pytest.mark.parametrize("bad_completion", [
    {"output": "partial", "stop_type": "limit"},
    {"output": "partial", "truncated": True},
    {"output": ""},
    {"output": "partial", "no_final": True},
])
def test_incomplete_or_empty_generation_is_not_returned_as_translation(fake, bad_completion):
    engine = fake.make()
    engine.runtime.outputs.append(bad_completion)
    with pytest.raises(RuntimeError, match="未完整生成"):
        engine.translate("current-source", "ja_to_zh")
    assert engine.last_diagnostics["calls"] == 1
    assert engine.last_diagnostics["completed_calls"] == 0


@pytest.mark.parametrize("failure", [HyRuntimeError("transport failed"), RuntimeError("decode failed")])
def test_failed_generation_is_counted_as_attempted_but_not_completed(fake, failure):
    engine = fake.make()
    engine.runtime.outputs.append(failure)
    with pytest.raises(type(failure), match=str(failure)):
        engine.translate("current-source", "ja_to_zh")
    assert engine.last_diagnostics["calls"] == 1
    assert engine.last_diagnostics["completed_calls"] == 0


def test_failed_independent_fallback_counts_both_completion_attempts(fake):
    engine = fake.make()
    engine.translate("previous-source", "ja_to_zh")
    engine.runtime.outputs.extend(["broken-context", {"output": "partial", "stop_type": "limit"}])
    fake.module.candidate_issues.side_effect = (
        lambda source, target, direction: ["output_repetition"] if target == "broken-context" else [])
    with pytest.raises(RuntimeError, match="未完整生成"):
        engine.translate("current-source", "ja_to_zh")
    assert engine.last_diagnostics["calls"] == 2
    assert engine.last_diagnostics["completed_calls"] == 1


def test_close_failure_can_be_retried_but_translation_is_already_closed(fake):
    engine = fake.make()
    fake.close_errors.append(RuntimeError("fixture cleanup failed"))
    with pytest.raises(RuntimeError, match="cleanup failed"):
        engine.close()
    with pytest.raises(RuntimeError, match="已关闭"):
        engine.translate("source", "zh_to_ja")
    engine.close()
    assert engine.runtime.closes == 2


@pytest.mark.parametrize("issue", ["output_repetition", "output_punctuation", "output_invalid_characters"])
def test_broken_first_output_is_rejected_with_quality_message(fake, issue):
    engine = fake.make()
    engine.runtime.outputs.append("想" * 15)
    fake.module.candidate_issues.return_value = [issue]
    with pytest.raises(RuntimeError, match="异常重复或乱码"):
        engine.translate("私の心そっと覗いてみませんか 欲し。", "ja_to_zh")
    assert engine.last_diagnostics["draft_issues"] == [issue]
    assert len(engine.runtime.generations) == 1


def test_broken_context_output_retries_current_source_without_reusing_garbage(fake):
    engine = fake.make()
    engine.translate("前の原文", "ja_to_zh")
    engine.runtime.outputs.extend(["想" * 15, "要不要悄悄看看我的内心？"])
    fake.module.candidate_issues.side_effect = [["output_repetition"], []]
    assert engine.translate("私の心そっと覗いてみませんか", "ja_to_zh") == "要不要悄悄看看我的内心？"
    assert len(engine.runtime.generations[-1]["messages"]) == 1
    assert "想" * 15 not in only_user(engine.runtime.generations[-1]["messages"])
    assert engine.last_diagnostics["calls"] == 2
    assert engine.last_diagnostics["fallback"] == "current_sentence_draft"


def test_long_japanese_context_echo_retries_only_current_sentence(fake):
    engine = fake.make()
    first_source = "私の心そっと覗いてみませんか 欲し。"
    second_source = "今物だけ溢れかえていませんか、まだ別。"
    first_target = "你能轻轻看看我的心吗？我想要。"
    second_target = "现在只有物质充满了你，还有别的吗？"
    engine.runtime.outputs.extend([first_target, second_target])
    engine.translate(first_source, "ja_to_zh", context_id="system")
    engine.translate(second_source, "ja_to_zh", context_id="system")
    echoed = "请把我的心轻轻看一眼吧，我想要。现在只有物质充满了你，还有别的吗？"
    current = "ばらよもっともっときゅっと詰め込みでいっそあな。"
    corrected = "把一切都塞得满满的，干脆就那样吧。"
    engine.runtime.outputs.extend([echoed, corrected])
    before = len(engine.runtime.generations)

    assert engine.translate(current, "ja_to_zh", context_id="system") == corrected

    contextual, independent = engine.runtime.generations[before:]
    assert first_source in only_user(contextual["messages"])
    assert second_source in only_user(contextual["messages"])
    assert first_target not in only_user(contextual["messages"])
    assert second_target not in only_user(contextual["messages"])
    assert independent["messages"] == [{"role": "user", "content": only_user(independent["messages"])}]
    assert only_user(independent["messages"]).endswith(current)
    assert engine.last_diagnostics["calls"] == 2
    assert engine.last_diagnostics["issues"] == ["context_echo"]
    assert engine.last_diagnostics["draft_issues"] == []
    assert engine.last_diagnostics["fallback"] == "current_sentence_draft"
    assert list(engine._successful_history[("system", "ja_to_zh")]) == [
        (second_source, second_target), (current, corrected)]
    for generation in (contextual, independent):
        assert generation["body"]["temperature"] == 0
        assert generation["body"]["n_predict"] == (816 if generation is contextual else 256)


def test_independent_paraphrase_can_reuse_valid_translation_after_context_echo_fallback(fake):
    engine = fake.make()
    old_source = "金額と日付に誤りがないか確認してください。"
    old_target = "请检查金额和日期是否存在任何错误。"
    current = "間違いを防ぐため、代金と予定日をもう一度見てください。"
    engine.runtime.outputs.append(old_target)
    engine.translate(old_source, "ja_to_zh")
    engine.runtime.outputs.extend([old_target, old_target])
    before = len(engine.runtime.generations)

    assert engine.translate(current, "ja_to_zh") == old_target

    assert len(engine.runtime.generations) == before + 2
    assert len(engine.runtime.generations[-1]["messages"]) == 1
    assert engine.last_diagnostics["issues"] == ["context_echo"]
    assert engine.last_diagnostics["draft_issues"] == []
    assert list(engine._successful_history[("default", "ja_to_zh")]) == [(old_source, old_target), (current, old_target)]
    assert list(engine._history[("default", "ja_to_zh")]) == [old_source, current]


@pytest.mark.parametrize("old_source,current,old_target,target", [
    ("明日の会議では企画書を全員に配布してください。", "明日の会議では企画書を全員に配布してください。",
     "请在明天的会议上向所有人分发企划书。", "请在明天的会议上向所有人分发企划书。"),
    ("明日の会議では企画書を全員に配布してください。", "明日の会議では企画書を全員に配布してくださいね。",
     "请在明天的会议上向所有人分发企划书。", "请在明天的会议上向所有人分发企划书。"),
    ("今からレイトレーシングのテストを始めます。", "レイトレーシングを有効にしてフレームレートを確認します。",
     "现在开始进行光线追踪测试，请打开性能统计。", "光线追踪功能已经开启，接下来检查帧率。"),
    ("番号をここに書きます。", "これが別の番号です。",
     "这个编号是12345678901234567890，请记下来。", "另一个编号是12345678901234567890，请确认。"),
    ("彼に渡してください。", "彼は今どこにいますか。",
     "请把那个东西交给他，然后回来这里。", "他现在在哪里，你能看到他吗？"),
])
def test_near_repeated_source_and_shared_short_terms_numbers_or_referents_do_not_retry(
        fake, old_source, current, old_target, target):
    engine = fake.make()
    engine.runtime.outputs.extend([old_target, target])
    engine.translate(old_source, "ja_to_zh")
    before = len(engine.runtime.generations)

    assert engine.translate(current, "ja_to_zh") == target

    assert len(engine.runtime.generations) == before + 1
    assert engine.last_diagnostics["calls"] == 1
    assert engine.last_diagnostics["issues"] == []
    assert engine.last_diagnostics["fallback"] == ""


@pytest.mark.parametrize("old_target,candidate,expected", [
    ("abcdefghijklmnopqrst", "abcdefghijklUVWXYZABCDEFGHIJKLMN", False),
    ("abcdefghijklmnopqrst", "abcdefghijklUVWXYZABCDEFGHIJKL", True),
    ("abcdefghijklmnopqrstu", "abcdefghijklUVWXYZABCDEFGHIJKL", False),
    ("abcdefghijk", "abcdefghijk", False),
    ("12345678901234567890", "12345678901234567890", False),
])
def test_echo_requires_twelve_language_characters_and_both_coverage_thresholds(fake, old_target, candidate, expected):
    assert fake.module._context_echo("ぜんぜん別の話です", candidate, [("以前の会話", old_target)]) is expected


def test_echo_reference_is_channel_scoped_bounded_successful_and_cleared_on_close(fake):
    engine = fake.make()
    targets = ["这是最初一句与其他句子不同的译文。", "第二条内容涉及今天早上的天气情况。",
               "第三条独立说明如何使用新的系统功能。"]
    sources = ["初めの挨拶をします。", "今日は暖かい朝です。", "新しいシステムを説明します。"]
    for source, target in zip(sources, targets):
        engine.runtime.outputs.append(target)
        engine.translate(source, "ja_to_zh", context_id="system")
    assert list(engine._successful_history[("system", "ja_to_zh")]) == list(zip(sources[1:], targets[1:]))
    engine.runtime.outputs.append(targets[-1])
    assert engine.translate("マイク側の最初の発言です。", "ja_to_zh", context_id="mic") == targets[-1]
    assert engine.last_diagnostics["calls"] == 1
    engine.runtime.outputs.append(targets[-1])
    engine.translate("中文当前句不采用日文回显检查。", "zh_to_ja", context_id="system")
    assert ("system", "zh_to_ja") not in engine._successful_history
    engine.close()
    assert engine._successful_history == {}
    assert engine._history == {}


def test_echo_ignores_successful_source_evicted_by_untranslated_fragments(fake):
    engine = fake.make()
    old_source = "最初の会話では窓を開けてください。"
    old_target = "请在最初的对话结束之后打开窗户。"
    engine.runtime.outputs.append(old_target)
    engine.translate(old_source, "ja_to_zh")
    fake.module.source_fragment_issue.return_value = "unresolved_phonetic_fragment"
    engine.translate("たったった！", "ja_to_zh")
    engine.translate("しーしー！", "ja_to_zh")
    fake.module.source_fragment_issue.return_value = None
    engine.runtime.outputs.append(old_target)
    before = len(engine.runtime.generations)

    assert engine.translate("全く異なる質問について、先に答えてください。", "ja_to_zh") == old_target

    assert len(engine.runtime.generations) == before + 1
    assert engine.last_diagnostics["calls"] == 1
    assert engine.last_diagnostics["completed_calls"] == 1
    assert engine.last_diagnostics["issues"] == []
    assert old_source not in only_user(engine.runtime.generations[-1]["messages"])


def test_echo_ignores_successful_source_dropped_from_actual_prompt_budget(fake):
    engine = fake.make()
    old_source = "最初の会話では窓を開けてください。"
    old_target = "请在最初的对话结束之后打开窗户。"
    newer_source = "今日の昼から雨が降り始めました。"
    engine.runtime.outputs.extend([old_target, "今天中午以后天气变得阴沉并开始下雨。"])
    engine.translate(old_source, "ja_to_zh")
    engine.translate(newer_source, "ja_to_zh")
    engine.runtime.token_count = lambda messages: (
        fake.module.MAX_INPUT + 1 if any(old_source in item["content"] for item in messages) else 20)
    engine.runtime.outputs.append(old_target)
    before = len(engine.runtime.generations)

    assert engine.translate("全く異なる質問について、先に答えてください。", "ja_to_zh") == old_target

    assert len(engine.runtime.generations) == before + 1
    assert engine.last_diagnostics["calls"] == 1
    assert engine.last_diagnostics["completed_calls"] == 1
    assert engine.last_diagnostics["history_used"] == 1
    assert engine.last_diagnostics["issues"] == []
    context = only_user(engine.runtime.generations[-1]["messages"])
    assert old_source not in context and newer_source in context


def test_failed_japanese_candidates_are_never_added_to_echo_reference(fake):
    engine = fake.make()
    source = "この機械の操作手順を最初から説明します。"
    target = "我会从头开始说明这台机器的操作步骤。"
    engine.runtime.outputs.append(target)
    engine.translate(source, "ja_to_zh")
    engine.runtime.outputs.extend(["bad-context", "bad-independent"])
    fake.module.candidate_issues.return_value = ["output_repetition"]
    with pytest.raises(RuntimeError, match="保留原文"):
        engine.translate("失敗した翻訳の原文です。", "ja_to_zh")
    assert list(engine._successful_history[("default", "ja_to_zh")]) == [(source, target)]


def test_constructor_preserves_loading_error_when_cleanup_also_fails(fake):
    fake.start_error = ValueError("fixture startup failure")
    fake.close_errors.append(RuntimeError("fixture cleanup failure"))
    with pytest.raises(ValueError, match="startup failure") as error:
        fake.make()
    assert fake.instances[0].closes == 1
    assert any("RuntimeError" in note for note in error.value.__notes__)


def test_constructor_runtime_cancellation_uses_translation_cancelled(fake):
    fake.start_error = HyRuntimeCancelled("fixture loading cancelled")
    with pytest.raises(translation_engines.TranslationCancelled) as error:
        fake.make()
    assert isinstance(error.value.__cause__, HyRuntimeCancelled)
    assert fake.instances[0].closes == 1


def test_runtime_failure_is_not_silently_replaced_by_draft_or_another_backend(fake):
    engine = fake.make()
    engine.translate("prior-source", "zh_to_ja")
    engine.runtime.outputs.extend(["safe-current-draft", HyRuntimeError("fixture server retired")])
    with pytest.raises(HyRuntimeError, match="retired"):
        engine.translate("current-source", "zh_to_ja")
    assert len(fake.instances) == 1


def test_factory_selects_hymt_and_preserves_stop_event(fake):
    stop = threading.Event()
    engine = fake.make(stop_event=stop, factory=True)
    assert isinstance(engine, fake.module.HyTranslationEngine)
    assert engine.backend_info["backend"] == "hymt"
    assert engine.runtime.kwargs["stop_event"] is stop
    assert len(fake.instances) == 1


def test_japanese_context_uses_two_source_sentences_without_a_draft_anchor(fake):
    engine = fake.make()
    for source in ("old-source", "previous-one", "previous-two"):
        engine.translate(source, "ja_to_zh", context_id="system")
    engine.runtime.outputs.append("current-result")
    before = len(engine.runtime.generations)
    assert engine.translate("current-source", "ja_to_zh", context_id="system") == "current-result"
    calls = engine.runtime.generations[before:]
    assert len(calls) == 1
    messages = calls[0]["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    context = only_user(messages)
    assert "previous-one\n§\nprevious-two" in context
    assert "old-source" not in context and "fixture-target" not in context
    assert only_user(messages).endswith("current-source")
    assert context.endswith("\n§\ncurrent-source")
    assert "当前句的独立初译" not in context
    assert calls[0]["body"]["n_predict"] == 816
    assert engine.last_diagnostics["history_used"] == 2
    assert engine.last_diagnostics["calls"] == 1


def test_rejected_japanese_context_retries_only_current_source(fake):
    engine = fake.make()
    engine.translate("prior-source", "ja_to_zh")
    engine.runtime.outputs.extend(["unsafe-ban", "safe-wish"])
    fake.module.candidate_issues.side_effect = lambda source, output, direction: ["intent_reversal"] if output == "unsafe-ban" else []
    assert engine.translate("current-wish", "ja_to_zh") == "safe-wish"
    assert "prior-source" in only_user(engine.runtime.generations[-2]["messages"])
    assert "prior-source" not in only_user(engine.runtime.generations[-1]["messages"])
    assert engine.last_diagnostics["fallback"] == "current_sentence_draft"
    assert engine.last_diagnostics["issues"] == ["intent_reversal"]
    assert engine.last_diagnostics["calls"] == 2


def test_both_japanese_candidates_rejected_preserve_source_without_returning_bad_text(fake):
    engine = fake.make()
    engine.translate("prior-source", "ja_to_zh")
    fake.module.candidate_issues.return_value = ["intent_reversal"]
    with pytest.raises(RuntimeError, match="保留原文"):
        engine.translate("failed-current-source", "ja_to_zh")
    assert list(engine._history[("default", "ja_to_zh")]) == ["prior-source", "failed-current-source"]


def test_japanese_context_budget_drops_whole_older_sentence_before_generation(fake):
    engine = fake.make()
    engine.translate("old-source", "ja_to_zh")
    engine.translate("new-source", "ja_to_zh")
    engine.runtime.token_count = lambda messages: fake.module.MAX_INPUT + 1 if any("old-source" in message["content"] for message in messages) else 20
    engine.runtime.outputs.append("safe-result")
    before = len(engine.runtime.generations)
    assert engine.translate("full-current-source", "ja_to_zh") == "safe-result"
    assert len(engine.runtime.generations) == before + 1
    messages = engine.runtime.generations[-1]["messages"]
    assert "new-source" in only_user(messages) and "old-source" not in only_user(messages)
    assert only_user(messages).endswith("full-current-source")
    assert engine.last_diagnostics["history_used"] == 1


def test_japanese_current_source_kept_when_all_context_is_over_budget(fake):
    engine = fake.make()
    engine.translate("prior-source", "ja_to_zh")
    engine.runtime.token_count = lambda messages: (
        fake.module.MAX_INPUT + 1 if messages[0]["content"].startswith(fake.module.PARAGRAPH_RULE) else 20)
    engine.runtime.outputs.append("current-only-result")
    assert engine.translate("full-current-source", "ja_to_zh") == "current-only-result"
    assert engine.last_diagnostics["calls"] == 1
    assert engine.last_diagnostics["history_used"] == 0
    assert engine.last_diagnostics["fallback"] == "context_over_budget"
    assert only_user(engine.runtime.generations[-1]["messages"]).endswith("full-current-source")


def test_unresolved_fragment_is_retained_without_model_guess_or_deduplication(fake):
    engine = fake.make()
    engine.translate("前の話題です。", "ja_to_zh", context_id="system")
    fake.module.source_fragment_issue.return_value = "unresolved_phonetic_fragment"
    before = len(engine.runtime.generations)
    for _ in range(2):
        assert engine.translate("たったった!", "ja_to_zh", context_id="system") == "たったった!（音节含义待确认）"
        assert engine.last_diagnostics["calls"] == 0
        assert engine.last_diagnostics["history_used"] == 0
        assert engine.last_diagnostics["fallback"] == "source_preserved"
    assert len(engine.runtime.generations) == before
    assert list(engine._history[("system", "ja_to_zh")]) == ["たったった!", "たったった!"]


def test_normal_short_positive_and_repeated_real_speech_still_reach_translation(fake):
    engine = fake.make()
    engine.runtime.outputs.extend(["太好了！", "成功了，成功了！"])
    assert engine.translate("ナイス！", "ja_to_zh") == "太好了！"
    assert engine.translate("やった、やった！", "ja_to_zh") == "成功了，成功了！"
    assert len(engine.runtime.generations) == 2
    assert list(engine._history[("default", "ja_to_zh")]) == ["ナイス！", "やった、やった！"]


def test_fragment_policy_keeps_ordinary_prompts_and_chinese_negative_anchor(fake):
    engine = fake.make()
    engine.translate("それはよかった。", "ja_to_zh")
    assert only_user(engine.runtime.generations[-1]["messages"]) == (
        "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\nそれはよかった。")
    engine.translate("不要付款。", "zh_to_ja", context_id="mic")
    messages = engine.runtime.generations[-1]["messages"]
    assert only_user(messages) == "将以下文本翻译为日语，注意只需要输出翻译后的结果，不要额外解释：\n不要付款。"
    engine.translate("不要发货。", "zh_to_ja", context_id="mic")
    assert "当前句的独立初译" in system_text(engine.runtime.generations[-1]["messages"])
    assert list(engine._history[("mic", "zh_to_ja")]) == ["不要付款。", "不要发货。"]


@pytest.mark.parametrize("previous,raw", [
    (["private-old-one"], "以前的译文\n§\nprivate-current-target"),
    (["private-old-one", "private-old-two"], "以前的译文一\n§\n以前的译文二\n§\nprivate-current-target"),
])
def test_paragraph_alignment_returns_only_current_and_keeps_text_out_of_diagnostics(fake, previous, raw):
    import json

    engine = fake.make()
    engine._history[("system", "ja_to_zh")] = deque(previous, maxlen=2)
    engine.runtime.outputs.append({"output": raw, "raw_paragraph_output": True})
    assert engine.translate("private-current-source", "ja_to_zh", context_id="system") == "private-current-target"
    call = engine.runtime.generations[-1]
    row_count = len(previous) + 1
    assert call["body"]["n_predict"] == row_count * 256 + 48
    assert "grammar" not in call["body"]
    assert call["body"]["temperature"] == 0
    assert call["body"]["seed"] == 42
    assert engine.last_diagnostics["context_info"]["input_budget"] == 2048 - row_count * 256 - 48 - 16
    assert engine.last_diagnostics["paragraph_alignment"]["validated"] is True
    assert engine.last_diagnostics["calls"] == engine.last_diagnostics["completed_calls"] == 1
    serialized = json.dumps(engine.last_diagnostics, ensure_ascii=False)
    for private_text in previous + ["private-current-source", "private-current-target", "以前的译文"]:
        assert private_text not in serialized


@pytest.mark.parametrize("raw,issue", [
    ("only-current", "paragraph_count_mismatch"),
    ("old§current§extra", "paragraph_count_mismatch"),
    ('{"current":"bad-json-response"}', "paragraph_count_mismatch"),
    ("old§   ", "paragraph_empty_segment"),
    ("  §current", "paragraph_empty_segment"),
])
def test_bad_paragraph_structure_is_never_published_and_retries_current_once(fake, raw, issue):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["previous-source"], maxlen=2)
    engine.runtime.outputs.extend([{"output": raw, "raw_paragraph_output": True}, "safe-independent"])
    assert engine.translate("current-source", "ja_to_zh") == "safe-independent"
    assert engine.last_diagnostics["issues"] == [issue]
    assert engine.last_diagnostics["fallback"] == "current_sentence_draft"
    assert engine.last_diagnostics["paragraph_alignment"]["validated"] is False
    assert engine.last_diagnostics["calls"] == engine.last_diagnostics["completed_calls"] == 2
    assert len(engine.runtime.generations) == 2
    assert only_user(engine.runtime.generations[-1]["messages"]).endswith("current-source")
    assert "previous-source" not in only_user(engine.runtime.generations[-1]["messages"])


def test_source_delimiter_collision_selects_absent_symbol_without_adding_it_to_instruction(fake):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["previous contains §"], maxlen=2)
    engine.runtime.outputs.append("current-result")
    assert engine.translate("current source", "ja_to_zh") == "current-result"
    prompt = only_user(engine.runtime.generations[-1]["messages"])
    assert prompt == fake.module.PARAGRAPH_RULE + "previous contains §\n¶\ncurrent source"
    assert prompt.count("¶") == 1
    assert "¶" not in fake.module.PARAGRAPH_RULE
    assert engine.last_diagnostics["paragraph_alignment"]["delimiter_index"] == 1


def test_no_source_absent_delimiter_uses_independent_request_without_native_paragraph_attempt(fake):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["previous §¶¤¦"], maxlen=2)
    engine.runtime.outputs.append("independent-target")
    assert engine.translate("current-source", "ja_to_zh") == "independent-target"
    assert engine.last_diagnostics["issues"] == ["paragraph_delimiter_unavailable"]
    assert engine.last_diagnostics["calls"] == engine.last_diagnostics["completed_calls"] == 1
    assert engine.last_diagnostics["history_used"] == 0
    assert engine.runtime.generations[0]["body"]["n_predict"] == 256


@pytest.mark.parametrize("over_three_row_budget", [False, True])
def test_paragraph_input_budget_boundary_drops_only_whole_oldest_source(fake, over_three_row_budget):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["oldest-complete", "newer-complete"], maxlen=2)
    engine.runtime.token_count = lambda messages: (
        1216 + int(over_three_row_budget) if "oldest-complete" in only_user(messages) else 1472)
    engine.runtime.outputs.append("current-result")
    assert engine.translate("complete-current-source", "ja_to_zh") == "current-result"
    assert len(engine.runtime.generations) == 1
    call = engine.runtime.generations[0]
    prompt = only_user(call["messages"])
    assert prompt.endswith("complete-current-source")
    assert "newer-complete" in prompt
    assert ("oldest-complete" in prompt) is not over_three_row_budget
    assert call["body"]["n_predict"] == (560 if over_three_row_budget else 816)
    assert engine.last_diagnostics["history_used"] == (1 if over_three_row_budget else 2)
    assert engine.last_diagnostics["calls"] == 1


def test_all_paragraph_budgets_exceeded_use_original_current_budget_without_truncation(fake):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["oldest-complete", "newer-complete"], maxlen=2)

    def tokens(messages):
        content = only_user(messages)
        if content.startswith(fake.module.PARAGRAPH_RULE):
            return 1217 if "oldest-complete" in content else 1473
        return 1776

    engine.runtime.token_count = tokens
    engine.runtime.outputs.append("independent-target")
    assert engine.translate("complete-current-source", "ja_to_zh") == "independent-target"
    assert len(engine.runtime.rendered) == 3
    assert len(engine.runtime.generations) == 1
    assert only_user(engine.runtime.generations[0]["messages"]).endswith("complete-current-source")
    assert engine.runtime.generations[0]["body"]["n_predict"] == 256
    assert engine.last_diagnostics["history_used"] == 0
    assert engine.last_diagnostics["fallback"] == "context_over_budget"
    assert engine.last_diagnostics["draft_info"]["input_tokens"] == 1776


def test_current_over_budget_after_context_removal_still_fails_before_native_generation(fake):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["oldest-complete", "newer-complete"], maxlen=2)
    engine.runtime.token_count = lambda messages: 1777
    with pytest.raises(RuntimeError, match="长度上限"):
        engine.translate("complete-current-source", "ja_to_zh")
    assert engine.runtime.generations == []
    assert all(only_user(messages).endswith("complete-current-source") for messages in engine.runtime.rendered)
    assert engine.last_diagnostics["calls"] == engine.last_diagnostics["completed_calls"] == 0


@pytest.mark.parametrize("previous,source,expected", [
    (["強化した攻撃を当てた。", "相手の体力ゲージが一気に減った。"], "え、こんなに早く溶けるの？",
     [("hp_loss", "溶ける", "掉血")]),
    (["配信終了まであと10分しかない。", "まだ二つコーナーが残っている。"], "後半は少し巻いていこう。",
     [("hurry_progress", "巻いて", "加快进度")]),
    (["新しい茶こしを使った。", "小さい茶葉が通り抜けた。"], "この目は少し粗いね。",
     [("mesh_opening_size", "目", "网眼"), ("mesh_opening_size", "粗い", "偏大")]),
    (["強化した攻撃を当てた。", "相手の体力ゲージが一気に減った。"], "こっちのHPがすぐ溶けるわけじゃないよ。",
     [("hp_loss", "溶ける", "掉血")]),
    (["配信終了まであと10分しかない。", "まだ二つコーナーが残っている。"], "後半も巻いていかないで。",
     [("hurry_progress", "巻いて", "加快进度")]),
    (["新しい茶こしを使った。", "小さい茶葉が通り抜けた。"], "この網の目は粗いわけじゃないよ。",
     [("mesh_opening_size", "目", "网眼"), ("mesh_opening_size", "粗い", "偏大")]),
])
def test_evidenced_context_terms_are_short_hints_without_rewriting_current_source(fake, previous, source, expected):
    import json

    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(previous, maxlen=2)
    engine.runtime.outputs.append({"output": "old-one\n§\nold-two\n§\nunchanged-model-output", "raw_paragraph_output": True})
    assert engine.translate(source, "ja_to_zh") == "unchanged-model-output"
    prompt = only_user(engine.runtime.generations[0]["messages"])
    prefix = "参考下面的翻译：\n" + "\n".join(f"{term} 翻译成 {meaning}" for _sense, term, meaning in expected) + "\n"
    assert prompt == prefix + fake.module.PARAGRAPH_RULE + "\n§\n".join(previous + [source])
    assert prompt.count("§") == 2
    assert "§" not in prefix
    assert engine.last_diagnostics["context_term_senses"] == list(dict.fromkeys(sense for sense, _term, _meaning in expected))
    assert engine.last_diagnostics["context_term_count"] == len(expected)
    assert engine.last_diagnostics["calls"] == 1
    request = engine.runtime.generations[0]["body"]
    assert request["n_predict"] == 816
    assert request["temperature"] == 0 and request["seed"] == 42
    serialized = json.dumps(engine.last_diagnostics, ensure_ascii=False)
    for private_text in previous + [source, "unchanged-model-output"] + [term for _sense, term, _meaning in expected]:
        assert private_text not in serialized


@pytest.mark.parametrize("previous,source", [
    (["強化した攻撃を当てた。", "相手の体力ゲージが一気に減った。"], "この氷、こんなに早く溶けるの？"),
    (["強化した攻撃を当てた。", "相手の体力ゲージが一気に減った。"], "ろうそくの蝋はまだ溶けるわけじゃないよ。"),
    (["相手の体力ゲージはまだ満タン。", "攻撃は一度も当たっていない。"], "え、こんなに早く溶けるの？"),
    (["配信終了まであと10分しかない。", "まだ二つコーナーが残っている。"], "後半に使うケーブルを巻いておこう。"),
    (["配信終了まであと10分しかない。", "まだ二つコーナーが残っている。"], "まず袖を巻いてから作業しよう。"),
    (["新しい茶こしを使った。", "網の穴の大きさを確認している。"], "この子の目は、少し粗い線で描かれているね。"),
    (["人物の絵を鉛筆で描いている。", "まだ輪郭しか描いていない。"], "この目は少し粗いね。"),
    (["何も変わっていない。", "普通に話しています。"], "これは新しい話題です。"),
])
def test_literal_objects_or_missing_sense_evidence_keep_paragraph_prompt_identical(fake, previous, source):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(previous, maxlen=2)
    engine.runtime.outputs.append("current-target")
    assert engine.translate(source, "ja_to_zh") == "current-target"
    prompt = only_user(engine.runtime.generations[0]["messages"])
    assert prompt == fake.module.PARAGRAPH_RULE + "\n§\n".join(previous + [source])
    assert engine.last_diagnostics["context_term_senses"] == []
    assert engine.last_diagnostics["context_term_count"] == 0


def test_term_evidence_is_removed_when_oldest_source_is_dropped_for_budget(fake):
    engine = fake.make()
    oldest, newer, source = "この茶こしは新品です。", "今日は晴れです。", "この目は少し粗いね。"
    engine._history[("default", "ja_to_zh")] = deque([oldest, newer], maxlen=2)
    engine.runtime.token_count = lambda messages: 1217 if oldest in only_user(messages) else 20
    engine.runtime.outputs.append("current-target")
    assert engine.translate(source, "ja_to_zh") == "current-target"
    assert only_user(engine.runtime.rendered[0]).startswith("参考下面的翻译：\n目 翻译成 网眼\n粗い 翻译成 偏大\n")
    assert only_user(engine.runtime.generations[0]["messages"]) == fake.module.PARAGRAPH_RULE + newer + "\n§\n" + source
    assert engine.last_diagnostics["history_used"] == 1
    assert engine.last_diagnostics["context_term_senses"] == []
    assert engine.last_diagnostics["context_term_count"] == 0
    assert engine.last_diagnostics["calls"] == 1


def test_rejected_terminology_context_uses_unmodified_current_only_fallback(fake):
    engine = fake.make()
    source = "この目は少し粗いね。"
    engine._history[("default", "ja_to_zh")] = deque(["茶こしを使った。", "小さな葉が通った。"], maxlen=2)
    engine.runtime.outputs.extend([{"output": "old-one§old-two§unsafe-context", "raw_paragraph_output": True}, "safe-independent"])
    fake.module.candidate_issues.side_effect = lambda src, target, direction: ["number_mismatch"] if target == "unsafe-context" else []
    assert engine.translate(source, "ja_to_zh") == "safe-independent"
    assert only_user(engine.runtime.generations[0]["messages"]).startswith("参考下面的翻译：")
    assert only_user(engine.runtime.generations[1]["messages"]) == (
        "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n" + source)
    assert engine.last_diagnostics["fallback"] == "current_sentence_draft"
    assert engine.last_diagnostics["calls"] == 2


def test_terminology_is_not_used_in_chinese_direction_or_context_free_japanese(fake, monkeypatch):
    engine = fake.make()
    monkeypatch.setattr(fake.module, "_context_term_hints", Mock(side_effect=AssertionError("Japanese context helper must not run")))
    engine.runtime.outputs.append("context-free-target")
    source = "HPが一気に減った。こんなに早く溶けるの？"
    assert engine.translate(source, "ja_to_zh") == "context-free-target"
    assert only_user(engine.runtime.generations[-1]["messages"]) == (
        "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n" + source)
    engine._history[("default", "zh_to_ja")] = deque(["茶こしを使った。", "小さな葉が通った。"], maxlen=2)
    engine.runtime.outputs.extend(["independent-draft", "context-result"])
    assert engine.translate("この目は少し粗いね。", "zh_to_ja") == "context-result"
    assert "参考下面的翻译：" not in system_text(engine.runtime.generations[-1]["messages"])
    assert "context_term_senses" not in engine.last_diagnostics


def test_truncated_paragraph_generation_uses_one_independent_retry_with_accurate_attempt_counts(fake):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["previous-source"], maxlen=2)
    engine.runtime.outputs.extend([{"output": "old§private-partial", "raw_paragraph_output": True, "stop_type": "limit"},
                                   "safe-independent"])
    assert engine.translate("current-source", "ja_to_zh") == "safe-independent"
    assert engine.last_diagnostics["issues"] == ["paragraph_incomplete"]
    assert engine.last_diagnostics["calls"] == 2
    assert engine.last_diagnostics["completed_calls"] == 1
    assert engine.last_diagnostics["history_used"] == 1
    assert [item["output_budget"] for item in engine.last_diagnostics["generation_attempts"]] == [560, 256]
    assert "private-partial" not in repr(engine.last_diagnostics)


def test_current_guards_receive_only_extracted_target_and_actual_source_context(fake):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["first-source", "second-source"], maxlen=2)
    engine.runtime.outputs.append({"output": "old-translation-one§old-translation-two§current-target", "raw_paragraph_output": True})
    assert engine.translate("current-source", "ja_to_zh") == "current-target"
    fake.module.candidate_issues.assert_called_once_with("current-source", "current-target", "ja_to_zh")
    fake.module.context_grounding_issues.assert_called_once_with("current-source", "current-target", ["first-source", "second-source"])


@pytest.mark.parametrize("fallback_still_ungrounded", [False, True])
def test_ungrounded_time_retries_current_without_history_then_rechecks(fake, fallback_still_ungrounded):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["source-with-no-time"], maxlen=2)
    engine.runtime.outputs.extend(["在上午进行。", "在上午进行。" if fallback_still_ungrounded else "在这个时段进行。"])
    fake.module.context_grounding_issues.side_effect = lambda source, target, prior: (
        ("ungrounded_time_reference",) if "上午" in target else ())
    if fallback_still_ungrounded:
        with pytest.raises(RuntimeError, match="未确认的时段，本句保留原文") as error:
            engine.translate("こっちの時間にやります。", "ja_to_zh")
        assert "上午" not in str(error.value)
    else:
        assert engine.translate("こっちの時間にやります。", "ja_to_zh") == "在这个时段进行。"
    assert engine.last_diagnostics["issues"] == ["ungrounded_time_reference"]
    assert engine.last_diagnostics["calls"] == 2
    assert fake.module.context_grounding_issues.call_args_list[0].args[2] == ["source-with-no-time"]
    assert fake.module.context_grounding_issues.call_args_list[1].args[2] == ()


def test_chinese_direction_does_not_call_japanese_context_grounding_guard(fake):
    engine = fake.make()
    fake.module.context_grounding_issues.side_effect = AssertionError("Japanese-only guard called")
    engine.translate("第一句中文。", "zh_to_ja")
    engine.translate("下一句中文。", "zh_to_ja")
    assert len(engine.runtime.generations) == 3
    assert "当前句的独立初译" in system_text(engine.runtime.generations[-1]["messages"])


def test_native_paragraph_failure_does_not_try_independent_translation_on_retired_server(fake):
    engine = fake.make()
    engine._history[("default", "ja_to_zh")] = deque(["previous-source"], maxlen=2)
    engine.runtime.outputs.extend([HyRuntimeError("fixture retired"), "never-used"])
    with pytest.raises(HyRuntimeError, match="retired"):
        engine.translate("current-source", "ja_to_zh")
    assert len(engine.runtime.generations) == 1
    assert engine.last_diagnostics["calls"] == 1
    assert engine.last_diagnostics["completed_calls"] == 0


def test_cancel_after_paragraph_guard_prevents_independent_retry(fake):
    stop = threading.Event()
    engine = fake.make(stop_event=stop)
    engine._history[("default", "ja_to_zh")] = deque(["previous-source"], maxlen=2)
    engine.runtime.outputs.extend(["unsafe-current", "never-used"])

    def reject_and_cancel(source, target, direction):
        stop.set()
        return ["prohibition"]

    fake.module.candidate_issues.side_effect = reject_and_cancel
    with pytest.raises(translation_engines.TranslationCancelled):
        engine.translate("current-source", "ja_to_zh")
    assert len(engine.runtime.generations) == 1
    assert engine.last_diagnostics["calls"] == engine.last_diagnostics["completed_calls"] == 1
    assert list(engine._history[("default", "ja_to_zh")]) == ["previous-source"]
