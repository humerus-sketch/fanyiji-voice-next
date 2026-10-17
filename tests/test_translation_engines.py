"""Translation contracts with local fixtures and fake native engines only."""

from dataclasses import dataclass
import json
from pathlib import Path
import runpy
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fanyiji_next import translation_engines as module
from fanyiji_next import native_runtime


TOKEN_IDS = {"</s>": 2, "<unk>": 3, "eng_Latn": 256047, "jpn_Jpan": 256079,
             "zho_Hans": 256200, "▁fixture": 10, "▁result": 11}


def tokenizer_json(layout="prefix", language="eng_Latn"):
    order = [language, "$A", "</s>"] if layout == "prefix" else ["$A", "</s>", language]
    return {"post_processor": {
        "type": "TemplateProcessing",
        "single": [{"Sequence": {"id": "A", "type_id": 0}} if item == "$A" else
                   {"SpecialToken": {"id": item, "type_id": 0}} for item in order],
        "special_tokens": {token: {"id": token, "ids": [TOKEN_IDS[token]], "tokens": [token]}
                           for token in (language, "</s>")},
    }, "truncation": {"max_length": 1}, "padding": {"length": 8}}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    bootstrap = Mock(return_value=())
    monkeypatch.setattr(native_runtime, "configure_cuda_runtime", bootstrap)
    old = tmp_path / "m2m"
    nllb = tmp_path / "nllb"
    for folder in (old, nllb):
        folder.mkdir()
        (folder / "model.bin").write_bytes(b"fixture")
        (folder / "config.json").write_text("{}", encoding="utf8")
        (folder / "shared_vocabulary.json").write_text("[]", encoding="utf8")
    spm = tmp_path / "spm.model"
    spm.write_bytes(b"fixture")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(tokenizer_json()), encoding="utf8")
    settings = SimpleNamespace(translation_model_dir=str(old), translation_spm=str(spm),
                               nllb_model_dir=str(nllb), nllb_tokenizer_dir=str(tokenizer_dir))
    translator = SimpleNamespace(device="cpu", compute_type="int8",
                                 translate_batch=Mock(return_value=[SimpleNamespace(hypotheses=[["__ja__", "out"]])]),
                                 unload_model=Mock())

    def native_factory(path, **kwargs):
        translator.device = kwargs["device"]
        translator.compute_type = kwargs["compute_type"]
        return translator

    native = Mock(side_effect=native_factory)
    support = Mock(return_value={"int8", "int8_float16", "float16"})
    sentencepiece = SimpleNamespace(encode=Mock(return_value=["piece"]), decode=Mock(return_value="fixture translation"))
    spm_factory = Mock(return_value=sentencepiece)
    tokenizer = SimpleNamespace(encode=Mock(return_value=SimpleNamespace(tokens=["▁fixture"])),
                                token_to_id=Mock(side_effect=TOKEN_IDS.get),
                                decode=Mock(return_value="fixture translation"),
                                no_truncation=Mock(), no_padding=Mock())
    tokenizer_factory = Mock(return_value=tokenizer)
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(Translator=native, get_supported_compute_types=support))
    monkeypatch.setitem(sys.modules, "sentencepiece", SimpleNamespace(SentencePieceProcessor=spm_factory))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=tokenizer_factory)))
    return SimpleNamespace(settings=settings, old=old, nllb=nllb, spm=sentencepiece,
                           spm_factory=spm_factory, tokenizer=tokenizer,
                           tokenizer_factory=tokenizer_factory, tokenizer_path=tokenizer_path,
                           translator=translator, native=native, support=support)


def make_nllb(fixture, *, layout="prefix", language="eng_Latn", config=None):
    fixture.settings.translation_backend = "nllb"
    fixture.tokenizer_path.write_text(json.dumps(tokenizer_json(layout, language)), encoding="utf8")
    (fixture.nllb / "config.json").write_text(json.dumps(config or {}), encoding="utf8")
    return module.create_translation_engine(fixture.settings)


def output_engine(fixture, backend, *, stop_event=None):
    fixture.settings.translation_backend = backend
    prefix = "zho_Hans" if backend == "nllb" else "__zh__"
    body = "▁result" if backend == "nllb" else "out"
    fixture.translator.translate_batch.return_value = [SimpleNamespace(hypotheses=[[prefix, body]])]
    engine = module.create_translation_engine(fixture.settings, stop_event=stop_event)
    return engine, fixture.tokenizer.decode if backend == "nllb" else fixture.spm.decode


def test_module_import_does_not_import_optional_native_dependencies(monkeypatch):
    import builtins
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"ctranslate2", "sentencepiece", "tokenizers", "transformers"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    runpy.run_path(module.__file__)


@pytest.mark.parametrize("config,source,prefix", [
    ({}, ["__zh__", "piece", "</s>"], ["__ja__"]),
    ({"add_source_eos": True}, ["__zh__", "piece"], ["__ja__"]),
    ({"decoder_start_token": None}, ["__zh__", "piece", "</s>"], ["</s>", "__ja__"]),
])
def test_m2m100_preserves_conversion_eos_and_decoder_prefix(fixture, config, source, prefix):
    (fixture.old / "config.json").write_text(json.dumps(config), encoding="utf8")
    engine = module.create_translation_engine(fixture.settings)
    assert engine.translate("fixture", "zh_to_ja") == "fixture translation"
    args, kwargs = fixture.translator.translate_batch.call_args
    assert args == ([source],)
    assert kwargs == {"target_prefix": [prefix], "beam_size": 2,
                      "max_input_length": 512, "max_decoding_length": 512}
    fixture.spm.decode.assert_called_once_with(["out"])
    assert engine.backend_info["backend"] == "m2m100"
    native_runtime.configure_cuda_runtime.assert_not_called()
    engine.close()


def test_m2m100_reverse_direction_and_special_token_cleanup(fixture):
    fixture.translator.translate_batch.return_value = [SimpleNamespace(hypotheses=[["<s>", "__zh__", "out", "</s>"]])]
    engine = module.create_translation_engine(fixture.settings)
    engine.translate("fixture", "ja_to_zh")
    args, kwargs = fixture.translator.translate_batch.call_args
    assert args[0] == [["__ja__", "piece", "</s>"]]
    assert kwargs["target_prefix"] == [["__zh__"]]
    fixture.spm.decode.assert_called_once_with(["out"])


@pytest.mark.parametrize("layout,language,expected", [
    ("prefix", "eng_Latn", ["zho_Hans", "▁fixture", "</s>"]),
    ("suffix", "eng_Latn", ["▁fixture", "</s>", "zho_Hans"]),
    ("suffix", "<unk>", ["▁fixture", "</s>", "zho_Hans"]),
])
def test_nllb_uses_local_template_and_language_codes(fixture, layout, language, expected):
    fixture.translator.translate_batch.return_value = [SimpleNamespace(hypotheses=[["jpn_Jpan", "▁result", "</s>"]])]
    engine = make_nllb(fixture, layout=layout, language=language)
    assert engine.translate("fixture", "zh_to_ja") == "fixture translation"
    args, kwargs = fixture.translator.translate_batch.call_args
    assert args[0] == [expected]
    assert kwargs["target_prefix"] == [["jpn_Jpan"]]
    fixture.tokenizer.encode.assert_called_once_with("fixture", add_special_tokens=False)
    fixture.tokenizer.decode.assert_called_once_with([11, 2], skip_special_tokens=True)
    fixture.tokenizer.no_truncation.assert_called_once()
    fixture.tokenizer.no_padding.assert_called_once()
    fixture.spm_factory.assert_not_called()
    assert engine.backend_info["source_layout"] == ("$LANG $A </s>" if layout == "prefix" else "$A </s> $LANG")


def test_nllb_reverse_direction_and_explicit_decoder_start(fixture):
    fixture.translator.translate_batch.return_value = [SimpleNamespace(hypotheses=[["</s>", "zho_Hans", "▁result"]])]
    engine = make_nllb(fixture, config={"decoder_start_token": None})
    engine.translate("fixture", "ja_to_zh")
    args, kwargs = fixture.translator.translate_batch.call_args
    assert args[0] == [["jpn_Jpan", "▁fixture", "</s>"]]
    assert kwargs["target_prefix"] == [["</s>", "zho_Hans"]]
    fixture.tokenizer.decode.assert_called_once_with([11], skip_special_tokens=True)


@pytest.mark.parametrize("backend", ["m2m100", "nllb"])
@pytest.mark.parametrize("broken,issue", [
    ("想" * 15, "output_repetition"),
    ("?? " * 150, "output_punctuation"),
    ("\ufffd" * 15, "output_invalid_characters"),
])
def test_degenerate_output_retries_once_without_changing_source_or_backend(fixture, backend, broken, issue):
    engine, decode = output_engine(fixture, backend)
    source = "私の心そっと覗いてみませんか 欲し。"
    decode.side_effect = [broken, "想轻轻窥探一下我的内心吗？"]

    assert engine.translate(source, "ja_to_zh") == "想轻轻窥探一下我的内心吗？"

    first, retry = fixture.translator.translate_batch.call_args_list
    assert first.args == retry.args
    assert first.kwargs == {"target_prefix": [["zho_Hans" if backend == "nllb" else "__zh__"]],
                            "beam_size": 2, "max_input_length": 512, "max_decoding_length": 512}
    assert retry.kwargs == {**first.kwargs, "beam_size": 5, "no_repeat_ngram_size": 3}
    assert engine.backend_info["backend"] == backend
    assert engine.last_diagnostics["calls"] == 2
    assert engine.last_diagnostics["first_issues"] == [issue]
    assert engine.last_diagnostics["retry_issues"] == []
    assert engine.last_diagnostics["issues"] == []
    assert engine.last_diagnostics["retry_outcome"] == "accepted"
    assert engine.last_diagnostics["retry_parameters"] == {
        "beam_size": 5, "no_repeat_ngram_size": 3, "max_input_length": 512, "max_decoding_length": 512}
    assert [attempt["issues"] for attempt in engine.last_diagnostics["attempts"]] == [[issue], []]
    encode = fixture.tokenizer.encode if backend == "nllb" else fixture.spm.encode
    assert encode.call_count == 1
    assert encode.call_args.args[0] == source


@pytest.mark.parametrize("backend", ["m2m100", "nllb"])
def test_second_degenerate_output_is_rejected_without_deduplication_or_third_call(fixture, backend):
    engine, decode = output_engine(fixture, backend)
    decode.side_effect = ["想" * 15, "?? " * 150]

    with pytest.raises(RuntimeError, match="译文出现异常重复或无有效文字，本句保留原文"):
        engine.translate("ばらよもっともっときゅっと詰め込みでいっそあな。", "ja_to_zh")

    assert fixture.translator.translate_batch.call_count == 2
    assert engine.last_diagnostics["calls"] == 2
    assert engine.last_diagnostics["first_issues"] == ["output_repetition"]
    assert engine.last_diagnostics["retry_issues"] == ["output_punctuation"]
    assert engine.last_diagnostics["issues"] == ["output_punctuation"]
    assert engine.last_diagnostics["retry_outcome"] == "rejected"


@pytest.mark.parametrize("backend", ["m2m100", "nllb"])
@pytest.mark.parametrize("source,retry,issue", [
    ("番号は111111111111111です。", "编号是111。", "number_mismatch"),
    ("まだ払わないでください。", "请付款。", "prohibition"),
    ("本を売れたらいいな。", "不要卖书。", "intent_reversal"),
])
def test_clean_looking_retry_cannot_lose_numbers_or_reverse_intent(fixture, backend, source, retry, issue):
    engine, decode = output_engine(fixture, backend)
    decode.side_effect = ["⁇ " * 20, retry]

    with pytest.raises(RuntimeError, match="重试译文的否定、意向或数值与原文不一致"):
        engine.translate(source, "ja_to_zh")

    assert fixture.translator.translate_batch.call_count == 2
    assert engine.last_diagnostics["retry_issues"] == [issue]
    assert engine.last_diagnostics["retry_outcome"] == "rejected"


@pytest.mark.parametrize("backend", ["m2m100", "nllb"])
@pytest.mark.parametrize("source,target", [
    ("はい" * 10, "是" * 15),
    ("番号は111111111111111です。", "编号是111111111111111。"),
    ("まだ30円を払わないでください。", "请先不要支付30日元。"),
    ("もっともっと強く。", "再用力一点，再用力一点。"),
])
def test_normal_or_explicit_source_repetition_keeps_existing_decoding(fixture, backend, source, target):
    engine, decode = output_engine(fixture, backend)
    decode.return_value = target

    assert engine.translate(source, "ja_to_zh") == target

    assert fixture.translator.translate_batch.call_count == 1
    assert fixture.translator.translate_batch.call_args.kwargs["beam_size"] == 2
    assert "no_repeat_ngram_size" not in fixture.translator.translate_batch.call_args.kwargs
    assert engine.last_diagnostics["first_issues"] == []
    assert engine.last_diagnostics["retry_outcome"] == "not_needed"


def test_diagnostics_are_replaced_for_each_attempted_translation(fixture):
    engine, decode = output_engine(fixture, "m2m100")
    decode.side_effect = ["想" * 15, "请轻轻看看我的内心。", "谢谢。"]
    engine.translate("私の心をそっと覗いてみませんか。", "ja_to_zh")
    previous = engine.last_diagnostics
    assert previous["calls"] == 2

    assert engine.translate("ありがとう。", "ja_to_zh") == "谢谢。"
    assert engine.last_diagnostics is not previous
    assert engine.last_diagnostics["calls"] == 1
    assert engine.last_diagnostics["first_issues"] == []
    assert engine.last_diagnostics["retry_parameters"] == {}
    assert engine.last_diagnostics["retry_outcome"] == "not_needed"


@pytest.mark.parametrize("backend", ["m2m100", "nllb"])
def test_cancellation_between_quality_check_and_retry_prevents_second_native_call(fixture, backend, monkeypatch):
    from fanyiji_next import translation_guard

    stop = threading.Event()
    engine, decode = output_engine(fixture, backend, stop_event=stop)
    decode.return_value = "想" * 15
    original_check = translation_guard.output_quality_issues

    def cancel_after_check(source, target):
        issues = original_check(source, target)
        stop.set()
        return issues

    monkeypatch.setattr(translation_guard, "output_quality_issues", cancel_after_check)
    with pytest.raises(module.TranslationCancelled):
        engine.translate("私の心をそっと覗いてみませんか。", "ja_to_zh")

    assert fixture.translator.translate_batch.call_count == 1
    assert engine.last_diagnostics["calls"] == 1
    assert engine.last_diagnostics["first_issues"] == ["output_repetition"]


@pytest.mark.parametrize("backend", ["m2m100", "nllb"])
def test_cancellation_during_quality_retry_suppresses_retry_decoding(fixture, backend):
    stop = threading.Event()
    engine, decode = output_engine(fixture, backend, stop_event=stop)
    decode.return_value = "想" * 15
    result = fixture.translator.translate_batch.return_value

    def translate(*args, **kwargs):
        if kwargs["beam_size"] == 5:
            stop.set()
        return result

    fixture.translator.translate_batch.side_effect = translate
    with pytest.raises(module.TranslationCancelled):
        engine.translate("私の心をそっと覗いてみませんか。", "ja_to_zh")

    assert fixture.translator.translate_batch.call_count == 2
    assert decode.call_count == 1
    assert engine.last_diagnostics["calls"] == 2


def test_close_waits_until_both_native_attempts_leave_the_shared_lock(fixture):
    engine, decode = output_engine(fixture, "m2m100")
    decode.side_effect = ["想" * 15, "请轻轻看看我的内心。"]
    retry_entered = threading.Event()
    release_retry = threading.Event()
    close_requested = threading.Event()
    output = []
    failures = []
    result = fixture.translator.translate_batch.return_value

    def translate(*args, **kwargs):
        if kwargs["beam_size"] == 5:
            retry_entered.set()
            assert release_retry.wait(3)
            fixture.translator.unload_model.assert_not_called()
        return result

    def run_translation():
        try:
            output.append(engine.translate("私の心をそっと覗いてみませんか。", "ja_to_zh"))
        except BaseException as exc:
            failures.append(exc)

    def run_close():
        close_requested.set()
        engine.close()

    fixture.translator.translate_batch.side_effect = translate
    translation_thread = threading.Thread(target=run_translation)
    close_thread = threading.Thread(target=run_close)
    translation_thread.start()
    try:
        assert retry_entered.wait(3)
        close_thread.start()
        assert close_requested.wait(3)
        fixture.translator.unload_model.assert_not_called()
    finally:
        release_retry.set()
        translation_thread.join(3)
        if close_thread.ident is not None:
            close_thread.join(3)
    assert not translation_thread.is_alive() and not close_thread.is_alive()
    assert failures == []
    assert output == ["请轻轻看看我的内心。"]
    fixture.translator.unload_model.assert_called_once()


def test_nllb_only_replaces_uninitialized_language_slot_not_unknown_body(fixture):
    fixture.tokenizer.encode.return_value = SimpleNamespace(tokens=["<unk>", "▁fixture"])
    fixture.translator.translate_batch.return_value = [SimpleNamespace(hypotheses=[["jpn_Jpan", "▁result"]])]
    engine = make_nllb(fixture, layout="suffix", language="<unk>")
    engine.translate("fixture", "zh_to_ja")
    assert fixture.translator.translate_batch.call_args.args[0] == [["<unk>", "▁fixture", "</s>", "zho_Hans"]]


@pytest.mark.parametrize("backend", ["m2m100", "nllb"])
def test_input_token_limit_accepts_exact_boundary_and_rejects_overflow_before_inference(fixture, backend):
    fixture.settings.translation_backend = backend
    if backend == "m2m100":
        engine = module.create_translation_engine(fixture.settings)
        fixture.spm.encode.return_value = ["piece"] * 510
    else:
        engine = make_nllb(fixture)
        fixture.tokenizer.encode.return_value = SimpleNamespace(tokens=["▁fixture"] * 510)
        fixture.translator.translate_batch.return_value = [SimpleNamespace(hypotheses=[["jpn_Jpan", "▁result"]])]
    engine.translate("fixture", "zh_to_ja")
    assert len(fixture.translator.translate_batch.call_args.args[0][0]) == 512
    fixture.translator.translate_batch.reset_mock()
    if backend == "m2m100":
        fixture.spm.encode.return_value = ["piece"] * 511
    else:
        fixture.tokenizer.encode.return_value = SimpleNamespace(tokens=["▁fixture"] * 511)
    with pytest.raises(RuntimeError, match="512 token"):
        engine.translate("fixture", "zh_to_ja")
    fixture.translator.translate_batch.assert_not_called()


def test_m2m100_automatic_eos_counts_towards_input_limit(fixture):
    (fixture.old / "config.json").write_text('{"add_source_eos": true}', encoding="utf8")
    fixture.spm.encode.return_value = ["piece"] * 511
    engine = module.create_translation_engine(fixture.settings)
    with pytest.raises(RuntimeError, match="512 token"):
        engine.translate("fixture", "zh_to_ja")
    fixture.translator.translate_batch.assert_not_called()


@pytest.mark.parametrize("hypotheses,error", [([], "候选文本"), ([[]], "空文本"), ([['out'] * 512], "长度上限")])
def test_empty_or_truncated_m2m100_output_is_an_explicit_failure(fixture, hypotheses, error):
    fixture.translator.translate_batch.return_value = [SimpleNamespace(hypotheses=hypotheses)]
    fixture.spm.decode.return_value = ""
    engine = module.create_translation_engine(fixture.settings)
    with pytest.raises(RuntimeError, match=error):
        engine.translate("fixture", "zh_to_ja")


@pytest.mark.parametrize("tokens,error", [(["zho_Hans", "▁result"], "目标语言前缀"),
                                         (["jpn_Jpan", "unknown_output"], "不在配套 tokenizer")])
def test_nllb_rejects_wrong_prefix_or_unknown_output_tokens(fixture, tokens, error):
    fixture.translator.translate_batch.return_value = [SimpleNamespace(hypotheses=[tokens])]
    engine = make_nllb(fixture)
    with pytest.raises(RuntimeError, match=error):
        engine.translate("fixture", "zh_to_ja")
    fixture.tokenizer.decode.assert_not_called()


@pytest.mark.parametrize("config", [{"add_source_eos": "false"}, {"decoder_start_token": "<bad>"}, {"eos_token": "<bad>"}])
def test_invalid_model_special_token_config_prevents_loading(fixture, config):
    (fixture.old / "config.json").write_text(json.dumps(config), encoding="utf8")
    with pytest.raises(RuntimeError):
        module.create_translation_engine(fixture.settings)
    fixture.native.assert_not_called()


@pytest.mark.parametrize("config", [{"add_source_eos": True}, {"add_source_bos": True}])
def test_nllb_rejects_implicit_source_special_tokens(fixture, config):
    with pytest.raises(RuntimeError, match="不能自动追加"):
        make_nllb(fixture, config=config)
    fixture.native.assert_not_called()


@pytest.mark.parametrize("damage", ["unknown_processor", "extra_sequence", "missing_language", "wrong_ids"])
def test_nllb_rejects_unrecognized_or_inconsistent_tokenizer_before_native_load(fixture, damage):
    fixture.settings.translation_backend = "nllb"
    data = tokenizer_json()
    if damage == "unknown_processor":
        data["post_processor"]["type"] = "BertProcessing"
    elif damage == "extra_sequence":
        data["post_processor"]["single"].append({"Sequence": {"id": "B"}})
    elif damage == "missing_language":
        fixture.tokenizer.token_to_id.side_effect = lambda token: None if token == "zho_Hans" else TOKEN_IDS.get(token)
    else:
        data["post_processor"]["special_tokens"]["</s>"]["ids"] = [999]
    fixture.tokenizer_path.write_text(json.dumps(data), encoding="utf8")
    with pytest.raises(RuntimeError):
        module.create_translation_engine(fixture.settings)
    fixture.native.assert_not_called()


def test_requested_cuda_compute_profile_is_forwarded_without_cpu_fallback(fixture):
    fixture.settings.translation_device = "cuda"
    fixture.settings.translation_compute_type = "int8_float16"
    engine = module.create_translation_engine(fixture.settings)
    assert fixture.native.call_args.kwargs["device"] == "cuda"
    assert fixture.native.call_args.kwargs["compute_type"] == "int8_float16"
    assert fixture.native.call_args.kwargs["inter_threads"] == 1
    assert engine.backend_info["device"] == "cuda"
    assert engine.backend_info["compute_type"] == "int8_float16"
    fixture.support.assert_called_once_with("cuda")
    native_runtime.configure_cuda_runtime.assert_called_once_with("cuda")


def test_unsupported_native_compute_profile_fails_instead_of_falling_back(fixture):
    fixture.support.return_value = {"float32"}
    with pytest.raises(RuntimeError, match="不会自动回退"):
        module.create_translation_engine(fixture.settings)
    fixture.native.assert_not_called()


def test_pre_cancelled_construction_does_not_load_any_tokenizer_or_native_model(fixture):
    stop = threading.Event()
    stop.set()
    with pytest.raises(module.TranslationCancelled):
        module.create_translation_engine(fixture.settings, stop_event=stop)
    fixture.spm_factory.assert_not_called()
    fixture.native.assert_not_called()


def test_cancellation_after_native_load_releases_partial_model(fixture):
    stop = threading.Event()

    def load(*args, **kwargs):
        stop.set()
        return fixture.translator

    fixture.native.side_effect = load
    with pytest.raises(module.TranslationCancelled):
        module.create_translation_engine(fixture.settings, stop_event=stop)
    fixture.translator.unload_model.assert_called_once()


def test_cancellation_during_translation_suppresses_decoding(fixture):
    stop = threading.Event()
    engine = module.create_translation_engine(fixture.settings, stop_event=stop)

    def translate(*args, **kwargs):
        stop.set()
        return [SimpleNamespace(hypotheses=[["__ja__", "out"]])]

    fixture.translator.translate_batch.side_effect = translate
    with pytest.raises(module.TranslationCancelled):
        engine.translate("fixture", "zh_to_ja")
    fixture.spm.decode.assert_not_called()
    engine.close()
    fixture.translator.unload_model.assert_called_once()


def test_close_is_idempotent_and_closed_engine_cannot_translate(fixture):
    engine = module.create_translation_engine(fixture.settings)
    engine.close()
    engine.close()
    fixture.translator.unload_model.assert_called_once()
    assert engine.tokenizer is None and engine.translator is None
    with pytest.raises(RuntimeError, match="已关闭"):
        engine.translate("fixture", "zh_to_ja")


@pytest.mark.parametrize("text,direction", [("fixture", "en_to_ja"), ("", "zh_to_ja"), ("   ", "zh_to_ja")])
def test_invalid_input_does_not_reach_native_translation(fixture, text, direction):
    engine = module.create_translation_engine(fixture.settings)
    with pytest.raises(ValueError):
        engine.translate(text, direction)
    fixture.translator.translate_batch.assert_not_called()


def test_unknown_backend_is_not_silently_replaced(fixture):
    fixture.settings.translation_backend = "unknown"
    with pytest.raises(ValueError, match="backend"):
        module.create_translation_engine(fixture.settings)
    fixture.native.assert_not_called()


def test_model_name_or_missing_directory_never_resolves_through_network(fixture):
    fixture.settings.translation_model_dir = "facebook/m2m100_418M"
    with pytest.raises(ValueError, match="目录不存在"):
        module.create_translation_engine(fixture.settings)
    fixture.native.assert_not_called()
