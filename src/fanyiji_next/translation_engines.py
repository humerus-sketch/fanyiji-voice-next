"""Local-only CTranslate2 translation adapters; importing this loads no engines."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading


TOKEN_LIMIT = 512
_NLLB_LANGUAGE = re.compile(r"[a-z]{3}_[A-Z][a-z]{3}\Z")


class TranslationCancelled(RuntimeError):
    """A stop request reached a boundary around native translation work."""


def _path(value: str, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"尚未选择{label}。")
    return Path(os.path.expandvars(value)).expanduser()


def _model_folder(value: str) -> Path:
    folder = _path(value, "CTranslate2 翻译模型目录")
    if not folder.is_dir():
        raise ValueError(f"CTranslate2 翻译模型目录不存在：{folder}")
    for name in ("model.bin", "config.json"):
        if not (folder / name).is_file():
            raise ValueError(f"CTranslate2 翻译模型目录缺少 {name}。")
    shared = (folder / "shared_vocabulary.json").is_file()
    separate = all((folder / name).is_file() for name in ("source_vocabulary.json", "target_vocabulary.json"))
    if not (shared or separate):
        raise ValueError("CTranslate2 翻译模型目录缺少配套词表。")
    return folder


def _read_json(path: Path, label: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"无法读取{label}：{exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{label}必须是 JSON object。")
    return data


def _config(folder: Path, backend: str) -> dict:
    config = _read_json(folder / "config.json", " CTranslate2 config.json")
    if type(config.get("add_source_eos", False)) is not bool:
        raise RuntimeError("CTranslate2 config.json 的 add_source_eos 必须为布尔值。")
    if config.get("eos_token", "</s>") != "</s>" or config.get("decoder_start_token", "</s>") not in (None, "</s>"):
        raise RuntimeError(f"翻译模型的特殊 token 与 {backend} 不匹配。")
    if backend == "nllb" and (
        config.get("add_source_bos", False) is not False or config.get("add_source_eos", False)
    ):
        raise RuntimeError("NLLB 需要 Transformers 转换的显式源语言标记/EOS 配置，不能自动追加 BOS/EOS。")
    return config


class _TranslationEngine:
    def __init__(self, settings, stop_event, backend: str) -> None:
        self.translator = None
        self.tokenizer = None
        self._stop_event = stop_event
        self._lock = threading.RLock()
        self._closed = False
        self._backend = backend
        self.last_diagnostics = {}
        self._check_cancelled()
        device = getattr(settings, "translation_device", "cpu")
        compute = getattr(settings, "translation_compute_type", "int8")
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"不支持的翻译设备：{device}")
        if compute not in {"int8", "int8_float16", "float16"}:
            raise ValueError(f"不支持的翻译计算类型：{compute}")
        self.backend_info = {"backend": backend, "device": device, "compute_type": compute}

    def _check_cancelled(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            raise TranslationCancelled("翻译已取消。")

    def _open_translator(self, folder: Path) -> None:
        self._check_cancelled()
        if self.backend_info["device"] == "cuda":
            from .native_runtime import configure_cuda_runtime
            configure_cuda_runtime("cuda")
        import ctranslate2

        self._check_cancelled()
        device = self.backend_info["device"]
        compute = self.backend_info["compute_type"]
        supported = ctranslate2.get_supported_compute_types(device)
        if compute not in supported:
            raise RuntimeError(f"当前 CTranslate2/{device} 不支持 {compute}，不会自动回退设备或计算类型。")
        self._check_cancelled()
        self.translator = ctranslate2.Translator(
            str(folder), device=device, compute_type=compute,
            inter_threads=1, intra_threads=max(1, min(4, os.cpu_count() or 1)),
        )
        self.backend_info.update(model_dir=str(folder), requested_compute_type=compute,
                                 device=getattr(self.translator, "device", device),
                                 compute_type=getattr(self.translator, "compute_type", compute))
        self._check_cancelled()

    def translate(self, text: str, direction: str) -> str:
        with self._lock:
            from .translation_guard import candidate_issues, output_quality_issues

            self.last_diagnostics = {"calls": 0, "issues": [], "attempts": [],
                                     "first_issues": [], "retry_issues": [],
                                     "retry_parameters": {}, "retry_outcome": "not_needed"}
            self._check_cancelled()
            if self._closed:
                raise RuntimeError("翻译引擎已关闭。")
            if direction not in {"zh_to_ja", "ja_to_zh"}:
                raise ValueError(f"不支持的翻译方向：{direction}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("待翻译文本不能为空。")
            source, prefix = self._encode(text, direction)
            self._check_cancelled()
            automatic_eos = int(self._config.get("add_source_eos", False))
            if len(source) + automatic_eos > TOKEN_LIMIT:
                raise RuntimeError("本句超过离线翻译的 512 token 上限；请缩短连续发言后重试。")
            for attempt_index in range(2):
                parameters = {"beam_size": 2, "max_input_length": TOKEN_LIMIT,
                              "max_decoding_length": TOKEN_LIMIT}
                if attempt_index:
                    # Only regenerate a conspicuously broken output. Changing
                    # ordinary decoding could suppress intentional repetition.
                    parameters.update(beam_size=5, no_repeat_ngram_size=3)
                    self.last_diagnostics.update(retry_parameters=dict(parameters), retry_outcome="running")
                self._check_cancelled()
                attempt = {"parameters": dict(parameters), "issues": []}
                self.last_diagnostics["attempts"].append(attempt)
                self.last_diagnostics["calls"] += 1
                results = self.translator.translate_batch([source], target_prefix=[prefix], **parameters)
                self._check_cancelled()
                if not results or not results[0].hypotheses:
                    raise RuntimeError("离线翻译没有返回候选文本，请检查翻译模型。")
                tokens = results[0].hypotheses[0]
                attempt["output_tokens"] = len(tokens)
                if len(tokens) >= TOKEN_LIMIT:
                    raise RuntimeError("离线译文达到长度上限，本句未完整翻译；请缩短连续发言。")
                translated = self._decode(tokens, prefix).strip()
                self._check_cancelled()
                if not translated:
                    raise RuntimeError("离线翻译返回空文本，请检查翻译模型与配套 tokenizer。")
                # Constrained regeneration can remove meaningful repeated
                # numbers or a prohibition. Keep normal decoding unchanged,
                # but do not accept a cleaner-looking retry with those losses.
                issues = list(candidate_issues(text, translated, direction) if attempt_index
                              else output_quality_issues(text, translated))
                attempt["issues"] = issues
                self.last_diagnostics["issues"] = issues
                self.last_diagnostics["retry_issues" if attempt_index else "first_issues"] = issues
                if not issues:
                    if attempt_index:
                        self.last_diagnostics["retry_outcome"] = "accepted"
                    return translated
                self.last_diagnostics["retry_outcome"] = "rejected" if attempt_index else "pending"
            if any(not issue.startswith("output_") for issue in issues):
                raise RuntimeError("重试译文的否定、意向或数值与原文不一致，本句保留原文；请重试或更换翻译引擎。")
            raise RuntimeError("译文出现异常重复或无有效文字，本句保留原文；请检查输入内容或更换翻译引擎。")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            translator, self.translator = self.translator, None
            try:
                if translator is not None:
                    translator.unload_model()
            finally:
                self.tokenizer = None


class _M2M100Engine(_TranslationEngine):
    def __init__(self, settings, stop_event=None) -> None:
        super().__init__(settings, stop_event, "m2m100")
        try:
            folder = _model_folder(settings.translation_model_dir)
            self._config = _config(folder, "m2m100")
            spm = _path(settings.translation_spm, "M2M100 SentencePiece 模型")
            if not spm.is_file():
                raise ValueError(f"M2M100 SentencePiece 模型不存在：{spm}")
            import sentencepiece

            self._check_cancelled()
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(spm))
            self._check_cancelled()
            self._open_translator(folder)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _encode(self, text: str, direction: str):
        source, target = ("zh", "ja") if direction == "zh_to_ja" else ("ja", "zh")
        tokens = [f"__{source}__"] + self.tokenizer.encode(text, out_type=str)
        if not self._config.get("add_source_eos", False):
            tokens.append("</s>")
        prefix = [f"__{target}__"]
        if self._config.get("decoder_start_token", "</s>") is None:
            prefix.insert(0, "</s>")
        return tokens, prefix

    def _decode(self, tokens, prefix):
        ignored = {prefix[-1], "</s>", "<s>"}
        return self.tokenizer.decode([token for token in tokens if token not in ignored])


def _nllb_template(data: dict, tokenizer) -> tuple[str, ...]:
    processor = data.get("post_processor")
    if not isinstance(processor, dict) or processor.get("type") != "TemplateProcessing":
        raise RuntimeError("NLLB tokenizer.json 缺少可识别的 TemplateProcessing。")
    template = []
    single = processor.get("single")
    if not isinstance(single, list):
        raise RuntimeError("NLLB tokenizer.json 的 single 模板无效。")
    for item in single:
        if (isinstance(item, dict) and set(item) == {"Sequence"}
                and isinstance(item["Sequence"], dict) and item["Sequence"].get("id") == "A"):
            template.append("$A")
        elif (isinstance(item, dict) and set(item) == {"SpecialToken"}
              and isinstance(item["SpecialToken"], dict) and isinstance(item["SpecialToken"].get("id"), str)):
            template.append(item["SpecialToken"]["id"])
        else:
            raise RuntimeError("NLLB tokenizer.json 的 single 模板不支持。")
    languages = [item for item in template if _NLLB_LANGUAGE.fullmatch(item) or item == "<unk>"]
    if (len(languages) != 1
            or template not in ([languages[0], "$A", "</s>"], ["$A", "</s>", languages[0]])
            or (languages[0] == "<unk>" and template != ["$A", "</s>", "<unk>"])):
        raise RuntimeError("NLLB 源模板必须是语言码 + 文本 + EOS，或文本 + EOS + 语言码。")
    # The published tokenizer has src_lang=null and stores <unk> specifically
    # in the suffix language slot. Replace that slot only, never body <unk>s.
    for token in {"</s>", "zho_Hans", "jpn_Jpan", languages[0]}:
        if tokenizer.token_to_id(token) is None:
            raise RuntimeError(f"NLLB tokenizer 缺少语言或特殊 token：{token}")
    declared = processor.get("special_tokens", {})
    if not isinstance(declared, dict):
        raise RuntimeError("NLLB tokenizer 缺少后处理 special_tokens 定义。")
    for token in (languages[0], "</s>"):
        entry = declared.get(token, {})
        if not isinstance(entry, dict) or entry.get("tokens") != [token] or entry.get("ids") != [tokenizer.token_to_id(token)]:
            raise RuntimeError(f"NLLB tokenizer 后处理 token 定义不匹配：{token}")
    return tuple("$LANG" if item == languages[0] else item for item in template)


class _NllbEngine(_TranslationEngine):
    def __init__(self, settings, stop_event=None) -> None:
        super().__init__(settings, stop_event, "nllb")
        try:
            folder = _model_folder(settings.nllb_model_dir)
            self._config = _config(folder, "nllb")
            directory = _path(settings.nllb_tokenizer_dir, "NLLB tokenizer 目录")
            path = directory / "tokenizer.json"
            data = _read_json(path, " NLLB tokenizer.json")
            import tokenizers

            self._check_cancelled()
            self.tokenizer = tokenizers.Tokenizer.from_file(str(path))
            # A serialized truncation policy must not hide an overlong input.
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            self._template = _nllb_template(data, self.tokenizer)
            del data
            self.backend_info.update(tokenizer_dir=str(directory), source_layout=" ".join(self._template))
            self._check_cancelled()
            self._open_translator(folder)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _encode(self, text: str, direction: str):
        source, target = ("zho_Hans", "jpn_Jpan") if direction == "zh_to_ja" else ("jpn_Jpan", "zho_Hans")
        pieces = self.tokenizer.encode(text, add_special_tokens=False).tokens
        if not pieces:
            raise RuntimeError("NLLB tokenizer 未生成正文 token。")
        tokens = []
        for item in self._template:
            tokens.extend(pieces if item == "$A" else [source if item == "$LANG" else item])
        prefix = [target]
        if self._config.get("decoder_start_token", "</s>") is None:
            prefix.insert(0, "</s>")
        return tokens, prefix

    def _decode(self, tokens, prefix):
        if tokens[:len(prefix)] != prefix:
            raise RuntimeError("NLLB 未返回指定的目标语言前缀。")
        ids = []
        for token in tokens[len(prefix):]:
            index = self.tokenizer.token_to_id(token)
            if index is None:
                raise RuntimeError(f"NLLB 输出 token 不在配套 tokenizer 中：{token}")
            ids.append(index)
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def create_translation_engine(settings, *, stop_event=None):
    """Load one local backend now; never resolve a model name through a hub."""
    backend = getattr(settings, "translation_backend", "m2m100")
    if backend == "translategemma":
        from .gemma_translation import GemmaTranslationEngine
        return GemmaTranslationEngine(settings, stop_event=stop_event, prompt_variant="current",
                                      prompt_transport="token_ids", repair_policy="source_structure_v5",
                                      application_mode=True)
    if backend == "m2m100":
        return _M2M100Engine(settings, stop_event)
    if backend == "nllb":
        return _NllbEngine(settings, stop_event)
    if backend == "hymt":
        from .hymt_translation import HyTranslationEngine
        return HyTranslationEngine(settings, stop_event)
    raise ValueError(f"不支持的翻译 backend：{backend}")
