"""TranslateGemma text-only runtime, with bounded per-source conversation history."""
from __future__ import annotations

from collections import deque
import json
import threading

from .hymt_runtime import LlamaServerRuntime, HyRuntimeCancelled
from .translation_engines import TranslationCancelled

LANGUAGES = {"zh": "Chinese", "ja": "Japanese", "en": "English", "ko": "Korean",
             "yue": "Cantonese", "fr": "French", "de": "German", "es": "Spanish",
             "ru": "Russian", "pt": "Portuguese", "it": "Italian"}
MAX_OUTPUT_TOKENS = 256


def translation_prompt(text, source, target, history=()):
    target_name = LANGUAGES.get(target, target)
    source_name = LANGUAGES.get(source, source) if source else "original-language"
    source_label = f"{source_name} ({source})" if source else source_name
    # This is an explicit context extension of the direct-translation template.
    # Do not concatenate and retranslate previous subtitles or slice model output.
    # Native text template from ggml-org's google-translategemma Jinja. The
    # reference-only block is an explicit extension, not a validated quality fix.
    instruction = (f"You are a professional {source_label} to {target_name} ({target}) translator. "
                   "Your goal is to accurately convey the meaning and "
                   f"nuances of the original {source_name} text while adhering to {target_name} grammar, "
                   "vocabulary, and cultural sensitivities.\n")
    if history:
        instruction += "PREVIOUS SOURCE TEXT (reference only; do not translate or repeat):\n"
        instruction += json.dumps(list(history)[-2:], ensure_ascii=False) + "\nEND OF REFERENCE\n"
    instruction += ("Preserve incomplete sentences, negation, numbers and speaker intent; "
                    "do not finish an unfinished thought.\n"
                    f"Produce only the {target_name} translation, without any additional explanations or "
                    f"commentary. Please translate the following {source_name} text into {target_name}:\n\n\n" + text)
    return "<bos><start_of_turn>user\n" + instruction + "<end_of_turn>\n<start_of_turn>model\n"


class GemmaTranslationEngine:
    def __init__(self, settings, *, stop_event=None, runtime_factory=LlamaServerRuntime):
        self._stop_event = stop_event
        self._lock = threading.RLock()
        self._history = {}
        self._closed = False
        self._check_cancelled()
        self.runtime = runtime_factory(settings.hymt_server_exe, settings.gemma_model_file,
                                       device=settings.translation_device, stop_event=stop_event,
                                       ctx_size=2048)
        try:
            self.runtime.start()
        except HyRuntimeCancelled as exc:
            self.runtime.close()
            raise TranslationCancelled(str(exc)) from exc
        except BaseException:
            self.runtime.close()
            raise
        self.backend_info = {"backend": "translategemma", "quantization": "Q4_K_M",
                             "device": settings.translation_device, "context_sentences": 2,
                             "context_policy": "source_only; no semantic acceptance claimed"}

    def translate(self, text, direction, *, context_id="default", source_language=None):
        if direction not in {"zh_to_ja", "ja_to_zh"}:
            raise ValueError("翻译方向必须为 zh_to_ja 或 ja_to_zh。")
        if not isinstance(text, str):
            raise ValueError("待翻译内容必须是文本。")
        target = "ja" if direction == "zh_to_ja" else "zh"
        with self._lock:
            if self._closed:
                raise RuntimeError("翻译引擎已关闭。")
            self._check_cancelled()
            if not text.strip():
                return ""
            # Language switches cannot carry one language's history into another.
            key = (context_id, source_language, target)
            for other in tuple(self._history):
                if other[0] == context_id and other != key:
                    del self._history[other]
            history = self._history.setdefault(key, deque(maxlen=2))
            if source_language == target:
                history.append(text)
                return text
            prompt = translation_prompt(text, source_language, target, list(history))
            try:
                tokenized = self.runtime.post_json("/tokenize", {"content": prompt, "add_special": False})
                if self._token_count(tokenized) > 1750:
                    # Trim only old context; never silently truncate the current sentence.
                    prompt = translation_prompt(text, source_language, target)
                    tokenized = self.runtime.post_json("/tokenize", {"content": prompt, "add_special": False})
                    if self._token_count(tokenized) > 1750:
                        raise RuntimeError("当前句超过翻译模型上下文长度，原文已保留。")
                self._check_cancelled()
                response = self.runtime.post_json("/completion", {
                    "prompt": prompt, "n_predict": MAX_OUTPUT_TOKENS, "temperature": 0.0,
                    "stop": ["<end_of_turn>"], "cache_prompt": True})
            except HyRuntimeCancelled as exc:
                raise TranslationCancelled(str(exc)) from exc
            self._check_cancelled()
            count = response.get("tokens_predicted")
            hit_limit = isinstance(count, int) and count >= MAX_OUTPUT_TOKENS
            complete_stop = response.get("stopped_eos") is True or response.get("stopped_word") is True
            if (response.get("stopped_limit") or response.get("truncated")
                    or response.get("stop_type") in {"limit", "length", "max_tokens"}
                    or (hit_limit and not complete_stop)):
                raise RuntimeError("翻译生成未完整结束，原文已保留。")
            value = response.get("content")
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError("翻译模型没有返回译文，原文已保留。")
            history.append(text)
            return value.strip()

    def _check_cancelled(self):
        if self._stop_event is not None and self._stop_event.is_set():
            raise TranslationCancelled("翻译已取消。")

    @staticmethod
    def _token_count(response):
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise RuntimeError("翻译运行库未返回有效的 token 计数。")
        return len(tokens)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._history.clear()
            self.runtime.close()
