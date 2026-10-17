"""Current-sentence-first local Hy-MT translation with bounded source context."""
from __future__ import annotations

from collections import deque
from difflib import SequenceMatcher
import re
import threading
import time
import unicodedata

from .hymt_runtime import HyRuntimeCancelled, LlamaServerRuntime
from .translation_engines import TranslationCancelled
from .translation_guard import candidate_issues, context_grounding_issues, prepare_source, source_fragment_issue


CTX_SIZE = 2048
MAX_OUTPUT = 256
MAX_INPUT = CTX_SIZE - MAX_OUTPUT - 16
PARAGRAPH_OUTPUT_OVERHEAD = 48
PARAGRAPH_DELIMITERS = ("§", "¶", "¤", "¦")
PARAGRAPH_RULE = (
    "请将以下文本准确翻译为中文。\n"
    "你必须在译文中保留等量的分隔符，绝对不可遗漏、转义或翻译该符号，并注意分隔符的位置。\n\n"
)
PARAMETERS = {
    "temperature": 0.0, "seed": 42, "top_k": 20, "top_p": .6,
    "min_p": 0.0, "repeat_penalty": 1.05, "repeat_last_n": CTX_SIZE,
    "samplers": ["penalties", "temperature", "top_k", "top_p"],
    "n_predict": MAX_OUTPUT, "cache_prompt": False, "stream": True,
}
CONTEXT_RULE = (
    "你是中日翻译器。只翻译当前用户提供的文本。以下上文仅用于消解指代、省略和术语。"
    "当前句是唯一待翻译内容；与上文冲突时必须以当前句为准。"
    "严格保留当前句的否定、数字、条件、人物和动作方向，不要翻译上文或添加解释。\n上文：\n"
)


class _PromptTooLong(RuntimeError):
    pass


class _IncompleteGeneration(RuntimeError):
    pass


class _ParagraphAlignmentError(RuntimeError):
    def __init__(self, issue):
        super().__init__("Hy-MT 段落译文未能与原文对齐，本句保留原文，请重试。")
        self.issue = issue


def _paragraph_delimiter(sources):
    return next((delimiter for delimiter in PARAGRAPH_DELIMITERS
                 if all(delimiter not in source for source in sources)), None)


def _current_paragraph(output, delimiter, source_count):
    if output.count(delimiter) != source_count - 1:
        raise _ParagraphAlignmentError("paragraph_count_mismatch")
    paragraphs = [part.strip() for part in output.split(delimiter)]
    if len(paragraphs) != source_count or any(not part for part in paragraphs):
        raise _ParagraphAlignmentError("paragraph_empty_segment")
    return paragraphs[-1]


def _context_term_hints(prior, source):
    # These narrow lexical senses need explicit source evidence. They are
    # translation hints only; current negation and grammar remain model input.
    sources = list(prior[-2:]) + [source]
    hints = []
    hp_evidence = any(re.search(r"HP|ＨＰ|体力ゲージ", text, re.I)
                      and re.search(r"減[っりるら]|削[られっるり]|ダメージ", text) for text in sources)
    hp_literal = re.search(r"氷|雪|蝋|ろうそく|ロウソク|ワックス|チョコ|アイス|バター", source)
    if "溶ける" in source and hp_evidence and not hp_literal:
        hints.append(("hp_loss", "溶ける", "掉血"))

    context = "\n".join(sources)
    live_time = (re.search(r"配信|生放送", context)
                 and re.search(r"終了|残り|あと", context)
                 and re.search(r"(?:\d+|[零〇一二三四五六七八九十百]+)(?:分|秒|時間)", context)
                 and re.search(r"コーナー|進行|急[ぐい]|間に合|時間がない|時間切れ", context))
    roll_word = re.search(r"巻いて|巻く", source)
    roll_literal = re.search(r"糸|釣り|釣竿|線|ケーブル|コード|袖|マフラー|海苔|寿司|包帯|髪", source)
    if roll_word and live_time and not roll_literal:
        hints.append(("hurry_progress", roll_word.group(), "加快进度"))

    mesh_evidence = any(re.search(r"網|ふるい|篩|茶こし|こし器|ざる|ザル", text) for text in sources)
    eye_literal = re.search(r"瞳|眼|目薬|まつげ|顔|人の目|子の目|私の目|彼の目|彼女の目|両目", source)
    if "目" in source and "粗い" in source and mesh_evidence and not eye_literal:
        hints.extend((("mesh_opening_size", "目", "网眼"), ("mesh_opening_size", "粗い", "偏大")))
    return hints


def _rejected_translation(issues):
    if any(issue.startswith("output_") for issue in issues):
        return RuntimeError("译文出现异常重复或乱码，本句保留原文，请重试。")
    if "ungrounded_time_reference" in issues:
        return RuntimeError("译文加入了当前句及上文未确认的时段，本句保留原文，请重试。")
    return RuntimeError("译文的否定、意向或数值未能与原文保持一致，本句保留原文，请重试。")


def _echo_text(text):
    # Digits alone, punctuation and spacing cannot establish a sentence echo.
    return "".join(char for char in unicodedata.normalize("NFKC", text)
                   if unicodedata.category(char).startswith("L"))


def _context_echo(source, target, successful):
    current_source = _echo_text(source)
    candidate = _echo_text(target)
    if len(candidate) < 12:
        return False
    for old_source, old_target in successful:
        if SequenceMatcher(None, current_source, _echo_text(old_source), autojunk=False).ratio() >= .75:
            continue
        previous = _echo_text(old_target)
        if len(previous) < 12:
            continue
        match = SequenceMatcher(None, candidate, previous, autojunk=False).find_longest_match()
        if match.size >= 12 and match.size * 5 >= len(previous) * 3 and match.size * 5 >= len(candidate) * 2:
            return True
    return False


class HyTranslationEngine:
    def __init__(self, settings, stop_event=None):
        self._stop_event = stop_event
        self._lock = threading.RLock()
        self._closed = False
        self._history: dict[tuple[str, str], deque[str]] = {}
        # Successful targets are a validation reference only, never prompt input.
        self._successful_history: dict[tuple[str, str], deque[tuple[str, str]]] = {}
        self.last_diagnostics = {}
        self.runtime = None
        self._check_cancelled()
        try:
            self.runtime = LlamaServerRuntime(
                settings.hymt_server_exe, settings.hymt_model_file,
                device="cuda", stop_event=stop_event, ctx_size=CTX_SIZE,
            )
            self.runtime.start()
            self.backend_info = {
                **self.runtime.backend_info, "backend": "hymt", "quantization": "Q8_0",
                "context_sentences": 2, "context_policy": "directional_source_context",
                "context_policies": {"ja_to_zh": "paragraph_source_context",
                                     "zh_to_ja": "current_draft_anchor"},
                "temperature": PARAMETERS["temperature"], "local_files_only": True,
            }
            self._check_cancelled()
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                exc.add_note(f"Hy-MT 本地服务收尾失败：{type(cleanup).__name__}")
            if isinstance(exc, HyRuntimeCancelled):
                raise TranslationCancelled("翻译已取消。") from exc
            raise

    def _check_cancelled(self):
        if self._stop_event is not None and self._stop_event.is_set():
            raise TranslationCancelled("翻译已取消。")

    def _generate(self, messages, *, max_output=MAX_OUTPUT):
        self._check_cancelled()
        if type(max_output) is not int or not 1 <= max_output < CTX_SIZE - 16:
            raise ValueError("Hy-MT 输出 token 预算无效。")
        rendered = self.runtime.post_json("/apply-template", {
            "messages": messages, "add_generation_prompt": True,
        }).get("prompt")
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError("Hy-MT 聊天模板没有返回有效文本。")
        tokens = self.runtime.post_json("/tokenize", {
            "content": rendered, "add_special": True, "parse_special": True,
        }).get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("Hy-MT 没有返回有效的输入 token 数。")
        input_budget = CTX_SIZE - max_output - 16
        if len(tokens) > input_budget:
            raise _PromptTooLong("当前句超过本地翻译长度上限；请缩短连续发言后重试。")
        pieces = []
        final = None
        started = time.perf_counter()
        first_ms = None
        # Count completion attempts even when decoding truncates or transport
        # fails. Template/token-budget failures never start a completion.
        self.last_diagnostics["calls"] = self.last_diagnostics.get("calls", 0) + 1
        info = {"input_tokens": len(tokens), "input_budget": input_budget, "output_budget": max_output}
        self.last_diagnostics.setdefault("generation_attempts", []).append(info)
        for event in self.runtime.iter_completion({**PARAMETERS, "n_predict": max_output, "prompt": rendered}):
            self._check_cancelled()
            content = event.get("content", "")
            if not isinstance(content, str):
                raise RuntimeError("Hy-MT 返回了无效的译文数据。")
            if content:
                pieces.append(content)
                if first_ms is None and content.strip():
                    first_ms = (time.perf_counter()-started)*1000
            if event.get("stop"):
                final = event
        self._check_cancelled()
        output = "".join(pieces).strip()
        info.update(ttft_generation_ms=first_ms, timings=final.get("timings") if final else None,
                    stop_type=final.get("stop_type") if final else None,
                    output_tokens=final.get("tokens_predicted") if final else None)
        if (final is None or not output or final.get("truncated")
                or final.get("stop_type") not in {"eos", "word"}):
            raise _IncompleteGeneration("Hy-MT 本句译文未完整生成；请缩短连续发言后重试。")
        self.last_diagnostics["completed_calls"] = self.last_diagnostics.get("completed_calls", 0) + 1
        return output, info

    def translate(self, text: str, direction: str, *, context_id: str = "default") -> str:
        with self._lock:
            self._check_cancelled()
            if self._closed:
                raise RuntimeError("翻译引擎已关闭。")
            if direction not in {"zh_to_ja", "ja_to_zh"}:
                raise ValueError(f"不支持的翻译方向：{direction}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("待翻译文本不能为空。")
            if not isinstance(context_id, str) or not context_id:
                raise ValueError("上文声道标识不能为空。")
            key = (context_id, direction)
            history = self._history.setdefault(key, deque(maxlen=2))
            prior = list(history)
            source = prepare_source(text.strip())
            language = "日语" if direction == "zh_to_ja" else "中文"
            user = {"role": "user", "content": (
                f"将以下文本翻译为{language}，注意只需要输出翻译后的结果，不要额外解释：\n" + source)}
            self.last_diagnostics = {"history_available": len(prior), "history_used": 0,
                                     "calls": 0, "completed_calls": 0, "fallback": "", "issues": [],
                                     "source_normalized": source != text.strip()}
            try:
                fragment_issue = source_fragment_issue(text.strip(), direction)
                if fragment_issue:
                    # Do not ask MT to invent semantics for an isolated beat.
                    # The source and each occurrence remain in the session.
                    self.last_diagnostics.update(issues=[fragment_issue], fallback="source_preserved")
                    return text.strip() + "（音节含义待确认）"
                if direction == "ja_to_zh" and prior:
                    successful = list(self._successful_history.get(key, ()))
                    translated = self._translate_japanese_context(source, prior, user, successful)
                    self._remember_success(key, source, translated)
                    return translated
                # A context-free draft anchors facts in the current sentence;
                # the second pass may use history to resolve language ambiguity.
                draft, info = self._generate([user])
                self.last_diagnostics.update(draft_info=info)
                draft_issues = list(candidate_issues(source, draft, direction))
                if direction == "ja_to_zh":
                    draft_issues.extend(context_grounding_issues(source, draft, ()))
                self.last_diagnostics["draft_issues"] = list(draft_issues)
                if not prior:
                    if draft_issues:
                        raise _rejected_translation(draft_issues)
                    if direction == "ja_to_zh":
                        self._remember_success(key, source, draft)
                    return draft
                context = prior[:]
                while context:
                    system = CONTEXT_RULE + "\n".join(prepare_source(x) for x in context)
                    system += ("\n\n当前句的独立初译：\n" + draft
                               + "\n请保留当前句的否定、数字、条件和动作方向，只根据上文修正指代、省略和词义。")
                    try:
                        translated, info = self._generate([{"role": "system", "content": system}, user])
                    except _PromptTooLong:
                        # Drop whole older utterances; never truncate the current
                        # sentence or pretend a clipped history was complete.
                        context.pop(0)
                        continue
                    self.last_diagnostics.update(history_used=len(context), context_info=info)
                    issues = candidate_issues(source, translated, direction)
                    if not issues:
                        return translated
                    self.last_diagnostics.update(issues=list(issues), fallback="current_sentence_draft")
                    if not draft_issues:
                        return draft
                    raise _rejected_translation(draft_issues)
                self.last_diagnostics["fallback"] = "context_over_budget"
                if not draft_issues:
                    return draft
                raise _rejected_translation(draft_issues)
            except HyRuntimeCancelled as exc:
                raise TranslationCancelled("翻译已取消。") from exc
            finally:
                # History contains recognized source utterances, even if their
                # translation failed; never feed previous target guesses back.
                if self._stop_event is None or not self._stop_event.is_set():
                    history.append(text.strip())

    def _remember_success(self, key, source, target):
        self._check_cancelled()
        self._successful_history.setdefault(key, deque(maxlen=2)).append((source, target))

    def _japanese_candidate_issues(self, source, target, successful, prior):
        issues = list(candidate_issues(source, target, "ja_to_zh"))
        issues.extend(context_grounding_issues(source, target, prior))
        if _context_echo(source, target, successful):
            issues.append("context_echo")
        return issues

    def _translate_japanese_context(self, source, prior, current_user, successful):
        # Translate continuous source paragraphs together, but never publish
        # previous translations or the delimiter instruction as current text.
        context = [prepare_source(text) for text in prior]
        self.last_diagnostics.update(context_term_senses=[], context_term_count=0)
        while context:
            sources = context + [source]
            delimiter = _paragraph_delimiter(sources)
            if delimiter is None:
                self.last_diagnostics.update(issues=["paragraph_delimiter_unavailable"],
                                             fallback="current_sentence_draft")
                break
            output_budget = len(sources) * MAX_OUTPUT + PARAGRAPH_OUTPUT_OVERHEAD
            alignment = {"source_paragraphs": len(sources), "expected_delimiters": len(context),
                         "delimiter_index": PARAGRAPH_DELIMITERS.index(delimiter),
                         "validated": False}
            self.last_diagnostics["paragraph_alignment"] = alignment
            hints = _context_term_hints(context, source)
            prefix = ("参考下面的翻译：\n" + "\n".join(
                f"{term} 翻译成 {meaning}" for _sense, term, meaning in hints) + "\n") if hints else ""
            before_calls = self.last_diagnostics["calls"]
            try:
                output, info = self._generate([{"role": "user", "content": (
                    prefix + PARAGRAPH_RULE + ("\n" + delimiter + "\n").join(sources))}], max_output=output_budget)
            except _PromptTooLong:
                context.pop(0)
                continue
            except _IncompleteGeneration:
                self.last_diagnostics.update(issues=["paragraph_incomplete"], fallback="current_sentence_draft")
                break
            finally:
                if self.last_diagnostics["calls"] > before_calls:
                    self.last_diagnostics["history_used"] = len(sources) - 1
                    self.last_diagnostics.update(context_term_senses=list(dict.fromkeys(sense for sense, _term, _meaning in hints)),
                                                 context_term_count=len(hints))
            self.last_diagnostics.update(history_used=len(context), context_info=info)
            alignment["actual_delimiters"] = output.count(delimiter)
            try:
                translated = _current_paragraph(output, delimiter, len(sources))
            except _ParagraphAlignmentError as exc:
                self.last_diagnostics.update(issues=[exc.issue], fallback="current_sentence_draft")
                break
            alignment["validated"] = True
            actual_sources = set(context)
            references = [(old_source, old_target) for old_source, old_target in successful
                          if old_source in actual_sources]
            issues = self._japanese_candidate_issues(source, translated, references, context)
            if not issues:
                return translated
            self.last_diagnostics.update(issues=list(issues), fallback="current_sentence_draft")
            break
        else:
            self.last_diagnostics["fallback"] = "context_over_budget"
        draft, info = self._generate([current_user])
        self.last_diagnostics.update(draft_info=info)
        # The independent request has no prior source or target prompt input.
        # A valid paraphrase can share a translation with a previous sentence;
        # historical target similarity alone cannot reject this independent pass.
        issues = candidate_issues(source, draft, "ja_to_zh")
        issues = list(issues) + list(context_grounding_issues(source, draft, ()))
        self.last_diagnostics["draft_issues"] = list(issues)
        if issues:
            raise _rejected_translation(issues)
        return draft

    def close(self):
        with self._lock:
            self._closed = True
            self._history.clear()
            self._successful_history.clear()
            if self.runtime is not None:
                self.runtime.close()
