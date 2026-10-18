"""Local-only A/B recognition. Importing this module never loads weights."""
from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path

from .recognition_engines import _RecognizerBase, RecognitionCancelled

MAX_NEW_TOKENS = 512


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    language: str | None = None


def normalize_language(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    language = value.strip().lower().replace("_", "-")
    names = {"chinese": "zh", "mandarin": "zh", "中文": "zh", "japanese": "ja",
             "日文": "ja", "日语": "ja", "english": "en", "英文": "en",
             "korean": "ko", "cantonese": "yue", "french": "fr", "german": "de",
             "spanish": "es", "russian": "ru", "portuguese": "pt", "italian": "it"}
    if language in names:
        return names[language]
    if len(language.split("-")[0]) in (2, 3) and language.replace("-", "").isalpha():
        return language.split("-")[0]
    return None


class ABRecognitionEngine(_RecognizerBase):
    def __init__(self, settings, *, stop_event=None):
        super().__init__(stop_event)
        self._model = self._processor = None
        self._backend = settings.recognition_backend
        self._device = settings.recognition_device
        self._check_cancelled()
        if self._backend not in {"qwen3_asr", "fun_asr"}:
            raise ValueError("不支持的 A/B 识别后端。")
        if self._device not in {"cpu", "cuda"}:
            raise ValueError("A/B 识别设备必须为 cpu 或 cuda。")
        folder_value = settings.qwen_asr_model_dir if self._backend == "qwen3_asr" else settings.fun_asr_model_dir
        if not isinstance(folder_value, (str, Path)) or not str(folder_value).strip():
            raise RuntimeError("尚未选择 A/B 本地识别模型，请先完成模型下载。")
        folder = Path(folder_value).absolute()
        if not folder.is_dir():
            raise RuntimeError("尚未下载所选识别模型，请打开“模型下载”准备方案 A 或 B。")
        try:
            import torch
            self._torch = torch
            if self._device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("未检测到可用的 CUDA 识别运行库，请检查 NVIDIA 驱动或选择 CPU。")
            if self._backend == "qwen3_asr":
                from transformers import AutoProcessor, AutoModelForMultimodalLM
                self._processor = AutoProcessor.from_pretrained(
                    str(folder), local_files_only=True, trust_remote_code=False)
                dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    str(folder), local_files_only=True, trust_remote_code=False, dtype=dtype,
                    attn_implementation="sdpa").to(self._device).eval()
            else:
                from funasr import AutoModel
                llm_folder = folder / "Qwen3-0.6B"
                required = (folder / "model.pt", folder / "config.yaml",
                            llm_folder / "config.json", llm_folder / "tokenizer.json",
                            llm_folder / "tokenizer_config.json", folder / "multilingual.tiktoken")
                if any(not path.is_file() for path in required):
                    raise RuntimeError("Fun-ASR 本地模型或同仓库 Qwen3 配置不完整，请完成方案 B 下载校验。")
                # Model code comes from the pinned installed runtime, never from a hub.
                self._model = AutoModel(model=str(folder), device=self._device, disable_update=True,
                                        trust_remote_code=False, local_files_only=True,
                                        check_latest=False, hub="hf", init_param=str(folder / "model.pt"),
                                        llm_conf={"init_param_path": str(llm_folder),
                                                  "llm_dtype": "bf16" if self._device == "cuda" else "fp32"},
                                        tokenizer_conf={"init_param_path": str(llm_folder)},
                                        ctc_tokenizer_conf={"vocab_path": str(folder / "multilingual.tiktoken"),
                                                            "is_multilingual": True, "num_languages": 8749},
                                        llm_dtype="bf16" if self._device == "cuda" else "fp32")
            self.backend_info = {"backend": self._backend, "device": self._device,
                                 "model_path": str(folder), "local_files_only": True,
                                 "source_language": "auto", "sample_rate": 16000}
            self._check_cancelled()
        except ImportError as exc:
            self.close()
            raise RuntimeError("此程序缺少 A/B 识别运行库，请使用包含 A/B 运行库的新版本；模型下载不会安装或升级系统环境。") from exc
        except BaseException:
            self.close()
            raise

    def recognize_result(self, samples) -> RecognitionResult:
        with self._lock:
            self._check_open()
            import numpy as np
            waveform = np.asarray(samples, dtype=np.float32)
            if waveform.ndim != 1 or not np.isfinite(waveform).all():
                raise ValueError("识别输入必须为16kHz有限数值单声道波形。")
            if not len(waveform):
                return RecognitionResult("")
            from transformers import StoppingCriteria, StoppingCriteriaList
            owner = self

            class StopRequested(StoppingCriteria):
                calls = 0
                last_token = None

                def __call__(self, input_ids, scores, **kwargs):
                    self.calls += 1
                    # Greedy single-sequence decoding invokes this once per new
                    # token. Do not synchronize a GPU tensor on every token.
                    if self.calls >= MAX_NEW_TOKENS and input_ids.shape[1]:
                        self.last_token = int(input_ids[0, -1])
                    return owner._stop_event is not None and owner._stop_event.is_set()

            stopper = StopRequested()
            criteria = StoppingCriteriaList([stopper])
            with self._torch.inference_mode():
                if self._backend == "qwen3_asr":
                    inputs = self._processor.apply_transcription_request(audio=waveform)
                    inputs = inputs.to(self._model.device, self._model.dtype)
                    self._check_cancelled()
                    output = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                                 stopping_criteria=criteria)
                    self._check_cancelled()
                    generated = output[:, inputs["input_ids"].shape[1]:]
                    if generated.shape[1] >= MAX_NEW_TOKENS and int(generated[0, -1]) not in self._eos_ids(self._model):
                        raise RuntimeError("识别达到生成长度上限，未交付可能截断的字幕。")
                    decoded = self._processor.decode(generated, return_format="parsed")
                    if not isinstance(decoded, (list, tuple)) or len(decoded) != 1 or not isinstance(decoded[0], dict):
                        raise RuntimeError("Qwen 识别模型未返回有效的解析结果。")
                    parsed = decoded[0]
                    text = parsed.get("transcription", "")
                    language = normalize_language(parsed.get("language"))
                else:
                    # Fun-ASR-Nano's chat template accepts Tensor or file path;
                    # NumPy input reaches it unchanged and produces no template.
                    audio_tensor = self._torch.from_numpy(np.ascontiguousarray(waveform))
                    # Its frontend supplies fs itself; forwarding fs here would
                    # pass that keyword twice inside data_load_speech.
                    result = self._model.generate(input=audio_tensor, cache={}, batch_size=1,
                                                  language=None, itn=False, disable_pbar=True,
                                                  max_length=MAX_NEW_TOKENS,
                                                  llm_kwargs={"do_sample": False, "stopping_criteria": criteria})
                    self._check_cancelled()
                    llm = getattr(getattr(self._model, "model", None), "llm", None)
                    if stopper.calls >= MAX_NEW_TOKENS and stopper.last_token not in self._eos_ids(llm):
                        raise RuntimeError("识别达到生成长度上限，未交付可能截断的字幕。")
                    if not isinstance(result, (list, tuple)) or (result and not isinstance(result[0], dict)):
                        raise RuntimeError("Fun-ASR 未返回有效的解析结果。")
                    item = result[0] if result else {}
                    text = item.get("text", "")
                    # Fun-ASR may omit language metadata. Do not infer Japanese from
                    # settings, previous audio, or Chinese/Japanese shared characters.
                    language = normalize_language(item.get("language"))
            self._check_cancelled()
            if not isinstance(text, str):
                raise RuntimeError("识别模型未返回有效文本。")
            return RecognitionResult(text.strip(), language)

    @staticmethod
    def _eos_ids(model) -> set[int]:
        result = set()
        for name in ("generation_config", "config"):
            value = getattr(getattr(model, name, None), "eos_token_id", None)
            values = value if isinstance(value, (tuple, list, set)) else (value,)
            result.update(item for item in values if isinstance(item, int) and not isinstance(item, bool))
        return result

    def recognize(self, samples, language=None):
        return self.recognize_result(samples).text

    def close(self):
        with self._lock:
            self._closed = True
            self._model = self._processor = None
            gc.collect()
            torch = getattr(self, "_torch", None)
            if torch is not None and self._device == "cuda":
                torch.cuda.empty_cache()
