"""Bounded, source-aligned CPU preprocessing for one system-audio stream.

The owner worker serializes calls. No devices, models, native runtimes or numpy
are loaded by importing this module. Cancellation is checked around native
calls; an in-flight ONNX run cannot be interrupted safely.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading
from typing import Any


OUTPUT_SAMPLE_RATE = 16000
MDX_SAMPLE_RATE = 44100
MDX_FFT_SIZE = 7680
MDX_HOP_SIZE = 1024
MDX_FREQUENCY_BINS = 3072
MDX_FRAMES = 256
MDX_WINDOW_SAMPLES = MDX_HOP_SIZE * (MDX_FRAMES - 1)
MDX_TRIM_SAMPLES = MDX_FFT_SIZE // 2
MDX_STEP_SAMPLES = MDX_WINDOW_SAMPLES - 2 * MDX_TRIM_SAMPLES
PROCESSING_MODES = frozenset(("raw", "gtcrn", "gtcrn_preserve", "mdx", "mdx_gtcrn"))
_DIRECT_GTCRN_MODES = frozenset(("gtcrn", "gtcrn_preserve"))
GTCRN_RAW_RESIDUAL_FRACTION = 0.5
GTCRN_MAX_PENDING_SAMPLES = 511  # One buffered 256-sample hop and a partial hop.


class AudioProcessingError(RuntimeError):
    """The preprocessing stream cannot continue with its current state."""


class AudioProcessingCancelled(RuntimeError):
    """A stop request reached a safe boundary outside native inference."""


@dataclass(frozen=True)
class ProcessedAudio:
    """A contiguous 16 kHz interval on the source timeline, after reset().

    ``input_end_frame`` is the native input frame count already accepted when
    this block became available. It is an availability watermark, not the end
    of the content represented by ``end_sample``. The caller maps content to
    its QPC origin with ``start_sample / 16000``; no extra left shift is needed.
    """

    samples: Any
    start_sample: int
    end_sample: int
    input_end_frame: int
    # Preserve mode supplies the same source interval in two acoustic views.
    # samples/raw_samples are the unchanged ASR input; callers must not mutate.
    raw_samples: Any = None
    denoised_samples: Any = None


def _stft(samples):
    import numpy as np

    if samples.shape != (2, MDX_WINDOW_SAMPLES):
        raise AudioProcessingError("MDX 输入窗口形状不匹配。")
    window = np.hanning(MDX_FFT_SIZE + 1)[:-1].astype(np.float32)
    padded = np.pad(samples, ((0, 0), (MDX_TRIM_SAMPLES, MDX_TRIM_SAMPLES)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, MDX_FFT_SIZE, axis=-1)[:, ::MDX_HOP_SIZE, :]
    spectrum = np.fft.rfft(frames * window, axis=-1).transpose(0, 2, 1)
    return np.stack((spectrum.real, spectrum.imag), axis=1).reshape(
        1, 4, MDX_FFT_SIZE // 2 + 1, MDX_FRAMES).astype(np.float32)


def _istft(packed):
    import numpy as np

    if packed.shape != (1, 4, MDX_FFT_SIZE // 2 + 1, MDX_FRAMES):
        raise AudioProcessingError("MDX 输出频谱形状不匹配。")
    window = np.hanning(MDX_FFT_SIZE + 1)[:-1].astype(np.float32)
    pair = packed.reshape(2, 2, MDX_FFT_SIZE // 2 + 1, MDX_FRAMES)
    frames = np.fft.irfft(pair[:, 0] + 1j * pair[:, 1], n=MDX_FFT_SIZE, axis=1).transpose(0, 2, 1)
    output = np.zeros((2, MDX_WINDOW_SAMPLES + MDX_FFT_SIZE), dtype=np.float32)
    weight = np.zeros(MDX_WINDOW_SAMPLES + MDX_FFT_SIZE, dtype=np.float32)
    for index in range(MDX_FRAMES):
        first = index * MDX_HOP_SIZE
        output[:, first:first + MDX_FFT_SIZE] += frames[:, index] * window
        weight[first:first + MDX_FFT_SIZE] += window * window
    output /= np.maximum(weight, 1e-8)
    return output[:, MDX_TRIM_SAMPLES:MDX_TRIM_SAMPLES + MDX_WINDOW_SAMPLES]


class _MdxModel:
    def __init__(self, model_file):
        path = Path(os.path.expandvars(os.fspath(model_file))).expanduser().resolve()
        if not os.fspath(model_file).strip() or not path.is_file():
            raise AudioProcessingError("MDX 本地模型文件不存在，请重新选择 Kim_Vocal_2.onnx。")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or tuple(inputs[0].shape[1:]) != (4, MDX_FREQUENCY_BINS, MDX_FRAMES):
            self.close()
            raise AudioProcessingError("MDX 模型输入与 Kim_Vocal_2 的固定窗口不匹配。")
        self._name = inputs[0].name

    def process_window(self, samples):
        import numpy as np

        spectrum = _stft(samples)[:, :, :MDX_FREQUENCY_BINS]
        estimated = np.asarray(self._session.run(None, {self._name: spectrum})[0], dtype=np.float32)
        if estimated.shape != (1, 4, MDX_FREQUENCY_BINS, MDX_FRAMES) or not np.isfinite(estimated).all():
            raise AudioProcessingError("MDX 返回了无效频谱。")
        full = np.pad(estimated, ((0, 0), (0, 0), (0, MDX_FFT_SIZE // 2 + 1 - MDX_FREQUENCY_BINS), (0, 0)))
        return _istft(full)

    def close(self):
        self._session = None


class _GtcrnModel:
    def __init__(self, model_file):
        path = Path(os.path.expandvars(os.fspath(model_file))).expanduser().resolve()
        if not os.fspath(model_file).strip() or not path.is_file():
            raise AudioProcessingError("GTCRN 本地模型文件不存在，请重新选择 gtcrn_simple.onnx。")
        import sherpa_onnx

        config = sherpa_onnx.OnlineSpeechDenoiserConfig(model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(path)),
            provider="cpu", num_threads=1, debug=False))
        if not config.validate():
            raise AudioProcessingError("GTCRN 流式配置无效。")
        self._denoiser = sherpa_onnx.OnlineSpeechDenoiser(config)
        self.frame_shift_in_samples = int(self._denoiser.frame_shift_in_samples)
        if self._denoiser.sample_rate != OUTPUT_SAMPLE_RATE or self.frame_shift_in_samples != 256:
            self.close()
            raise AudioProcessingError("GTCRN 模型采样率或帧移不匹配。")

    def process(self, samples):
        return self._denoiser.run(samples, OUTPUT_SAMPLE_RATE).samples

    def finish(self):
        return self._denoiser.flush().samples

    def reset(self):
        self._denoiser.reset()

    def close(self):
        self._denoiser = None


class _IdentityResampler:
    def resample_chunk(self, samples, last=False):
        return samples

    def clear(self):
        pass

    def delay(self):
        return 0.0


def _make_resampler(source_rate, target_rate, channels):
    if source_rate == target_rate:
        return _IdentityResampler()
    import soxr

    return soxr.ResampleStream(source_rate, target_rate, channels, dtype="float32", quality="HQ")


class SystemAudioProcessor:
    """One bounded stream; call only from the owning audio-processing worker.

    Each input call accepts at most one native second. GTCRN-only follows raw
    channel averaging and resamples directly to 16 kHz. GTCRN-only protects weak
    speech with a source-aligned 50% raw residual. gtcrn_preserve instead emits
    the unchanged raw stream for ASR and a separate, aligned pure GTCRN view
    for diagnostics. Both share the single HQ 16 kHz resampler; only the raw
    view determines production speech boundaries and ASR input.
    MDX retains one fixed
    window plus that bounded block. finish() is normal EOS; close()/reset()
    discard pending audio and never manufacture a trailing inference on stop.
    """

    def __init__(self, mode, *, sample_rate, channels, mdx_model_file="", gtcrn_model_file="",
                 stop_event=None, _mdx_model=None, _gtcrn_model=None, _resampler_factory=None):
        import numpy as np

        if mode not in PROCESSING_MODES:
            raise ValueError("不支持的电脑声音处理模式。")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
            raise ValueError("电脑声音采样率必须为 8000 至 192000 的整数。")
        if isinstance(channels, bool) or not isinstance(channels, int) or not 1 <= channels <= 32:
            raise ValueError("电脑声音通道数无效。")
        has_mdx = mode in ("mdx", "mdx_gtcrn")
        has_gtcrn = mode in ("gtcrn", "gtcrn_preserve", "mdx_gtcrn")
        if has_mdx and channels not in (1, 2):
            raise ValueError("MDX 处理只支持单声道或立体声设备。")
        self.mode, self.sample_rate, self.channels = mode, sample_rate, channels
        self.max_input_frames = sample_rate
        self._stop_event = stop_event
        self._lock = threading.RLock()
        self._closed = False
        self._finished = False
        self._failed = False
        self._mdx = self._gtcrn = self._input_resampler = self._output_resampler = None
        self._resampler_factory = _resampler_factory or _make_resampler
        self._input_frames = self._output_samples = self._mono_samples = self._mdx_input_samples = self._mdx_output_samples = 0
        self._mdx_pending = np.zeros((2, MDX_TRIM_SAMPLES), dtype=np.float32) if has_mdx else None
        self._gtcrn_raw_pending = np.empty(0, dtype=np.float32) if mode in _DIRECT_GTCRN_MODES else None
        try:
            self._check_cancelled()
            if has_mdx:
                self._mdx = _mdx_model if _mdx_model is not None else _MdxModel(mdx_model_file)
                self._check_cancelled()
            if has_gtcrn:
                self._gtcrn = _gtcrn_model if _gtcrn_model is not None else _GtcrnModel(gtcrn_model_file)
                self._check_cancelled()
            self._input_resampler = self._resampler_factory(sample_rate,
                MDX_SAMPLE_RATE if has_mdx else OUTPUT_SAMPLE_RATE, 2 if has_mdx else 1)
            if has_mdx:
                self._output_resampler = self._resampler_factory(MDX_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, 1)
            self.backend_info = {
                "mode": mode, "provider": "cpu", "input_sample_rate": sample_rate, "input_channels": channels,
                "output_sample_rate": OUTPUT_SAMPLE_RATE, "output_channels": 1,
                "max_input_frames": self.max_input_frames, "resampler_quality": "HQ",
                "source_alignment": "output sample 0 corresponds to source origin; do not subtract buffering delay",
                "mdx_window_samples": MDX_WINDOW_SAMPLES if has_mdx else 0,
                "mdx_window_sample_rate": 0 if mode in _DIRECT_GTCRN_MODES else MDX_SAMPLE_RATE,
                "mdx_step_samples": MDX_STEP_SAMPLES if has_mdx else 0,
                "mdx_first_window_input_seconds": (MDX_WINDOW_SAMPLES - MDX_TRIM_SAMPLES) / MDX_SAMPLE_RATE if has_mdx else 0.0,
                "mdx_batch_step_seconds": MDX_STEP_SAMPLES / MDX_SAMPLE_RATE if has_mdx else 0.0,
                "mdx_right_context_seconds": MDX_TRIM_SAMPLES / MDX_SAMPLE_RATE if has_mdx else 0.0,
                "gtcrn_frame_shift_samples": 256 if has_gtcrn else 0,
                "gtcrn_first_output_input_samples": 512 if has_gtcrn else 0,
                "gtcrn_steady_pending_samples": 256 if has_gtcrn else 0,
                "delay_note": ("streaming resampler, GTCRN buffering and CPU inference add wall latency"
                    if mode in _DIRECT_GTCRN_MODES else
                    "batching, resampler buffering and CPU inference add wall latency; no subsecond MDX guarantee"),
            }
            if mode == "gtcrn":
                self.backend_info.update({
                    "gtcrn_raw_residual_fraction": GTCRN_RAW_RESIDUAL_FRACTION,
                    "gtcrn_raw_pending_limit_samples": GTCRN_MAX_PENDING_SAMPLES,
                    "gtcrn_residual_alignment": "same 16 kHz source samples, including normal EOS; no callback-time pairing",
                })
            elif mode == "gtcrn_preserve":
                self.backend_info.update({
                    "asr_audio_source": "hq_resampled_raw",
                    "vad_audio_sources": ["hq_resampled_raw"],
                    "diagnostic_audio_sources": ["pure_gtcrn"],
                    "gtcrn_raw_pending_limit_samples": GTCRN_MAX_PENDING_SAMPLES,
                    "gtcrn_dual_stream_alignment": "identical source interval and length; raw ASR bytes are not mixed or amplified",
                })
        except BaseException:
            self.close()
            raise

    def _check_cancelled(self):
        if self._stop_event is not None and self._stop_event.is_set():
            if self.mode in _DIRECT_GTCRN_MODES:
                self._gtcrn_raw_pending = None
            raise AudioProcessingCancelled("电脑声音处理已取消。")

    def _check_open(self):
        self._check_cancelled()
        if self._closed:
            raise AudioProcessingError("电脑声音处理器已关闭。")
        if self._failed:
            raise AudioProcessingError("电脑声音处理器状态无效，请重新开始。")
        if self._finished:
            raise AudioProcessingError("电脑声音输入已经结束，请 reset 后再使用。")

    @property
    def input_end_frame(self):
        return self._input_frames

    @property
    def output_end_sample(self):
        return self._output_samples

    @property
    def pending_mdx_samples(self):
        return 0 if self._mdx_pending is None else self._mdx_pending.shape[1]

    @property
    def pending_gtcrn_raw_samples(self):
        return 0 if self._gtcrn_raw_pending is None else len(self._gtcrn_raw_pending)

    def _emit(self, values):
        import numpy as np

        self._check_cancelled()
        samples = np.array(values, dtype=np.float32, order="C", copy=True)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise AudioProcessingError("电脑声音处理返回了无效波形。")
        if not len(samples):
            return []
        first = self._output_samples
        self._output_samples += len(samples)
        return [ProcessedAudio(samples, first, self._output_samples, self._input_frames)]

    def _emit_protected_gtcrn(self, values):
        import numpy as np

        self._check_cancelled()
        samples = np.asarray(values, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise AudioProcessingError("GTCRN 返回了无效波形。")
        raw = self._gtcrn_raw_pending
        if raw is None or self._mono_samples - len(raw) != self._output_samples:
            raise AudioProcessingError("GTCRN 原声保护的源样本位置不连续。")
        if len(samples) > len(raw):
            raise AudioProcessingError("GTCRN 输出样本总数超过已收到的原声，无法按源时间轴混合。")
        remaining = len(raw) - len(samples)
        if remaining > GTCRN_MAX_PENDING_SAMPLES:
            raise AudioProcessingError("GTCRN 返回延迟超过流式上限，已停止处理以免积压或错配原声。")
        # A denoiser callback can contain an older source interval. Consume the
        # matching oldest raw samples, never the just-arrived native block.
        if self.mode == "gtcrn_preserve":
            result = self._emit(raw[:len(samples)])
            if result:
                block = result[0]
                result = [ProcessedAudio(block.samples, block.start_sample, block.end_sample,
                    block.input_end_frame, raw_samples=block.samples,
                    denoised_samples=np.array(samples, dtype=np.float32, order="C", copy=True))]
        else:
            mixed = (GTCRN_RAW_RESIDUAL_FRACTION * raw[:len(samples)]
                     + (1.0 - GTCRN_RAW_RESIDUAL_FRACTION) * samples)
            result = self._emit(mixed)
        self._gtcrn_raw_pending = raw[len(samples):].copy()
        self._check_cancelled()
        return result

    def _mono(self, values):
        import numpy as np

        self._check_cancelled()
        samples = np.ascontiguousarray(values, dtype=np.float32)
        self._mono_samples += len(samples)
        if self.mode in _DIRECT_GTCRN_MODES:
            if self._gtcrn_raw_pending is None or len(self._gtcrn_raw_pending) > GTCRN_MAX_PENDING_SAMPLES:
                raise AudioProcessingError("GTCRN 原声保护缓冲状态无效。")
            self._gtcrn_raw_pending = np.concatenate((self._gtcrn_raw_pending, samples))
        if self._gtcrn is not None:
            samples = self._gtcrn.process(samples)
            self._check_cancelled()
        if self.mode in _DIRECT_GTCRN_MODES:
            return self._emit_protected_gtcrn(samples)
        return self._emit(samples)

    def _last_mono(self, values):
        import numpy as np

        target = round(self._input_frames * OUTPUT_SAMPLE_RATE / self.sample_rate)
        samples = np.asarray(values, dtype=np.float32)
        difference = target - (self._mono_samples + len(samples))
        if abs(difference) > 1:
            raise AudioProcessingError("重采样输出长度与输入时长不匹配。")
        # Cascaded rational-rate rounding can differ by one final sample. Never
        # shift or rewrite audio already delivered; adjust only the EOS tail.
        if difference == 1:
            samples = np.pad(samples, (0, 1))
        elif difference == -1:
            if not len(samples):
                raise AudioProcessingError("无法裁剪重采样尾音。")
            samples = samples[:-1]
        result = self._mono(samples)
        if self._gtcrn is not None:
            tail = self._gtcrn.finish()
            self._check_cancelled()
            result.extend(self._emit_protected_gtcrn(tail) if self.mode in _DIRECT_GTCRN_MODES else self._emit(tail))
        if self._output_samples != target:
            raise AudioProcessingError("处理后的样本总数与源时间轴不一致。")
        return result

    def _run_mdx(self, *, final=False):
        import numpy as np

        result = []
        while self._mdx_pending.shape[1] >= MDX_WINDOW_SAMPLES or (
                final and self._mdx_output_samples < self._mdx_input_samples):
            self._check_cancelled()
            count = MDX_STEP_SAMPLES if not final else min(
                MDX_STEP_SAMPLES, self._mdx_input_samples - self._mdx_output_samples)
            window = self._mdx_pending[:, :MDX_WINDOW_SAMPLES]
            if window.shape[1] < MDX_WINDOW_SAMPLES:
                window = np.pad(window, ((0, 0), (0, MDX_WINDOW_SAMPLES - window.shape[1])))
            processed = np.asarray(self._mdx.process_window(window), dtype=np.float32)
            self._check_cancelled()
            if processed.shape != (2, MDX_WINDOW_SAMPLES) or not np.isfinite(processed).all():
                raise AudioProcessingError("MDX 返回了无效的音频窗口。")
            center = processed[:, MDX_TRIM_SAMPLES:MDX_TRIM_SAMPLES + count]
            self._mdx_output_samples += count
            self._mdx_pending = self._mdx_pending[:, MDX_STEP_SAMPLES:].copy()
            mono = np.ascontiguousarray(center.mean(axis=0), dtype=np.float32)
            resampled = self._output_resampler.resample_chunk(mono, last=False)
            result.extend(self._mono(resampled))
        return result

    def _accept_mdx(self, samples):
        import numpy as np

        stereo = np.asarray(samples, dtype=np.float32)
        if stereo.ndim != 2 or stereo.shape[1] != 2 or not np.isfinite(stereo).all():
            raise AudioProcessingError("MDX 重采样输入形状无效。")
        self._mdx_input_samples += len(stereo)
        self._mdx_pending = np.concatenate((self._mdx_pending, stereo.T), axis=1)
        if self._mdx_pending.shape[1] > MDX_WINDOW_SAMPLES + MDX_SAMPLE_RATE * 2:
            raise AudioProcessingError("MDX 内部音频缓冲超过限制。")
        return self._run_mdx()

    def process_block(self, samples):
        import numpy as np

        with self._lock:
            self._check_open()
            values = np.asarray(samples)
            if values.ndim != 2 or values.shape[1] != self.channels:
                raise ValueError("输入音频必须为 N×设备通道数 的数组。")
            if len(values) > self.max_input_frames:
                raise ValueError("单次输入最多为一秒，不能积压整段音频。")
            if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
                raise ValueError("输入音频必须为有限浮点采样值。")
            if not len(values):
                return []
            if np.max(np.abs(values)) > 1.000001:
                raise ValueError("输入音频必须先归一为 -1 至 1 的采样值。")
            values = np.ascontiguousarray(values, dtype=np.float32)
            self._input_frames += len(values)
            try:
                if self.mode in ("raw", "gtcrn", "gtcrn_preserve"):
                    mono = np.ascontiguousarray(values.mean(axis=1), dtype=np.float32)
                    return self._mono(self._input_resampler.resample_chunk(mono, last=False))
                stereo = np.repeat(values, 2, axis=1) if self.channels == 1 else values
                return self._accept_mdx(self._input_resampler.resample_chunk(stereo, last=False))
            except BaseException:
                if self.mode in _DIRECT_GTCRN_MODES:
                    self._gtcrn_raw_pending = None
                self._failed = True
                raise

    def finish(self):
        import numpy as np

        with self._lock:
            if self._finished and not self._closed:
                self._check_cancelled()
                return []
            self._check_open()
            try:
                if self.mode in ("raw", "gtcrn", "gtcrn_preserve"):
                    result = self._last_mono(self._input_resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))
                else:
                    remaining = self._input_resampler.resample_chunk(np.empty((0, 2), dtype=np.float32), last=True)
                    result = self._accept_mdx(remaining)
                    result.extend(self._run_mdx(final=True))
                    result.extend(self._last_mono(self._output_resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)))
                self._finished = True
                return result
            except BaseException:
                if self.mode in _DIRECT_GTCRN_MODES:
                    self._gtcrn_raw_pending = None
                self._failed = True
                raise

    def reset(self):
        import numpy as np

        with self._lock:
            self._check_cancelled()
            if self._closed:
                raise AudioProcessingError("电脑声音处理器已关闭。")
            self._input_resampler.clear()
            if self._output_resampler is not None:
                self._output_resampler.clear()
            if self._gtcrn is not None:
                self._gtcrn.reset()
            self._mdx_pending = np.zeros((2, MDX_TRIM_SAMPLES), dtype=np.float32) if self._mdx is not None else None
            if self.mode in _DIRECT_GTCRN_MODES:
                self._gtcrn_raw_pending = np.empty(0, dtype=np.float32)
            self._input_frames = self._output_samples = self._mono_samples = self._mdx_input_samples = self._mdx_output_samples = 0
            self._finished = self._failed = False

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._gtcrn is not None:
                    self._gtcrn.close()
            finally:
                try:
                    if self._mdx is not None:
                        self._mdx.close()
                finally:
                    self._gtcrn = self._mdx = self._input_resampler = self._output_resampler = self._mdx_pending = None
                    self._gtcrn_raw_pending = None


def create_system_audio_processor(settings, *, sample_rate, channels, stop_event=None):
    return SystemAudioProcessor(getattr(settings, "system_audio_processing", "raw"),
        sample_rate=sample_rate, channels=channels,
        mdx_model_file=getattr(settings, "mdx_model_file", ""),
        gtcrn_model_file=getattr(settings, "gtcrn_model_file", ""), stop_event=stop_event)
