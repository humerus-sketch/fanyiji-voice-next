"""UI-independent settings, runtime events, and caption history."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass(frozen=True)
class ChannelSettings:
    enabled: bool = True
    direction: str = "zh_to_ja"
    device_index: int | None = None


@dataclass(frozen=True)
class AppSettings:
    microphone: ChannelSettings = field(default_factory=ChannelSettings)
    system: ChannelSettings = field(
        default_factory=lambda: ChannelSettings(direction="ja_to_zh")
    )
    asr_model: str = ""
    asr_tokens: str = ""
    translation_model_dir: str = ""
    translation_spm: str = ""
    font_size: int = 22
    show_original: bool = True
    always_on_top: bool = False
    window_opacity: int = 100
    recognition_backend: str = "sensevoice"
    whisper_model_dir: str = ""
    recognition_device: str = "cpu"
    recognition_compute_type: str = "int8"
    translation_backend: str = "hymt"
    nllb_model_dir: str = ""
    nllb_tokenizer_dir: str = ""
    hymt_model_file: str = ""
    hymt_server_exe: str = ""
    translation_device: str = "cpu"
    translation_compute_type: str = "int8"
    vad_model: str = ""
    system_audio_processing: str = "raw"
    mdx_model_file: str = ""
    gtcrn_model_file: str = ""


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    session_id: int
    channel: str = ""
    segment_id: str = ""
    text: str = ""
    level: float = 0.0
    message: str = ""
    status: str = ""
    elapsed_ms: int = 0
    queue_depth: int = 0
    speech_start_at: float = 0.0
    speech_end_at: float = 0.0
    segment_submitted_at: float = 0.0
    emitted_at: float = 0.0
    endpoint_ms: int = 0
    queue_ms: int = 0
    asr_ms: int = 0
    translation_ms: int = 0
    end_reason: str = ""
    speech_detector: str = ""


@dataclass
class Caption:
    id: str
    channel: str
    original: str = ""
    translated: str = ""
    status: str = "translating"
    created_at: float = field(default_factory=time)


class CaptionStore:
    """Match results to rows; voice sessions can retain all rows until cleared."""

    def __init__(self, *, max_per_channel: int | None = 100) -> None:
        if max_per_channel is not None and max_per_channel < 1:
            raise ValueError("max_per_channel must be positive or None")
        self._max_per_channel = max_per_channel
        self._items: dict[tuple[str, str], Caption] = {}

    def add_original(self, channel: str, segment_id: str, text: str) -> Caption:
        key = (channel, segment_id)
        caption = self._items.get(key)
        if caption is not None:
            caption.original = text
            return caption

        caption = Caption(id=segment_id, channel=channel, original=text)
        self._items[key] = caption
        if self._max_per_channel is not None:
            channel_keys = [item_key for item_key in self._items if item_key[0] == channel]
            if len(channel_keys) > self._max_per_channel:
                del self._items[channel_keys[0]]
        return caption

    def set_translation(
        self,
        channel: str,
        segment_id: str,
        text: str,
        failed: bool = False,
    ) -> Caption | None:
        caption = self._items.get((channel, segment_id))
        if caption is None:
            # A result can arrive after clear() or eviction; never create an orphan.
            return None
        caption.translated = text
        caption.status = "failed" if failed else "complete"
        return caption

    def clear(self) -> None:
        self._items.clear()

    def cancel_pending(self) -> list[Caption]:
        cancelled = []
        for caption in self._items.values():
            if caption.status == "translating":
                caption.status = "cancelled"
                cancelled.append(caption)
        return cancelled

    def items(self, channel: str | None = None) -> list[Caption]:
        return [
            caption
            for caption in self._items.values()
            if channel is None or caption.channel == channel
        ]
