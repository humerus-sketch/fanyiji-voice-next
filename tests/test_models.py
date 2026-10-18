"""Domain checks without Qt, model files, network access, or audio hardware."""

from dataclasses import FrozenInstanceError

import pytest

from fanyiji_next.models import AppSettings, CaptionStore, ChannelSettings, RuntimeEvent


def test_default_channel_directions_are_independent():
    settings = AppSettings()
    assert settings.microphone == ChannelSettings(direction="zh_to_ja")
    assert settings.system == ChannelSettings(direction="ja_to_zh")
    assert settings.microphone is not settings.system


def test_runtime_event_is_frozen():
    event = RuntimeEvent(kind="original", session_id=1)
    with pytest.raises(FrozenInstanceError):
        event.text = "changed"


def test_translation_updates_its_channel_and_segment():
    store = CaptionStore()
    first = store.add_original("microphone", "same", "你好")
    second = store.add_original("system", "same", "こんにちは")
    assert store.set_translation("microphone", "same", "こんにちは") is first
    assert first.translated == "こんにちは"
    assert first.status == "complete"
    assert second.translated == ""
    assert store.items("microphone") == [first]
    assert store.items() == [first, second]


def test_repeated_original_does_not_duplicate_row():
    store = CaptionStore()
    first = store.add_original("microphone", "1", "你")
    assert store.add_original("microphone", "1", "你好") is first
    assert first.original == "你好"
    assert len(store.items()) == 1


def test_failure_updates_existing_row():
    store = CaptionStore()
    row = store.add_original("microphone", "1", "你好")
    assert store.set_translation("microphone", "1", "翻译失败", failed=True) is row
    assert row.status == "failed"
    assert row.original == "你好"
    assert row.translated == ""
    assert row.error_message == "翻译失败"


def test_success_after_failure_replaces_diagnostic_with_translation():
    store = CaptionStore()
    row = store.add_original("system", "1", "やってみよう。")
    store.set_translation("system", "1", "未通过一致性检查", failed=True)
    assert store.set_translation("system", "1", "试试看吧。") is row
    assert row.status == "complete"
    assert row.translated == "试试看吧。"
    assert row.error_message == ""


def test_late_failure_does_not_create_caption_without_original():
    store = CaptionStore()
    assert store.set_translation("system", "missing", "识别失败", failed=True) is None
    store.add_original("system", "1", "保留这句原文")
    store.clear()
    assert store.set_translation("system", "1", "迟到的翻译失败", failed=True) is None
    assert store.items() == []


def test_clear_rejects_late_translation():
    store = CaptionStore()
    store.add_original("microphone", "1", "你好")
    store.clear()
    assert store.set_translation("microphone", "1", "こんにちは") is None
    assert store.items() == []


def test_history_is_bounded_per_channel_and_rejects_evicted_results():
    store = CaptionStore()
    system_row = store.add_original("system", "0", "残す")
    for index in range(101):
        store.add_original("microphone", str(index), str(index))
    assert len(store.items("microphone")) == 100
    assert store.items("microphone")[0].id == "1"
    assert store.items("system") == [system_row]
    assert store.set_translation("microphone", "0", "late") is None
    assert len(store.items()) == 101


def test_unknown_translation_does_not_add_original():
    store = CaptionStore()
    assert store.set_translation("system", "missing", "translation") is None
    assert store.items() == []


def test_full_session_history_keeps_early_rows_and_their_late_translations():
    store = CaptionStore(max_per_channel=None)
    first = store.add_original("system", "0", "最初の文です。")
    for index in range(1, 251):
        store.add_original("system", str(index), f"公開例{index}です。")
    assert len(store.items("system")) == 251
    assert store.items("system")[0] is first
    assert store.set_translation("system", "0", "这是第一句。") is first
    assert first.translated == "这是第一句。"
    store.clear()
    assert not store.items()
    assert store.set_translation("system", "0", "过期结果") is None


def test_cancelling_pending_captions_preserves_originals_and_completed_results():
    store = CaptionStore()
    pending = store.add_original("mic", "pending", "保留这句原文")
    complete = store.add_original("system", "complete", "こんにちは")
    store.set_translation("system", "complete", "你好")
    failed = store.add_original("mic", "failed", "另一句")
    store.set_translation("mic", "failed", "失败原因", failed=True)
    assert store.cancel_pending() == [pending]
    assert pending.status == "cancelled" and pending.translated == ""
    assert pending.original == "保留这句原文"
    assert complete.status == "complete" and complete.translated == "你好"
    assert failed.status == "failed"
    assert store.cancel_pending() == []
