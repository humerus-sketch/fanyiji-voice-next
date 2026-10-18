"""Voice application entry points; capture starts only through an explicit action."""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import tomllib
from pathlib import Path

from . import __version__


def write_report(report, path=None):
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if sys.stdout is not None:
        print(content)


def _legacy_modules():
    return sorted(name for name in sys.modules if name == "fanyiji" or name.startswith("fanyiji."))


def _discover_project_root():
    """Recognize our source or dist layout without searching CWD/ancestors."""
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        base = executable.parent
        if (executable.name.casefold() != "fanyijivoicenext.exe" or base.name.casefold() != "fanyijivoicenext"
                or base.parent.parent.name.casefold() != "dist"
                or not re.fullmatch(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9][A-Za-z0-9.-]*)?", base.parent.name)):
            return None
        candidate = base.parents[2]
    else:
        source = Path(__file__).resolve()
        if source.parent.name != "fanyiji_next" or source.parent.parent.name != "src":
            return None
        candidate = source.parents[2]
    try:
        metadata = tomllib.loads((candidate / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return None
    project = metadata.get("project")
    if not isinstance(project, dict) or project.get("name") != "fanyiji-voice-next":
        return None
    return candidate


def _startup_profile_error(target, detail, report_path):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWidgets import QMessageBox

    message = (f"无法读取启动配置：{target}\n{detail}\n\n"
               "请修复或从备份恢复此文件；也可用 --settings-file 指定已有的有效配置。"
               "本次未启动语音处理，未覆盖原配置。")
    write_report({"application": "fanyiji-voice-next", "version": __version__,
                  "startup_error": message, "settings_file": str(target),
                  "audio_capture": False, "models_loaded": False, "settings_written": False}, report_path)
    title = "翻译姬 · 启动配置错误"
    timer = QTimer()
    timer.setSingleShot(True)
    def dismiss_error():
        # This timer belongs only to this ordinary error; never accept a
        # confirmation, license or another window's dialog.
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMessageBox) and widget.windowTitle() == title and widget.text() == message:
                widget.accept()
    timer.timeout.connect(dismiss_error)
    timer.start(3000)
    try:
        QMessageBox.critical(None, title, message)
    finally:
        timer.stop()
    return 2


def check_runtime(report_path=None, *, ab_engines=None):
    modules = ("PySide6.QtWidgets", "numpy", "sherpa_onnx", "ctranslate2", "sentencepiece", "pyaudiowpatch", "soxr",
               "faster_whisper", "tokenizers", "av", "onnxruntime", "huggingface_hub", "tqdm")
    if ab_engines is None and getattr(sys, "frozen", False):
        try:
            build_info = json.loads((Path(sys.executable).parent / "build-info.json").read_text(encoding="utf-8"))
            ab_engines = build_info.get("ab_engines_bundled", False)
        except (OSError, ValueError):
            ab_engines = False
    if ab_engines:
        modules += ("torch", "torchaudio", "transformers", "funasr", "librosa", "soundfile", "safetensors")
    results = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            results[name] = {"available": True, "version": getattr(module, "__version__", "bundled")}
        except Exception as exc:
            results[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if ab_engines:
        # Transformers exposes lazy exports: importing only its root package
        # does not prove the actual audio/model entry points are bundled.
        for package, attribute in (("transformers", "AutoProcessor"),
                                   ("transformers", "AutoModelForMultimodalLM"),
                                   ("funasr", "AutoModel")):
            name = package + "." + attribute
            try:
                entry = getattr(importlib.import_module(package), attribute)
                if not callable(entry):
                    raise ImportError("模型入口不是可调用的类")
                results[name] = {"available": True, "class_module": entry.__module__}
            except Exception as exc:
                results[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    passed = all(result["available"] for result in results.values()) and not _legacy_modules()
    write_report({"application": "fanyiji-voice-next", "version": __version__,
                  "runtime_imports_passed": passed, "audio_capture": False, "models_loaded": False,
                  "ab_engines_checked": bool(ab_engines),
                  "settings_written": False, "legacy_modules_loaded": _legacy_modules(), "modules": results}, report_path)
    return 0 if passed else 2


def _configure_application(app):
    from PySide6.QtGui import QFont, QFontDatabase

    app.setApplicationName("FanyiJiVoiceNext")
    app.setApplicationDisplayName("翻译姬 · 语音版 NEXT")
    app.setOrganizationName("FanyiJiVoiceNext")
    for name in ("msyh.ttc", "msyhbd.ttc"):
        candidate = Path("C:/Windows/Fonts") / name
        if candidate.is_file():
            QFontDatabase.addApplicationFont(str(candidate))
    app.setFont(QFont("Microsoft YaHei UI", 10))


def _settle(app):
    for _ in range(8):
        app.processEvents()


def smoke_test(screenshot=None, report_path=None):
    """Exercise the live UI/controller contract with a fake capture runtime."""
    from time import perf_counter

    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from .controller import AppController
    from .models import AppSettings, RuntimeEvent
    from .studio import StudioWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    _configure_application(app)
    runtimes = []

    class SmokeRuntime:
        is_running = False
        is_busy = False

        def __init__(self, callback):
            self.callback = callback
            runtimes.append(self)

        def start(self, settings, session_id):
            self.session_id = session_id
            self.is_running = self.is_busy = True
            self.callback(RuntimeEvent("state", session_id, status="running"))

        def stop(self):
            self.is_running = self.is_busy = False
            self.callback(RuntimeEvent("state", self.session_id, status="stopped"))

    window = StudioWindow(AppSettings(font_size=30), prototype=False)
    controller = AppController(window, runtime_factory=SmokeRuntime, persist_settings=False,
                               device_lister=lambda: [])
    window._controller = controller
    window.show()
    _settle(app)
    checks = {"initially_idle": controller.state == "stopped" and not runtimes}
    window.start_button.click()
    _settle(app)
    checks["manual_start_connected"] = controller.state == "running" and len(runtimes) == 1
    examples = (
        ("mic", "1", "你好，今天也请多关照。", "こんにちは。今日もよろしくお願いします。"),
        ("system", "2", "今日はいい天気ですね。", "今天天气真好。"),
        ("mic", "3", "我们先确认一下明天的安排。", "まず、明日の予定を確認しましょう。"),
        ("system", "4", "資料を先に送ってください。", "请先把资料发过来。"),
    )
    for channel, segment_id, original, translated in examples:
        runtimes[0].callback(RuntimeEvent("original", controller.session_id, channel, segment_id, text=original))
        runtimes[0].callback(RuntimeEvent("translation", controller.session_id, channel, segment_id, text=translated))
    _settle(app)
    checks["paired_captions"] = len(window.store.items()) == 4 and all(item.original and item.translated for item in window.store.items())
    caption_ids = [item.id for item in window.store.items()]
    window.transparency_button.click()
    window.transparency_slider.setValue(45)
    _settle(app)
    checks["transparency_adjusts_live"] = (
        window.settings.window_opacity == 55 and abs(window.windowOpacity() - .55) < .01)
    window.transparency_button.click()
    checks["transparency_turns_off"] = abs(window.windowOpacity() - 1) < .01
    window.transparency_button.click()
    checks["transparency_restores_level"] = (
        window.settings.window_opacity == 55 and abs(window.windowOpacity() - .55) < .01)
    checks["transparency_preserves_session"] = (
        len(runtimes) == 1 and runtimes[0].is_running and controller.state == "running"
        and [item.id for item in window.store.items()] == caption_ids)
    window.transparency_slider.setValue(0)
    failed_channel, failed_id, failed_source, restored_translation = examples[2]
    failure_message = "界面检查：本句翻译未完成。"
    notice_started = perf_counter()
    runtimes[0].callback(RuntimeEvent("error", controller.session_id, failed_channel, failed_id,
                                    message=failure_message, status="recoverable"))
    _settle(app)
    failed_caption = window.store.items(failed_channel)[-1]
    checks["failed_caption_shows_original"] = (
        failed_caption.original == failed_source and not failed_caption.translated
        and window.panels[failed_channel].latest_translation.text() == failed_source
        and failure_message not in window.transcript())
    checks["recoverable_notice_three_second_timer"] = window._toast_timer.interval() == 3000
    notice_loop = QEventLoop()
    window._toast_timer.timeout.connect(notice_loop.quit)
    QTimer.singleShot(3600, notice_loop.quit)
    notice_loop.exec()
    window._toast_timer.timeout.disconnect(notice_loop.quit)
    notice_elapsed_ms = round((perf_counter() - notice_started) * 1000)
    checks["recoverable_notice_expires"] = window.notice.isHidden() and not window._toast_timer.isActive()
    runtimes[0].callback(RuntimeEvent("translation", controller.session_id, failed_channel, failed_id,
                                    text=restored_translation))
    _settle(app)
    window.model_button.click()
    _settle(app)
    dialog = window._runtime_dialog
    editors = [*dialog.engine_combos.values(), *dialog.device_combos.values(),
               *dialog.path_edits.values(), *dialog.browse_buttons.values(), dialog.apply_button,
               dialog.refresh_button, *(control for pair in dialog.channel_controls.values() for control in pair)]
    checks["runtime_settings_locked"] = (window.model_button.isEnabled() and dialog.isVisible()
                                         and all(not editor.isEnabled() for editor in editors)
                                         and not window.clear_button.isEnabled())
    dialog.stop_to_edit_button.click()
    _settle(app)
    checks["stop_preserves_history"] = controller.state == "stopped" and len(window.store.items()) == 4
    checks["settings_unlock_after_stop"] = (dialog.isVisible() and all(editor.isEnabled() for editor in editors)
                                             and window.clear_button.isEnabled())
    dialog.reject()
    _settle(app)
    window.start_button.click()
    _settle(app)
    checks["restart_preserves_history"] = controller.state == "running" and len(window.store.items()) == 4
    window.stop_button.click()
    _settle(app)
    window.show_error("离线界面检查使用预置文本与模拟采集器，未开启声音设备。")
    if screenshot:
        if app.platformName() != "offscreen":
            loop = QEventLoop()
            QTimer.singleShot(280, loop.quit)
            loop.exec()
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        checks["screenshot_saved"] = window.grab().save(str(screenshot))
    result = {"application": "fanyiji-voice-next", "version": __version__,
              "ui_smoke": "passed" if all(checks.values()) else "failed",
              "ok": all(checks.values()) and not _legacy_modules(), "checks": checks,
              "audio_capture": False, "models_loaded": False, "settings_written": False,
              "legacy_modules_loaded": _legacy_modules(), "caption_count": len(window.store.items())}
    result["recoverable_notice_elapsed_ms"] = notice_elapsed_ms
    window.close()
    window.deleteLater()
    _settle(app)
    write_report(result, report_path)
    return 0 if result["ok"] else 2


def main(argv=None):
    from .verification import BACKEND_OPTIONS, BACKEND_PATH_OPTIONS, add_backend_arguments

    parser = argparse.ArgumentParser(description="翻译姬 · 语音版 NEXT，本地中日语音字幕")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke-test", action="store_true", help="模拟运行界面和会话控制，不采音、不加载模型、不保存设置")
    modes.add_argument("--check-runtime", action="store_true", help="检查本地运行依赖，不采音或加载模型")
    modes.add_argument("--verify-models", action="store_true", help="仅用固定官方 WAV 验证本地模型，不采音或保存设置")
    modes.add_argument("--verify-ab-audio", type=Path, help="只读取录音前8秒检查 A/B 推理；需显式配置，不采音或保存设置")
    modes.add_argument("--demo", action="store_true", help="打开保留的界面演示，不采集声音")
    parser.add_argument("--model-root", type=Path, help="为尚未设置的模型提供本地目录，不下载或自动开始")
    parser.add_argument("--settings-file", type=Path, help="使用指定的新项目配置文件")
    parser.add_argument("--require-settings", action="store_true", help="配置必须存在且有效；失败时停止启动，不使用默认值")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--report", type=Path)
    # Omitted options preserve a GUI user's existing settings. The verifier has
    # its own isolated SenseVoice/M2M100 CPU profile and never reads that file.
    add_backend_arguments(parser, defaults=False)
    args = parser.parse_args(argv)
    if args.check_runtime:
        return check_runtime(args.report)
    if args.smoke_test:
        return smoke_test(args.screenshot, args.report)
    if args.verify_ab_audio is not None:
        if args.settings_file is None:
            parser.error("--verify-ab-audio 需要 --settings-file 指定本地 A/B 配置。")
        try:
            from .ab_verification import verify_audio
            result = verify_audio(args.verify_ab_audio, args.settings_file)
            result.update(application="fanyiji-voice-next", version=__version__, passed=result["nonempty_transcription"])
        except Exception as exc:
            result = {"application": "fanyiji-voice-next", "version": __version__, "passed": False,
                      "error": str(exc), "audio_capture": False, "settings_written": False}
        write_report(result, args.report)
        return 0 if result["passed"] else 1
    if args.verify_models:
        if args.model_root is None:
            parser.error("--verify-models 需要 --model-root 指定本地模型目录。")
        from .verification import main as verification_main
        command = ["--model-root", str(args.model_root)]
        if args.report is not None:
            command.extend(("--report", str(args.report)))
        for name, _choices, _default in BACKEND_OPTIONS:
            value = getattr(args, name)
            if value is not None:
                command.extend(("--" + name.replace("_", "-"), value))
        for name in BACKEND_PATH_OPTIONS:
            value = getattr(args, name)
            if value is not None:
                command.extend(("--" + name.replace("_", "-"), value))
        return verification_main(command)
    if args.demo:
        from .prototype import main as prototype_main
        return prototype_main([])

    from PySide6.QtWidgets import QApplication
    from dataclasses import replace
    from .controller import AppController
    from .models import AppSettings
    from .settings import (SettingsError, load_settings, settings_path, validate_settings,
                           selected_model_combination, with_model_combination)
    from .release_profile import new_application_settings
    from .model_paths import external_model_root
    from .studio import StudioWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    _configure_application(app)
    project_root = _discover_project_root()
    project_profile = project_root / "settings" / "voice.settings.json" if project_root is not None else None
    target = (args.settings_file or project_profile or settings_path()).resolve()
    preview_profile = args.settings_file is None and project_profile is not None and __version__.endswith("-dev")
    if preview_profile:
        target = (project_root / "settings" / "voice.preview.settings.json").resolve()
        # Read the stable profile only to seed a first preview in memory. Future
        # edits belong to the preview target so older releases stay compatible.
        source = target if target.exists() else project_profile.resolve()
    else:
        source = target
    strict_profile = args.require_settings or preview_profile or (project_profile is not None and target == project_profile.resolve())
    error = None
    try:
        if strict_profile and not source.is_file():
            return _startup_profile_error(source, "配置文件不存在或不是普通文件。", args.report)
        if source.exists():
            settings = load_settings(source)
        elif getattr(sys, "frozen", False) and args.settings_file is None and project_root is None:
            from .release_profile import load_release_defaults
            defaults = Path(sys.executable).resolve().parent / "release-defaults.json"
            settings = load_release_defaults(defaults) if defaults.is_file() else new_application_settings()
        else:
            settings = new_application_settings()
    except SettingsError as exc:
        if strict_profile:
            return _startup_profile_error(source, str(exc), args.report)
        settings = new_application_settings()
        error = f"设置读取失败，当前使用默认值：{exc}"
    overrides = {name: getattr(args, name) for name, _choices, _default in BACKEND_OPTIONS
                 if getattr(args, name) is not None}
    overrides.update({name: getattr(args, name) for name in BACKEND_PATH_OPTIONS
                      if getattr(args, name) is not None})
    if overrides:
        settings = replace(settings, **overrides)
        try:
            validate_settings(settings)
        except SettingsError as exc:
            parser.error(str(exc))
    base = project_root or (Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2])
    # Resolve anew on every launch: renaming the EXE changes the prefix. Never
    # reuse an old absolute model path, CWD or PyInstaller's extraction folder.
    settings = with_model_combination(settings, selected_model_combination(settings), external_model_root())
    window = StudioWindow(settings, prototype=False)
    # Preserve explicit profile/model selections when the updater restarts this EXE.
    window._update_restart_args = tuple(sys.argv[1:] if argv is None else argv)
    window.set_settings_source(str(target.resolve()))
    controller = AppController(window, settings_file=target)
    window._controller = controller
    from .update_dialog import open_update_dialog
    window.update_requested.connect(lambda: open_update_dialog(window))
    from .model_download_dialog import ModelDownloadDialog

    def show_models():
        dialog = getattr(window, "_model_download_dialog", None)
        if dialog is None:
            dialog = ModelDownloadDialog(external_model_root(), parent=window,
                                         capture_active=lambda: controller.state not in {"stopped", "error"})
            def apply_pair(pair):
                if controller.state not in {"stopped", "error"} or controller._runtime_busy():
                    window.show_error("请先停止翻译，再切换 A/B 方案。")
                    return
                selected = with_model_combination(window.settings, pair, external_model_root())
                window.update_settings(selected)
                window.settings_changed.emit(selected)
            dialog.bundle_selected.connect(apply_pair)
            window._model_download_dialog = dialog
        dialog.refresh_states()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    window.models_requested.connect(show_models)
    app.aboutToQuit.connect(controller.shutdown)
    if error:
        window.show_error(error)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
