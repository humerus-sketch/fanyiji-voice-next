"""Create a fresh portable folder; never replace an existing application."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
VOICE_COLLECT_ALL = ("numpy", "sherpa_onnx", "ctranslate2", "sentencepiece", "pyaudiowpatch", "soxr")
VOICE_SUBMODULES = ("faster_whisper", "tokenizers", "av", "onnxruntime", "huggingface_hub", "tqdm")
VOICE_BINARIES = ("tokenizers", "av", "onnxruntime")
VOICE_MODULES = (*VOICE_COLLECT_ALL, *VOICE_SUBMODULES)
NVIDIA_RUNTIMES = (
    ("nvidia-cublas-cu12", "nvidia/cublas/bin", {"cublas64_12.dll", "cublaslt64_12.dll"}),
    ("nvidia-cudnn-cu12", "nvidia/cudnn/bin", {"cudnn64_9.dll"}),
)
VOICE_METADATA = ("faster-whisper", "tokenizers", "av", "onnxruntime", "huggingface-hub", "tqdm",
                  *(package for package, _destination, _required in NVIDIA_RUNTIMES))
CORE_LICENSES = ("PySide6-Essentials", "shiboken6", "pyinstaller")
VOICE_LICENSES = ("numpy", "sherpa-onnx", "sherpa-onnx-core", "ctranslate2", "sentencepiece", "PyAudioWPatch", "soxr",
                  *VOICE_METADATA)
PROTOTYPE_EXCLUDES = ("fanyiji_next.app", "fanyiji_next.controller", "fanyiji_next.audio_runtime",
                      "fanyiji_next.recognition_engines", "fanyiji_next.translation_engines",
                      "fanyiji_next.verification", "fanyiji_next.speech_gate", "fanyiji_next.native_runtime",
                      "nvidia", "nvidia.cublas", "nvidia.cudnn", *VOICE_MODULES)


def build_environment():
    environment = os.environ.copy()
    windows = Path(environment.get("SystemRoot", "C:/Windows"))
    # Unrelated media tools can export incompatible DLLs under the same name.
    environment["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), sys.base_prefix,
                                          str(windows / "System32"), str(windows)))
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def project_version():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def license_packages(edition):
    return CORE_LICENSES + (VOICE_LICENSES if edition == "voice" else ())


def nvidia_binaries():
    """Read wheel manifests, preserving split CUDA/cuDNN DLL directories."""
    result = []
    for package, destination, required in NVIDIA_RUNTIMES:
        distribution = importlib.metadata.distribution(package)
        root = Path(distribution.locate_file("")).resolve()
        prefix = tuple(destination.split("/"))
        found = set()
        for item in distribution.files or []:
            relative = Path(str(item))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            if tuple(part.lower() for part in relative.parts[:len(prefix)]) != prefix:
                continue
            if relative.suffix.lower() != ".dll":
                continue
            source = Path(distribution.locate_file(item)).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise RuntimeError(f"NVIDIA wheel DLL 不存在或离开安装目录：{package}/{relative.as_posix()}")
            found.add(source.name.lower())
            result.append((source, relative.parent.as_posix()))
        if missing := required - found:
            raise RuntimeError(f"NVIDIA wheel 缺少 Windows 运行库：{package} ({', '.join(sorted(missing))})")
    return sorted(result, key=lambda item: (item[1], item[0].name.lower()))


def build_command(output, work, edition):
    if edition not in {"prototype", "voice"}:
        raise ValueError("Unknown edition: " + edition)
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed",
               "--name", "FanyiJiVoiceNext", "--paths", str(ROOT / "src"),
               "--distpath", str(output), "--workpath", str(work / "work"),
               "--specpath", str(work)]
    if edition == "prototype":
        for package in PROTOTYPE_EXCLUDES:
            command.extend(("--exclude-module", package))
    else:
        for package in VOICE_COLLECT_ALL:
            command.extend(("--collect-all", package))
        # faster-whisper includes Silero weights, and onnxruntime includes ONNX
        # example datasets. Collect code/binaries without either package's data.
        for package in VOICE_SUBMODULES:
            command.extend(("--collect-submodules", package))
        for package in VOICE_BINARIES:
            command.extend(("--collect-binaries", package))
        for package in VOICE_METADATA:
            command.extend(("--copy-metadata", package))
        # Namespace packages do not reliably expose native DLLs to collection.
        # Include every split cuDNN library rather than just the loader DLL.
        for source, destination in nvidia_binaries():
            command.extend(("--add-binary", f"{source}:{destination}"))
    command.append(str(ROOT / ("run_prototype.py" if edition == "prototype" else "run.py")))
    return command


def build_update_helper(folder, work, environment):
    """A standalone helper can replace the main EXE after its process exits."""
    destination = folder / "_updater"
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
               "--name", "FanyiJiUpdate", "--paths", str(ROOT / "src"),
               "--distpath", str(destination), "--workpath", str(work / "helper-work"),
               "--specpath", str(work), str(ROOT / "tools" / "update_helper.py")]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    if not (destination / "FanyiJiUpdate.exe").is_file():
        raise RuntimeError("独立更新程序未成功构建。")


def collect_licenses(folder, edition):
    from packaging.requirements import Requirement

    licenses = folder / "licenses"
    pending = list(license_packages(edition))
    seen = set()
    while pending:
        package = pending.pop(0)
        key = package.lower().replace("_", "-").replace(".", "-")
        if key in seen:
            continue
        seen.add(key)
        distribution = importlib.metadata.distribution(package)
        # Include the runtime dependencies' notices too; optional conversion,
        # training and development extras are not part of the portable app.
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
        for item in distribution.files or []:
            if any(word in Path(str(item)).name.lower() for word in ("license", "copying", "notice")):
                source = Path(distribution.locate_file(item))
                if source.is_file():
                    relative = Path(str(item))
                    parts = relative.parts[1:] if relative.parts and relative.parts[0].endswith(".dist-info") else relative.parts
                    if relative.is_absolute() or ".." in parts:
                        continue
                    target = licenses / package / Path(*parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        (licenses / "Python").mkdir(parents=True, exist_ok=True)
        shutil.copy2(python_license, licenses / "Python" / "LICENSE.txt")


def _is_certifi_public_bundle(file, folder):
    relative = file.relative_to(folder)
    if relative.as_posix() != "_internal/certifi/cacert.pem":
        return False
    if file.resolve() != folder.resolve() / relative:
        return False
    try:
        import certifi

        content = file.read_bytes()
        if b"PRIVATE KEY" in content.upper():
            return False
        expected = Path(certifi.where()).read_bytes()
    except (ImportError, OSError):
        return False
    return hashlib.sha256(content).digest() == hashlib.sha256(expected).digest()


def assert_public_payload(folder):
    forbidden = {".onnx", ".gguf", ".safetensors", ".pth", ".pt", ".ckpt", ".tflite", ".key", ".pem"}
    private = {"settings.json", "auth.json", "credentials.json", "cookies", "cookies.sqlite", "login data", "web data"}
    model_files = {"model.bin", "pytorch_model.bin", "sentencepiece.bpe.model", "tokenizer.model", "spiece.model"}
    for file in folder.rglob("*"):
        if file.is_file() and (file.suffix.lower() in forbidden or file.name.lower() in private | model_files):
            if _is_certifi_public_bundle(file, folder):
                continue
            raise RuntimeError("输出目录出现模型或用户资料，请检查打包依赖：" + file.name)


def validate_frozen_report(report, edition, mode):
    if report.get("audio_capture") is not False or report.get("settings_written") is not False:
        raise RuntimeError("冻结验证未明确保证不采音、不保存设置。")
    model_flags = [report[name] for name in ("model_loading", "models_loaded") if name in report]
    if not model_flags or any(value is not False for value in model_flags):
        raise RuntimeError("冻结验证未明确保证不加载模型。")
    if report.get("legacy_modules_loaded"):
        raise RuntimeError("冻结程序导入了旧项目模块。")
    if edition == "prototype":
        if (report.get("ok") is not True or report.get("voice_backend_loaded") is not False
                or report.get("forbidden_modules") != [] or report.get("prototype_only") is not True):
            raise RuntimeError("原型冻结验证失败或意外载入语音模块。")
    elif mode == "--check-runtime":
        if report.get("runtime_imports_passed") is not True:
            raise RuntimeError("语音版冻结运行库检查失败。")
    elif report.get("ui_smoke") != "passed" and report.get("ok") is not True:
        raise RuntimeError("语音版冻结界面检查失败。")


def run_frozen_checks(folder, checks, edition, environment):
    checks.mkdir()
    environment = environment.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    windows = Path(environment.get("SystemRoot", "C:/Windows"))
    environment["PATH"] = os.pathsep.join((str(windows / "System32"), str(windows)))
    # Any accidental settings access is isolated from the actual user profile.
    profile = checks / "isolated-profile"
    environment["LOCALAPPDATA"] = str(profile)
    environment["APPDATA"] = str(profile / "roaming")
    environment["HF_HOME"] = str(profile / "huggingface")
    environment["XDG_CACHE_HOME"] = str(profile / "cache")
    environment["HF_HUB_OFFLINE"] = "1"
    executable = folder / "FanyiJiVoiceNext.exe"
    for mode, filename in (("--check-runtime", "runtime.json"), ("--smoke-test", "ui.json")):
        report_path = checks / filename
        command = [str(executable), mode, "--report", str(report_path)]
        if mode == "--smoke-test":
            command.extend(("--screenshot", str(checks / "ui.png")))
        subprocess.run(command, cwd=folder, env=environment, check=True, timeout=60)
        validate_frozen_report(json.loads(report_path.read_text(encoding="utf-8")), edition, mode)
    if (checks / "isolated-profile" / "FanyiJiVoiceNext" / "settings.json").exists():
        raise RuntimeError("冻结检查意外保存了设置。")


def main(argv=None):
    parser = argparse.ArgumentParser(description="分别构建独立原型或语音版，始终保留已有输出。")
    parser.add_argument("--edition", choices=("prototype", "voice"), default="prototype")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-updater", action="store_true", help="同时构建独立的退出后更新程序（仅语音版）")
    args = parser.parse_args(argv)
    if args.with_updater and args.edition != "voice":
        parser.error("只有语音版支持独立更新程序。")
    version = project_version()
    output = (args.output or ROOT / "dist" / f"{version}-{args.edition}").resolve()
    if output.exists() or output.is_symlink():
        parser.error("输出目录已存在。请用 --output 指定一个新的空路径，以保留已有程序。")
    entrypoint = ROOT / ("run_prototype.py" if args.edition == "prototype" else "run.py")
    if not entrypoint.is_file():
        parser.error("构建入口不存在：" + str(entrypoint))
    try:
        for package in license_packages(args.edition):
            importlib.metadata.distribution(package)
        work = ROOT / "build" / f"{args.edition}-{output.name}-{uuid4().hex[:8]}"
        command = build_command(output, work, args.edition)
    except importlib.metadata.PackageNotFoundError as exc:
        parser.error(f"构建依赖缺失：{exc.name}；请先在新项目独立环境中安装对应依赖。")
    except RuntimeError as exc:
        parser.error(str(exc))
    output.mkdir(parents=True)
    environment = build_environment()
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    folder = output / "FanyiJiVoiceNext"
    if args.with_updater:
        build_update_helper(folder, work, environment)
    shutil.copy2(ROOT / "README.md", folder / "使用说明.md")
    collect_licenses(folder, args.edition)
    assert_public_payload(folder)
    (folder / "build-info.json").write_text(json.dumps({
        "application": "fanyiji-voice-next", "version": version, "edition": args.edition, "standalone": True,
        "voice_backend_bundled": args.edition == "voice",
        "gpu_runtime_bundled": args.edition == "voice",
        "model_weights_bundled": False, "user_settings_bundled": False,
        "update_helper_bundled": args.with_updater,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    run_frozen_checks(folder, output / "checks", args.edition, environment)
    print(json.dumps({"portable_folder": str(folder), "edition": args.edition, "version": version}, ensure_ascii=False))


if __name__ == "__main__":
    main()
