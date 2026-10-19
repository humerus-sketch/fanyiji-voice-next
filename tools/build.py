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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_input_guard import assert_build_input
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
AB_MODULES = ("torch", "torchaudio", "transformers", "funasr", "safetensors", "librosa", "soundfile")
AB_DISTRIBUTIONS = AB_MODULES
NORMALIZATION_TABLES = {f"_internal/sentencepiece/package_data/{name}.bin"
                        for name in ("nfkc", "nfkc_cf", "nmt_nfkc", "nmt_nfkc_cf")}
MODEL_CODE_PREFIXES = {("_internal", package, "models")
                       for package in ("ctranslate2", "transformers", "funasr", "torchaudio")}
MODEL_CODE_PREFIXES.add(("_internal", "torchaudio", "prototype", "models"))
FORBIDDEN_RUNTIME_PREFIXES = {("_internal", "jieba", "lac_small", "model_baseline")}
PUBLIC_CERTIFICATE_FILES = {"_internal/certifi/cacert.pem",
                            "_internal/aliyunsdkcore/vendored/requests/packages/certifi/cacert.pem"}
PROTOTYPE_EXCLUDES = ("fanyiji_next.app", "fanyiji_next.controller", "fanyiji_next.audio_runtime",
                      "fanyiji_next.recognition_engines", "fanyiji_next.translation_engines",
                      "fanyiji_next.verification", "fanyiji_next.speech_gate", "fanyiji_next.native_runtime",
                      "nvidia", "nvidia.cublas", "nvidia.cudnn", *VOICE_MODULES, *AB_MODULES)


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


def license_packages(edition, with_ab_engines=False):
    return CORE_LICENSES + (VOICE_LICENSES if edition == "voice" else ()) + (AB_DISTRIBUTIONS if with_ab_engines else ())


def ab_runtime_metadata():
    """Fail before building if the explicitly requested A/B runtime is missing."""
    return {name: importlib.metadata.distribution(name).version for name in AB_DISTRIBUTIONS}


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


def build_command(output, work, edition, with_ab_engines=False):
    if getattr(sys, "frozen", False):
        raise ValueError("请使用独立 Python 从项目源码构建，不能用验证 EXE 构建。")
    assert_build_input(ROOT, purpose="source checkout")
    assert_build_input(Path(sys.executable), purpose="build interpreter")
    if edition not in {"prototype", "voice"}:
        raise ValueError("Unknown edition: " + edition)
    if with_ab_engines and edition != "voice":
        raise ValueError("只有语音版支持 A/B 模型运行库。")
    if with_ab_engines:
        ab_runtime_metadata()
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
        if with_ab_engines:
            command.extend(("--additional-hooks-dir", str(ROOT / "tools" / "pyinstaller_hooks")))
            # Collect implementation and native libraries, never package data:
            # these libraries can include pretrained/test weight files.
            for package in AB_MODULES:
                command.extend(("--collect-submodules", package))
            for package in ("torch", "torchaudio", "safetensors", "soundfile"):
                command.extend(("--collect-binaries", package))
            for package in AB_DISTRIBUTIONS:
                command.extend(("--copy-metadata", package))
        # Namespace packages do not reliably expose native DLLs to collection.
        # Include every split cuDNN library rather than just the loader DLL.
        for source, destination in nvidia_binaries():
            command.extend(("--add-binary", f"{source}:{destination}"))
    command.append(str(ROOT / ("run_prototype.py" if edition == "prototype" else "run.py")))
    return command


def build_update_helper(folder, work, environment):
    """A standalone helper can replace the main EXE after its process exits."""
    assert_build_input(ROOT / "tools/update_helper.py", purpose="helper source")
    assert_build_input(Path(sys.executable), purpose="helper build interpreter")
    destination = folder / "_updater"
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
               "--name", "FanyiJiUpdate", "--paths", str(ROOT / "src"),
               "--distpath", str(destination), "--workpath", str(work / "helper-work"),
               "--specpath", str(work), str(ROOT / "tools" / "update_helper.py")]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    if not (destination / "FanyiJiUpdate.exe").is_file():
        raise RuntimeError("独立更新程序未成功构建。")


def collect_llama_runtime(folder):
    """Make the A/B build usable without borrowing an old release's runtime."""
    tools_path = str(Path(__file__).resolve().parent)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from package_release import VC_RUNTIME_NAMES, regular_files, safe_update_path

    source_root = ROOT / "runtime" / "llama-b11696-cuda12.4"
    assert_build_input(source_root, purpose="native runtime")
    entries = [path for path in regular_files(source_root)
               if path.suffix.casefold() == ".dll" or path.name == "llama-server.exe"
               or path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))]
    if source_root / "llama-server.exe" not in entries:
        raise RuntimeError("A/B 运行库缺少 llama-server.exe。")
    destination = folder / "runtime" / source_root.name
    copies = [(path, destination / path.relative_to(source_root)) for path in entries]
    for name in VC_RUNTIME_NAMES:
        if not (source_root / name).is_file():
            source = folder / "_internal" / name
            if not source.is_file():
                raise RuntimeError(f"A/B 运行库缺少 {name}。")
            copies.append((source, destination / name))
    for source, target in copies:
        safe_update_path(target.relative_to(folder).as_posix())
        if target.exists() or target.is_symlink() or source.is_symlink() or source.is_junction():
            raise RuntimeError(f"运行库路径已存在或是链接：{target}")
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def collect_licenses(folder, edition, with_ab_engines=False):
    import re

    from packaging.requirements import Requirement

    licenses = folder / "licenses"
    notice_name = re.compile(r"(?:LICENSE|LICENCE|COPYING|NOTICE|COPYRIGHT|AUTHORS)(?:[.-][A-Z0-9][A-Z0-9._-]*)?", re.I)
    # These nonstandard basenames are actual notices in the pinned runtime
    # wheels. A substring such as ModelScope's license_plate_detection is code.
    additional_notices = {"copying_qhull.txt", "license_dop", "licenses.third-party",
                          "licenseref-qt-commercial.txt", "thirdpartynotices.txt"}
    code_suffixes = {".py", ".pyc", ".pyo", ".pyw", ".pyd", ".dll", ".so", ".dylib", ".exe", ".com",
                     ".lib", ".a", ".o", ".obj", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh",
                     ".js", ".ts", ".java", ".class", ".bat", ".cmd", ".ps1", ".sh"}
    pending = list(license_packages(edition, True) if with_ab_engines else license_packages(edition))
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
            relative = Path(str(item))
            if ("__pycache__" in (part.casefold() for part in relative.parts)
                    or any(suffix.casefold() in code_suffixes for suffix in relative.suffixes)):
                continue
            if notice_name.fullmatch(relative.name) or relative.name.casefold() in additional_notices:
                source = Path(distribution.locate_file(item))
                if source.is_file():
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
    if relative.as_posix() not in PUBLIC_CERTIFICATE_FILES:
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
    forbidden = {".onnx", ".gguf", ".safetensors", ".pth", ".pt", ".ckpt", ".tflite", ".bin", ".model", ".key", ".pem"}
    private = {"settings.json", "auth.json", "credentials.json", "cookies", "cookies.sqlite", "login data", "web data"}
    model_files = {"model.bin", "pytorch_model.bin", "sentencepiece.bpe.model", "tokenizer.model", "spiece.model"}
    for file in folder.rglob("*"):
        relative = file.relative_to(folder)
        if len(relative.parts) == 2 and relative.parts[0] == "_internal" and file.name.casefold() == "icuuc.dll":
            raise RuntimeError("输出目录带入了可遮蔽 Windows 系统 ICU 的同名 DLL；请检查构建环境 PATH。")
        lowered = tuple(part.casefold() for part in relative.parts)
        if "__pycache__" in lowered or any(lowered[:len(prefix)] == prefix for prefix in FORBIDDEN_RUNTIME_PREFIXES):
            raise RuntimeError("输出目录出现模型或用户资料，请检查打包依赖：" + str(relative))
        if any((part.endswith("_models") or part == "models")
               and not (lowered[:index + 1] in MODEL_CODE_PREFIXES
                        and (file.is_dir() or file.suffix.casefold() in {".py", ".pyc", ".pyd"}))
               for index, part in enumerate(lowered[:-1])):
            raise RuntimeError("输出目录出现模型或用户资料，请检查打包依赖：" + str(relative))
        if file.is_file() and (file.suffix.lower() in forbidden or file.name.lower() in private | model_files):
            if relative.as_posix() in NORMALIZATION_TABLES:
                continue
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


def registered_primary_for_build(edition, custom_output):
    if edition != "voice" or custom_output or os.environ.get("GITHUB_ACTIONS") == "true":
        return None
    from fanyiji_next import managed_installation as managed
    from fanyiji_next.app_update import _version_key
    registry = ROOT / managed.REGISTRY_NAME
    if not os.path.lexists(registry):
        return None
    value = managed._json(managed._read(registry))
    _version_key(value.get("current_version"))
    return managed.read_managed_installation(ROOT / "dist" / value["current_version"] / "FanyiJiVoiceNext")


def primary_program_running(executables):
    if os.name != "nt":
        raise RuntimeError("当前平台无法核对主用进程；保留新构建但未切换主用登记。")
    # Query executable paths only; never read or print process command lines.
    script = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
              "$ErrorActionPreference='Stop';"
              "ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process "
              "-Filter \"Name = 'FanyiJiVoiceNext.exe'\" | Select-Object -ExpandProperty ExecutablePath)")
    powershell = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, encoding="utf-8", check=True, timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    paths = json.loads(result.stdout)
    if not isinstance(paths, list) or any(not isinstance(path, str) or not path for path in paths):
        raise RuntimeError("无法可靠核对主用进程路径；保留新构建但未切换主用登记。")
    expected = {os.path.normcase(str(path)) for path in executables}
    return any(os.path.normcase(str(Path(path).absolute())) in expected for path in paths)


def activate_built_primary(registered, folder, version):
    if registered is None:
        return False
    from fanyiji_next import managed_installation as managed
    from fanyiji_next.app_update import _installation_lock
    with _installation_lock(registered.project_root):
        current = managed.read_managed_installation(registered.program_dir)
        if current is None or current.registry_bytes != registered.registry_bytes:
            raise RuntimeError("构建期间主用登记发生变化；保留新构建但未切换登记。")
        expected = registered.project_root / "dist" / version / "FanyiJiVoiceNext"
        if folder != expected:
            raise RuntimeError("构建输出不是本项目对应版本目录，未切换主用登记。")
        managed._plain(folder)
        executable = managed._plain(folder / "FanyiJiVoiceNext.exe")
        if not executable.is_file():
            raise RuntimeError("新构建 EXE 不存在，未切换主用登记。")
        build = managed._json(managed._read(folder / "build-info.json"))
        if build.get("application") != managed.APPLICATION or build.get("version") != version or build.get("edition") != "voice":
            raise RuntimeError("新构建版本信息不一致，未切换主用登记。")
        if primary_program_running((current.program_dir / "FanyiJiVoiceNext.exe", executable)):
            raise RuntimeError("主用程序或新构建仍在运行；保留新构建，关闭后再切换主用登记。")
        # Recheck immediately before the one atomic pointer replacement.
        if managed._read(registered.registry_path) != registered.registry_bytes:
            raise RuntimeError("主用登记在进程检查期间发生变化，未切换登记。")
        value = dict(registered.value, current_version=version,
                     program_relative=f"dist/{version}/FanyiJiVoiceNext")
        managed._write(registered.registry_path, managed._encoded(value))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="分别构建独立原型或语音版，始终保留已有输出。")
    parser.add_argument("--edition", choices=("prototype", "voice"), default="prototype")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-updater", action="store_true", help="同时构建独立的退出后更新程序（仅语音版）")
    parser.add_argument("--with-ab-engines", action="store_true", help="打包已安装的 A/B 推理运行库；缺少依赖时报错，不下载或内置模型")
    args = parser.parse_args(argv)
    if args.with_updater and args.edition != "voice":
        parser.error("只有语音版支持独立更新程序。")
    if args.with_ab_engines and args.edition != "voice":
        parser.error("只有语音版支持 A/B 模型运行库。")
    version = project_version()
    default_name = version if args.edition == "voice" else f"{version}-prototype"
    output = (args.output or ROOT / "dist" / default_name).resolve()
    if output.exists() or output.is_symlink():
        parser.error("输出目录已存在。请用 --output 指定一个新的空路径，以保留已有程序。")
    entrypoint = ROOT / ("run_prototype.py" if args.edition == "prototype" else "run.py")
    if not entrypoint.is_file():
        parser.error("构建入口不存在：" + str(entrypoint))
    registered = registered_primary_for_build(args.edition, args.output is not None)
    try:
        packages = license_packages(args.edition, True) if args.with_ab_engines else license_packages(args.edition)
        for package in packages:
            importlib.metadata.distribution(package)
        ab_versions = ab_runtime_metadata() if args.with_ab_engines else {}
        work = ROOT / "build" / f"{args.edition}-{output.name}-{uuid4().hex[:8]}"
        command = build_command(output, work, args.edition, args.with_ab_engines)
    except importlib.metadata.PackageNotFoundError as exc:
        parser.error(f"构建依赖缺失：{exc.name}；请先在新项目独立环境中安装对应依赖。")
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True)
    environment = build_environment()
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    folder = output / "FanyiJiVoiceNext"
    if args.with_updater:
        build_update_helper(folder, work, environment)
    if args.with_ab_engines:
        collect_llama_runtime(folder)
    shutil.copy2(ROOT / "README.md", folder / "使用说明.md")
    if args.with_ab_engines:
        collect_licenses(folder, args.edition, True)
    else:
        collect_licenses(folder, args.edition)
    assert_public_payload(folder)
    (folder / "build-info.json").write_text(json.dumps({
        "application": "fanyiji-voice-next", "version": version, "edition": args.edition, "standalone": True,
        "voice_backend_bundled": args.edition == "voice",
        "gpu_runtime_bundled": args.edition == "voice",
        "model_weights_bundled": False, "user_settings_bundled": False,
        "ab_engines_bundled": args.with_ab_engines, "ab_runtime_versions": ab_versions,
        "external_model_directory": "<executable-stem>_models",
        "update_helper_bundled": args.with_updater,
        "build_source": "project_source", "build_entrypoint": entrypoint.name,
        "build_from_validation_binary": False,
        "build_environment": "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    run_frozen_checks(folder, output / "checks", args.edition, environment)
    activated = activate_built_primary(registered, folder, version)
    print(json.dumps({"portable_folder": str(folder), "edition": args.edition, "version": version,
                      "primary_registration_updated": activated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
