"""Assemble a new portable release and a model-free, verifiable update ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fanyiji_next.release_profile import RELEASE_DEFAULTS_NAME, write_release_defaults

APPLICATION = "fanyiji-voice-next"
UPDATE_MANIFEST = "update-manifest.json"
FORBIDDEN_PARTS = {"models", "settings", "user-data", "artifacts", "logs", "cache", "__pycache__", ".git", ".venv"}
PRIVATE_NAMES = {"settings.json", "auth.json", "credentials.json", "cookies", "cookies.sqlite", "login data", "web data"}
WEIGHT_SUFFIXES = {".onnx", ".gguf", ".safetensors", ".pth", ".pt", ".ckpt", ".tflite"}
WEIGHT_NAMES = {"model.bin", "pytorch_model.bin", "sentencepiece.bpe.model", "tokenizer.model", "spiece.model"}
PROGRAM_ROOT_FILES = {"FanyiJiVoiceNext.exe", "build-info.json", "使用说明.md", "README.md", "LICENSE", "LICENSE.txt", "公开包说明.md"}
PROGRAM_ROOT_DIRS = {"_internal", "licenses", "_updater"}
MODEL_FILES = {
    "asr": ("model.int8.onnx", "tokens.txt"),
    "translation": ("model.bin", "config.json", "shared_vocabulary.json", "sentencepiece.bpe.model"),
    "whisper-large-v3-ct2": ("model.bin", "config.json", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"),
    "nllb-200-distilled-1.3B-ct2": ("model.bin", "config.json", "shared_vocabulary.json"),
    "nllb-200-distilled-1.3B-tokenizer": ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"),
    "hymt2-1.8b": ("Hy-MT2-1.8B-Q8_0.gguf",),
    "audio-processing": ("Kim_Vocal_2.onnx", "gtcrn_simple.onnx"),
    "vad": ("silero_vad.onnx",),
}
MODEL_NOTICE_NAMES = {"LICENSE", "LICENSE.txt", "MODEL_LICENSE", "README.md", "provenance.json", "model-sources.json"}
MODEL_METADATA = (
    ("SenseVoiceSmall int8", "asr/model.int8.onnx"),
    ("M2M100 418M", "translation/model.bin"),
    ("Whisper large-v3 CTranslate2", "whisper-large-v3-ct2/model.bin"),
    ("NLLB-200-distilled-1.3B CTranslate2", "nllb-200-distilled-1.3B-ct2/model.bin"),
    ("Hy-MT2-1.8B Q8_0", "hymt2-1.8b/Hy-MT2-1.8B-Q8_0.gguf"),
    ("MDX Kim Vocal 2", "audio-processing/Kim_Vocal_2.onnx"),
    ("GTCRN", "audio-processing/gtcrn_simple.onnx"),
    ("Silero VAD", "vad/silero_vad.onnx"),
)
VC_RUNTIME_NAMES = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def regular_files(root: Path):
    """Reject links/junctions instead of accidentally following a private tree."""
    root = Path(root)
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise ValueError(f"需要普通目录：{root}")
    resolved = root.resolve()
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        for name in (*directories, *files):
            path = Path(directory) / name
            if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(resolved):
                raise ValueError(f"不允许符号链接、目录联接或越界文件：{path}")
        for name in files:
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError(f"不允许特殊文件：{path}")
            yield path


def safe_update_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if (not relative or "\\" in relative or ":" in relative or path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError(f"无效更新路径：{relative}")
    lowered = tuple(part.casefold() for part in path.parts)
    # CTranslate2 ships a Python module named models, not model weights.
    if any(part in FORBIDDEN_PARTS and not (index == 2 and lowered[:3] == ("_internal", "ctranslate2", "models"))
           for index, part in enumerate(lowered)):
        raise ValueError(f"更新包含用户数据或模型目录：{relative}")
    name = path.name.casefold()
    if (name in PRIVATE_NAMES or name.endswith(".settings.json") or path.suffix.casefold() in WEIGHT_SUFFIXES
            or name in WEIGHT_NAMES or path.suffix.casefold() in {".key", ".wav", ".mp3", ".flac"}
            or (path.suffix.casefold() in {".py", ".pyc"} and path.parts[0] not in {"_internal", "licenses"})
            or (path.suffix.casefold() == ".pem" and relative != "_internal/certifi/cacert.pem")):
        raise ValueError(f"更新包含模型、源码、录音或用户资料：{relative}")


def _copy_file(source: Path, source_root: Path, target_root: Path, relative: Path) -> None:
    if source.is_symlink() or source.is_junction() or not source.is_file() or not source.resolve().is_relative_to(source_root.resolve()):
        raise ValueError(f"输入文件不存在、是链接或越界：{source}")
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"发布包文件已存在：{target}")
    shutil.copy2(source, target)


def _program_files(folder: Path) -> list[tuple[Path, str]]:
    entries = []
    for file in regular_files(folder):
        relative = file.relative_to(folder).as_posix()
        if PurePosixPath(relative).parts[0].casefold() == "models" or relative == UPDATE_MANIFEST:
            continue
        safe_update_path(relative)
        entries.append((file, relative))
    return sorted(entries, key=lambda entry: entry[1])


def create_update_zip(folder: Path, target: Path, version: str) -> dict:
    """Include exactly the manifested program files, leaving installed models alone."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"更新包已存在：{target}")
    entries = _program_files(folder)
    manifest = {"schema_version": 1, "application": APPLICATION, "version": version,
                "files": [{"path": relative, "sha256": sha256(file), "size": file.stat().st_size} for file, relative in entries]}
    manifest_path = folder / UPDATE_MANIFEST
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"更新清单已存在：{manifest_path}")
    write_json(manifest_path, manifest)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as archive:
        for file, relative in entries:
            archive.write(file, relative)
        archive.write(manifest_path, UPDATE_MANIFEST)
    return {"name": target.name, "sha256": sha256(target), "size": target.stat().st_size}


def package_release(*, program_folder: Path, models: Path, runtime: Path, output: Path,
                    version: str, model_notices: Path | None = None, github_repo: str | None = None,
                    include_mdx: bool = True) -> dict:
    if type(include_mdx) is not bool:
        raise ValueError("include_mdx 必须是布尔值。")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9][A-Za-z0-9.-]*)?", version):
        raise ValueError("版本号必须为 semver 形式，例如 0.5.0。")
    if github_repo is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", github_repo):
        raise ValueError("GitHub 仓库必须为 owner/repo。")
    program_folder, models, runtime, output = (Path(path).absolute() for path in (program_folder, models, runtime, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"输出目录已存在，请指定新目录：{output}")
    for source in (program_folder, models, runtime, *([Path(model_notices)] if model_notices else [])):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("输出目录不能位于任一输入目录内。")
    source_files = list(regular_files(program_folder))
    if not (program_folder / "FanyiJiVoiceNext.exe").is_file():
        raise ValueError("输入缺少 FanyiJiVoiceNext.exe。")
    source_info = program_folder / "build-info.json"
    info = json.loads(source_info.read_text(encoding="utf-8")) if source_info.exists() else {}
    if info.get("version") not in {None, version}:
        raise ValueError("构建程序版本与发布版本不一致。")
    for file in source_files:
        relative = file.relative_to(program_folder)
        if not ((len(relative.parts) == 1 and relative.name in PROGRAM_ROOT_FILES) or relative.parts[0] in PROGRAM_ROOT_DIRS):
            raise ValueError(f"构建目录包含非白名单文件：{relative.as_posix()}")
        safe_update_path(relative.as_posix())
        if relative.as_posix() == "_internal/certifi/cacert.pem":
            pem = file.read_bytes()
            if b"PRIVATE KEY" in pem.upper() or b"-----BEGIN CERTIFICATE-----" not in pem:
                raise ValueError("certifi 公共证书文件含非证书内容或私钥。")
    model_entries = []
    # Enumerate only the runtime model allowlist, never source weights or recordings.
    list(regular_files(models))
    for directory, required in MODEL_FILES.items():
        for name in required:
            if not include_mdx and directory == "audio-processing" and name == "Kim_Vocal_2.onnx":
                continue
            source = models / directory / name
            if not source.is_file():
                raise ValueError(f"缺少运行模型文件：{directory}/{name}")
            model_entries.append(source)
        for name in sorted(MODEL_NOTICE_NAMES):
            candidate = models / directory / name
            if candidate.is_file() and candidate not in model_entries:
                model_entries.append(candidate)
    runtime_entries = [file for file in regular_files(runtime)
                       if file.suffix.casefold() == ".dll" or file.name == "llama-server.exe" or file.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))]
    if runtime / "llama-server.exe" not in runtime_entries:
        raise ValueError("运行库缺少 llama-server.exe。")
    vc_fallbacks = []
    for name in VC_RUNTIME_NAMES:
        # The child server searches its own directory, not the parent's
        # PyInstaller _internal directory. Prefer that canonical build copy
        # over older copies that Qt may bundle under its package directories.
        if (runtime / name).is_file():
            continue
        canonical = program_folder / "_internal" / name
        if canonical.is_file():
            source = canonical
        else:
            candidates = [file for file in source_files if file.relative_to(program_folder).parts[0] == "_internal"
                          and file.name.casefold() == name]
            if len(candidates) != 1:
                raise ValueError(f"运行库缺少 {name}，且冻结程序未提供唯一的匹配 DLL。")
            source = candidates[0]
        vc_fallbacks.append(source)
    notice_entries = list(regular_files(Path(model_notices))) if model_notices else []
    for file in notice_entries:
        safe_update_path((Path("licenses/model-notices") / file.relative_to(model_notices)).as_posix())
    output.mkdir(parents=True)
    folder = output / "FanyiJiVoiceNext"
    folder.mkdir()
    for file in source_files:
        _copy_file(file, program_folder, folder, file.relative_to(program_folder))
    for file in model_entries:
        _copy_file(file, models, folder, Path("models") / file.relative_to(models))
    for file in runtime_entries:
        _copy_file(file, runtime, folder, Path("runtime/llama-b11696-cuda12.4") / file.relative_to(runtime))
    for file in vc_fallbacks:
        _copy_file(file, program_folder, folder, Path("runtime/llama-b11696-cuda12.4") / file.name.casefold())
    for file in notice_entries:
        _copy_file(file, Path(model_notices), folder, Path("licenses/model-notices") / file.relative_to(model_notices))
    write_release_defaults(folder / RELEASE_DEFAULTS_NAME)
    selected_metadata = [(name, relative) for name, relative in MODEL_METADATA
                         if include_mdx or relative != "audio-processing/Kim_Vocal_2.onnx"]
    excluded_models = [] if include_mdx else [{"name": "MDX Kim Vocal 2", "path": "models/audio-processing/Kim_Vocal_2.onnx",
                       "bundled": False, "reason": "Not included in this public package at the user's request."}]
    source_metadata = folder / "models/audio-processing/model-sources.json"
    if source_metadata.exists():
        provenance = json.loads(source_metadata.read_text(encoding="utf-8"))
        for row in provenance.get("models", []):
            if isinstance(row, dict):
                row["bundled_in_this_package"] = include_mdx or row.get("file") != "Kim_Vocal_2.onnx"
        provenance["package_excluded_models"] = excluded_models
        write_json(source_metadata, provenance)
    if not include_mdx:
        (folder / "公开包说明.md").write_text(
            "# 公开包说明\n\n"
            "本公开包按用户要求不附带 MDX Kim Vocal 2 模型（`models/audio-processing/Kim_Vocal_2.onnx`）。"
            "其余识别、翻译、GTCRN 和 Silero VAD 运行模型均随包提供；没有复制训练用原始权重或用户录音。\n\n"
            "首次启动默认组合仍保持 **Whisper large-v3＋Hy-MT2 Q8＋MDX→GTCRN**，没有因发布而自动改动默认模式。"
            "因此首次使用该组合前，需要自行提供有权使用的 `Kim_Vocal_2.onnx`，放入上述位置或通过“声音与模型”选择文件；"
            "也可以自行在“声音与模型”选择其他声音处理模式。未补齐时，MDX 模式会明确提示缺少模型，不自动回退。\n\n"
            "NLLB-200-distilled-1.3B 遵循 **CC-BY-NC-4.0（非商业用途）**，其余模型同样应遵守各自许可证。"
            "完整官方许可原文、来源与转换说明见 `licenses/model-notices/` 和模型目录附带的说明。"
            "本版本仍有已知识别和残句翻译问题；打包发布不表示此前准确性与 3 秒延迟联合验收通过。\n",
            encoding="utf-8")
    inventory = {"schema_version": 1, "application": APPLICATION, "version": version,
                 "note": "Local model inventory; redistribution rights are stated in bundled licenses and notices, not inferred from a model file.",
                 "mdx_bundled": include_mdx, "excluded_models": excluded_models,
                 "models": [{"name": name, "path": f"models/{relative}", "size": (folder / "models" / relative).stat().st_size,
                             "sha256": sha256(folder / "models" / relative)} for name, relative in selected_metadata]}
    write_json(folder / "models/model-inventory.json", inventory)
    info_path = folder / "build-info.json"
    info.update(application=APPLICATION, version=version, edition="voice", standalone=True,
                model_weights_bundled=True, user_settings_bundled=False, portable_release=True, mdx_model_bundled=include_mdx,
                llama_vc_runtime_bundled=True)
    write_json(info_path, info)
    update = create_update_zip(folder, output / f"FanyiJiVoiceNext-{version}-update.zip", version)
    # The full manifest is outside the executable folder and update ZIP, so it
    # can describe every packaged file without introducing a self-hash cycle.
    known_hashes = {row["path"]: row["sha256"] for row in inventory["models"]}
    program_manifest = json.loads((folder / UPDATE_MANIFEST).read_text(encoding="utf-8"))
    known_hashes.update({row["path"]: row["sha256"] for row in program_manifest["files"]})
    full_manifest = {"schema_version": 1, "application": APPLICATION, "version": version,
                     "files": [{"path": file.relative_to(folder).as_posix(), "size": file.stat().st_size,
                                "sha256": known_hashes.get(file.relative_to(folder).as_posix()) or sha256(file)}
                               for file in regular_files(folder)]}
    write_json(output / "full-manifest.json", full_manifest)
    if github_repo:
        update["url"] = f"https://github.com/{github_repo}/releases/download/v{version}/{update['name']}"
        write_json(output / "update-feed.json", {"schema_version": 1, "application": APPLICATION,
                   "platform": "windows-x64", "version": version, "program": update})
    summary = {"schema_version": 1, "application": APPLICATION, "version": version,
               "portable_folder": str(folder), "program_update": update,
               "model_count": len(selected_metadata), "mdx_bundled": include_mdx, "excluded_models": excluded_models,
               "llama_vc_runtime_bundled": True, "llama_vc_runtime_files": list(VC_RUNTIME_NAMES),
               "model_source_weights_bundled": False,
               "user_settings_bundled": False, "full_manifest": "full-manifest.json",
               "archive_compression_pending": True}
    write_json(output / "package-report.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="从新的冻结程序构建内置模型便携目录和纯程序更新 ZIP；不覆盖已有输出。")
    for name in ("program-folder", "models", "runtime", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--model-notices", type=Path)
    parser.add_argument("--github-repo", help="已确认的 GitHub owner/repo；未指定时不生成更新 feed URL")
    parser.add_argument("--exclude-mdx", dest="include_mdx", action="store_false", help="公开包不附带 Kim_Vocal_2.onnx；保留默认模式并提供补齐说明")
    args = parser.parse_args(argv)
    try:
        report = package_release(**vars(args))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
