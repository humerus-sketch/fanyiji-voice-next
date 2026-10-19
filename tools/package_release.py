"""Assemble program-only portable releases and verifiable updates."""
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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_input_guard import assert_build_input
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fanyiji_next.release_profile import RELEASE_DEFAULTS_NAME, write_release_defaults

APPLICATION = "fanyiji-voice-next"
UPDATE_MANIFEST = "update-manifest.json"
FORBIDDEN_PARTS = {"models", "settings", "user-data", "artifacts", "logs", "cache", "__pycache__", ".git", ".venv"}
PRIVATE_NAMES = {"settings.json", "auth.json", "credentials.json", "cookies", "cookies.sqlite", "login data", "web data"}
WEIGHT_SUFFIXES = {".onnx", ".gguf", ".safetensors", ".pth", ".pt", ".ckpt", ".tflite", ".bin", ".model"}
WEIGHT_NAMES = {"model.bin", "pytorch_model.bin", "sentencepiece.bpe.model", "tokenizer.model", "spiece.model"}
# These are Unicode normalization tables shipped by SentencePiece, not learned
# model weights. No other .bin payload is permitted in a program package.
RUNTIME_DATA_FILES = {f"_internal/sentencepiece/package_data/{name}.bin"
                      for name in ("nfkc", "nfkc_cf", "nmt_nfkc", "nmt_nfkc_cf")}
RUNTIME_MODEL_CODE_PREFIXES = {("_internal", package, "models")
                               for package in ("ctranslate2", "transformers", "funasr", "torchaudio")}
RUNTIME_MODEL_CODE_PREFIXES.add(("_internal", "torchaudio", "prototype", "models"))
# PyTorch names generated heuristic implementation modules "artifacts". Only
# these shipped source files are runtime code; arbitrary artifacts stay private.
TORCH_AUTOHEURISTIC_FILES = {
    "_internal/torch/_inductor/autoheuristic/artifacts/" + name + ".py"
    for name in ("__init__", "_MixedMMA100", "_MixedMMH100", "_MMRankingA100", "_MMRankingH100", "_PadMMA100")
}
PUBLIC_CERTIFICATE_FILES = {"_internal/certifi/cacert.pem",
                            "_internal/aliyunsdkcore/vendored/requests/packages/certifi/cacert.pem"}
PROGRAM_ROOT_FILES = {"FanyiJiVoiceNext.exe", "build-info.json", "使用说明.md", "README.md", "LICENSE", "LICENSE.txt", "公开包说明.md"}
PROGRAM_ROOT_DIRS = {"_internal", "licenses", "_updater", "runtime"}
VC_RUNTIME_NAMES = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
FORBIDDEN_RUNTIME_PREFIXES = {("_internal", "jieba", "lac_small", "model_baseline")}


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
    if any(lowered[:len(prefix)] == prefix for prefix in FORBIDDEN_RUNTIME_PREFIXES):
        raise ValueError(f"更新包含依赖附带的模型参数：{relative}")
    # Runtime libraries also name implementation packages "models". Only these
    # precise code prefixes are exempt; weight suffix checks still apply.
    if any((part in FORBIDDEN_PARTS or part.endswith("_models"))
           and not ((lowered[:index + 1] in RUNTIME_MODEL_CODE_PREFIXES
                     and path.suffix.casefold() in {".py", ".pyc", ".pyd"})
                    or (index == 4 and part == "artifacts" and relative in TORCH_AUTOHEURISTIC_FILES))
           for index, part in enumerate(lowered)):
        raise ValueError(f"更新包含用户数据或模型目录：{relative}")
    name = path.name.casefold()
    if (name in PRIVATE_NAMES or name.endswith(".settings.json")
            or (path.suffix.casefold() in WEIGHT_SUFFIXES and relative not in RUNTIME_DATA_FILES)
            or name in WEIGHT_NAMES or path.suffix.casefold() in {".key", ".wav", ".mp3", ".flac"}
            or (path.suffix.casefold() in {".py", ".pyc"} and path.parts[0] not in {"_internal", "licenses"})
            or (path.suffix.casefold() == ".pem" and relative not in PUBLIC_CERTIFICATE_FILES)):
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
        if relative == UPDATE_MANIFEST:
            continue
        safe_update_path(relative)
        entries.append((file, relative))
    return sorted(entries, key=lambda entry: entry[1])


def create_update_zip(folder: Path, target: Path, version: str) -> dict:
    """Include exactly the manifested program files, leaving installed models alone."""
    assert_build_input(folder, purpose="update program folder")
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


def package_release(*, program_folder: Path, runtime: Path, output: Path, models: Path | None = None,
                    version: str, model_notices: Path | None = None, github_repo: str | None = None,
                    include_mdx: bool = True) -> dict:
    if type(include_mdx) is not bool:
        raise ValueError("include_mdx 必须是布尔值。")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9][A-Za-z0-9.-]*)?", version):
        raise ValueError("版本号必须为 semver 形式，例如 0.5.0。")
    if github_repo is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", github_repo):
        raise ValueError("GitHub 仓库必须为 owner/repo。")
    # Kept as compatibility-only arguments for old release scripts. A supplied
    # model directory is never enumerated, required, hashed or copied.
    program_folder, runtime, output = (Path(path).absolute() for path in (program_folder, runtime, output))
    assert_build_input(program_folder, purpose="portable program folder")
    assert_build_input(runtime, purpose="portable native runtime")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"输出目录已存在，请指定新目录：{output}")
    for source in (program_folder, runtime, *([Path(model_notices)] if model_notices else [])):
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
        if relative.as_posix() in PUBLIC_CERTIFICATE_FILES:
            pem = file.read_bytes()
            if b"PRIVATE KEY" in pem.upper() or b"-----BEGIN CERTIFICATE-----" not in pem:
                raise ValueError("certifi 公共证书文件含非证书内容或私钥。")
            if relative.as_posix() != "_internal/certifi/cacert.pem":
                canonical = program_folder / "_internal/certifi/cacert.pem"
                if not canonical.is_file() or sha256(file) != sha256(canonical):
                    raise ValueError("嵌套 certifi 公共证书与标准证书包不匹配。")
    runtime_entries = [file for file in regular_files(runtime)
                       if file.suffix.casefold() == ".dll" or file.name == "llama-server.exe" or file.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))]
    if runtime / "llama-server.exe" not in runtime_entries:
        raise ValueError("运行库缺少 llama-server.exe。")
    for file in runtime_entries:
        safe_update_path((Path("runtime/llama-b11696-cuda12.4") / file.relative_to(runtime)).as_posix())
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
        # The explicit runtime input is authoritative. A modern fresh build may
        # already contain a copy; do not duplicate it or overwrite that tree.
        relative = file.relative_to(program_folder)
        if relative.parts[0] != "runtime":
            _copy_file(file, program_folder, folder, relative)
    for file in runtime_entries:
        _copy_file(file, runtime, folder, Path("runtime/llama-b11696-cuda12.4") / file.relative_to(runtime))
    for file in vc_fallbacks:
        _copy_file(file, program_folder, folder, Path("runtime/llama-b11696-cuda12.4") / file.name.casefold())
    for file in notice_entries:
        _copy_file(file, Path(model_notices), folder, Path("licenses/model-notices") / file.relative_to(model_notices))
    write_release_defaults(folder / RELEASE_DEFAULTS_NAME)
    (folder / "公开包说明.md").write_text(
        "# 模型按需下载\n\n"
        "本安装包和软件更新包只包含程序与运行库，不内置识别、翻译、VAD、降噪或分离权重。\n\n"
        "首次使用请打开“声音与模型”中的模型下载入口，选择 A 或 B 方案并下载所需模型。"
        "A：Qwen3-ASR-0.6B-hf BF16＋TranslateGemma 4B Q4_K_M；"
        "B：Fun-ASR-Nano-2512 BF16＋同一份 TranslateGemma 4B Q4_K_M。"
        "两方案共用翻译与 VAD 文件，不重复下载已校验文件。\n\n"
        "下载模型保存在主程序同级的 `<主程序文件名（不含 .exe）>_models` 目录。"
        "例如 `FanyiJiVoiceNext.exe` 对应 `FanyiJiVoiceNext_models`；"
        "改名为 `直播字幕.exe` 后对应 `直播字幕_models`。前缀跟随主程序名称，固定后缀为 `_models`。"
        "如需复用改名前的下载，可一并重命名模型目录；程序不会自行删除旧目录。\n\n"
        "程序更新保留已下载模型和用户设置，不把权重混入更新清单。"
        "程序不会在启动时自动下载模型或开启采音；模型及量化文件应遵守各自许可证。"
        "新组合的准确率、演唱识别和三秒端到端延迟需要实际验收，不能以下载或启动成功代替。\n",
        encoding="utf-8")
    info_path = folder / "build-info.json"
    info.update(application=APPLICATION, version=version, edition="voice", standalone=True,
                model_weights_bundled=False, user_settings_bundled=False, portable_release=True, mdx_model_bundled=False,
                external_model_directory="<executable-stem>_models", model_download_required=True,
                llama_vc_runtime_bundled=True)
    write_json(info_path, info)
    update = create_update_zip(folder, output / f"FanyiJiVoiceNext-{version}-update.zip", version)
    # The full manifest is outside the executable folder and update ZIP, so it
    # can describe every packaged file without introducing a self-hash cycle.
    known_hashes = {}
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
               "model_count": 0, "mdx_bundled": False, "external_model_directory": "<executable-stem>_models",
               "model_weights_bundled": False, "model_download_required": True,
               "llama_vc_runtime_bundled": True, "llama_vc_runtime_files": list(VC_RUNTIME_NAMES),
               "model_source_weights_bundled": False,
               "user_settings_bundled": False, "full_manifest": "full-manifest.json",
               "archive_compression_pending": True}
    write_json(output / "package-report.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="从新的冻结程序构建无模型便携目录和纯程序更新 ZIP；模型在软件内按需下载。")
    for name in ("program-folder", "runtime", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--models", type=Path, help="兼容旧命令；忽略此目录，不读取或打包模型")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model-notices", type=Path)
    parser.add_argument("--github-repo", help="已确认的 GitHub owner/repo；未指定时不生成更新 feed URL")
    parser.add_argument("--exclude-mdx", dest="include_mdx", action="store_false", help="兼容旧命令；所有发布包均不附任何模型")
    args = parser.parse_args(argv)
    try:
        report = package_release(**vars(args))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
