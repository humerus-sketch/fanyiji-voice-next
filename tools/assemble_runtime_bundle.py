"""Reuse measured ZIP payloads, overlay fresh executables, and split raw bytes.

The source archive and its streamed SHA manifest must both be pinned. Retained
entries keep their compressed bytes; no runtime library is recompressed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import zipfile

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import package_release as release
from fanyiji_next.release_profile import release_defaults_payload

BLOCK = 4 * 1024 * 1024
GIB = 1024**3
READY = "_updater/runtime-ready.json"
BOOTSTRAP = "_updater/runtime-bootstrap.json"
GENERATED = {READY, "release-defaults.json", release.UPDATE_MANIFEST}
EXTRA_ROOT_FILES = {"release-defaults.json", release.UPDATE_MANIFEST}
SHA = re.compile(r"[a-f0-9]{64}\Z")


def _ordinary(path):
    for node in (path, *path.parents):
        details = node.lstat()
        if getattr(details, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT or stat.S_ISLNK(details.st_mode):
            raise ValueError("Bundle input/output passes through a link: " + str(node))


def _identity(path):
    _ordinary(path)
    item = path.stat()
    if not stat.S_ISREG(item.st_mode):
        raise ValueError("Expected a regular file: " + str(path))
    return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bytes_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _exclude(name):
    path = PurePosixPath(name)
    parts = tuple(part.casefold() for part in path.parts)
    return (name.casefold() == BOOTSTRAP.casefold() or "__pycache__" in parts
            or any(part.endswith("_models") for part in parts)
            or (parts and parts[0] in {"models", "cache"})
            or any(parts[:len(prefix)] == prefix for prefix in release.FORBIDDEN_RUNTIME_PREFIXES))


def _public_path(name):
    release.safe_update_path(name)
    path = PurePosixPath(name)
    if (path.as_posix() != name or any(ord(char) < 32 for char in name)
            or any(part.endswith((".", " ")) or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part, re.I)
                   or any(char in part for char in '<>"|?*') for part in path.parts)):
        raise ValueError("Unsafe Windows program path: " + name)
    if ((len(path.parts) == 1 and name not in release.PROGRAM_ROOT_FILES | EXTRA_ROOT_FILES)
            or (len(path.parts) > 1 and path.parts[0] not in release.PROGRAM_ROOT_DIRS)):
        raise ValueError("Not a public program path: " + name)


def _case_paths(names):
    seen = {}
    for name in names:
        parts = PurePosixPath(name).parts
        for length in range(1, len(parts) + 1):
            value = "/".join(parts[:length])
            previous = seen.setdefault(value.casefold(), value)
            if previous != value:
                raise ValueError("Case-colliding program paths")


def _validate_source(archive, manifest):
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing streamed source manifest")
    mapped, names = {}, set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                or type(row.get("size")) is not int or row["size"] < 0
                or not isinstance(row.get("sha256"), str) or not SHA.fullmatch(row["sha256"])
                or type(row.get("compressed_size")) is not int or row["compressed_size"] < 0):
            raise ValueError("Invalid streamed source manifest row")
        name = row["path"]
        # Validate lexical safety even for deliberately excluded model data.
        path = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or path.is_absolute() or path.as_posix() != name
                or any(part in {"", ".", ".."} for part in name.split("/")) or "\x00" in name):
            raise ValueError("Unsafe source ZIP path")
        if name.casefold() in names:
            raise ValueError("Duplicate or case-colliding source path")
        names.add(name.casefold())
        mapped[name] = row
    entries = archive.infolist()
    if len(entries) != len(mapped) or {entry.filename for entry in entries} != set(mapped):
        raise ValueError("Source ZIP differs from the pinned streamed manifest")
    for info in entries:
        row = mapped[info.filename]
        if (info.is_dir() or info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16)
                or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or (info.file_size, info.compress_size) != (row["size"], row["compressed_size"])):
            raise ValueError("Unsupported or mismatched source ZIP entry: " + info.filename)
    return mapped


def _without_zip64_extra(extra):
    result, offset = bytearray(), 0
    while offset < len(extra):
        if offset + 4 > len(extra):
            raise ValueError("Malformed ZIP extra field")
        kind, size = struct.unpack_from("<HH", extra, offset)
        end = offset + 4 + size
        if end > len(extra):
            raise ValueError("Truncated ZIP extra field")
        if kind != 1:
            result.extend(extra[offset:end])
        offset = end
    return bytes(result)


def _payload_offset(stream, source, info):
    stream.seek(info.header_offset)
    header = stream.read(30)
    if len(header) != 30:
        raise ValueError("Truncated source local ZIP header")
    fields = struct.unpack("<4s5H3I2H", header)
    if fields[0] != b"PK\x03\x04" or fields[2] & 1 or fields[3] != info.compress_type:
        raise ValueError("Invalid source local ZIP header")
    filename = stream.read(fields[9]).decode("utf-8" if fields[2] & 0x800 else "cp437")
    if filename != info.filename:
        raise ValueError("Source local and central ZIP names differ")
    start = info.header_offset + 30 + fields[9] + fields[10]
    if start + info.compress_size > source.start_dir:
        raise ValueError("Source compressed payload overlaps central directory")
    return start


def _copy_raw(source, stream, target, info):
    """Use ZipFile's ZIP64 central-directory writer, retaining the DEFLATE stream."""
    offset = _payload_offset(stream, source, info)
    cloned = copy.copy(info)
    cloned.extra = _without_zip64_extra(cloned.extra)
    cloned.flag_bits &= ~8  # Known sizes go in the new header; no descriptor.
    cloned.header_offset = target.fp.tell()
    target._writecheck(cloned)
    target._didModify = True
    zip64 = max(cloned.file_size, cloned.compress_size) > zipfile.ZIP64_LIMIT
    target.fp.write(cloned.FileHeader(zip64=zip64))
    stream.seek(offset)
    remaining = cloned.compress_size
    while remaining:
        block = stream.read(min(BLOCK, remaining))
        if not block:
            raise ValueError("Truncated retained compressed payload")
        target.fp.write(block)
        remaining -= len(block)
    target.filelist.append(cloned)
    target.NameToInfo[cloned.filename] = cloned
    target.start_dir = target.fp.tell()


def _write_overlay(archive, path, name):
    before = _identity(path)
    checksum = hashlib.sha256()
    info = zipfile.ZipInfo.from_file(path, arcname=name)
    info.compress_type = zipfile.ZIP_DEFLATED
    info._compresslevel = 5
    count = 0
    with path.open("rb") as stream, archive.open(info, "w", force_zip64=True) as target:
        while block := stream.read(BLOCK):
            target.write(block)
            checksum.update(block)
            count += len(block)
    if _identity(path) != before or count != before[2]:
        raise ValueError("Overlay changed during assembly: " + name)
    return {"path": name, "size": count, "sha256": checksum.hexdigest()}


def _split(archive, size, version, repository):
    parts, checksum, number = [], hashlib.sha256(), 1
    with archive.open("rb") as stream:
        while True:
            first = stream.read(min(size, BLOCK))
            if not first:
                break
            path = archive.with_name(archive.name + f".{number:03d}")
            digest, count = hashlib.sha256(), 0
            with path.open("xb") as output:
                block = first
                while block:
                    output.write(block)
                    checksum.update(block)
                    digest.update(block)
                    count += len(block)
                    block = stream.read(min(size - count, BLOCK)) if count < size else b""
            parts.append({"name": path.name, "url": f"https://github.com/{repository}/releases/download/v{version}/{path.name}",
                          "size": count, "sha256": digest.hexdigest()})
            number += 1
    return parts, checksum.hexdigest()


def assemble_runtime_bundle(*, source_zip: Path, source_manifest: Path, source_sha256: str,
                            source_manifest_sha256: str, overlay: Path, output_zip: Path,
                            version: str, build_id: str, repository: str,
                            split_size: int = 1536 * 1024**2) -> dict:
    source_zip, source_manifest, overlay, output_zip = (Path(path).absolute()
                                                      for path in (source_zip, source_manifest, overlay, output_zip))
    if (not re.fullmatch(r"\d+\.\d+\.\d+", version) or not re.fullmatch(r"[A-Za-z0-9._-]+", build_id)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repository)
            or not re.fullmatch(r"[A-Za-z0-9._-]+\.zip", output_zip.name)
            or type(split_size) is not int or not 0 < split_size < 2 * GIB
            or not SHA.fullmatch(source_sha256) or not SHA.fullmatch(source_manifest_sha256)):
        raise ValueError("Invalid runtime bundle version, identity, hashes, or part size")
    release.assert_build_input(source_zip, purpose="runtime assembly source archive")
    release.assert_build_input(overlay, purpose="runtime assembly program overlay")
    for path in (source_zip, source_manifest, overlay, output_zip.parent):
        _ordinary(path)
    reserved = [output_zip, *(output_zip.parent / name for name in
                ("runtime-bundle.json", "full-manifest.json", "runtime-assembly-report.json", "SHA256SUMS.txt"))]
    if any(path.exists() or path.is_symlink() for path in reserved) or list(output_zip.parent.glob(output_zip.name + ".[0-9][0-9][0-9]")):
        raise FileExistsError("Never overwrite a previous assembly, report, or part")
    if output_zip == source_zip or output_zip.is_relative_to(overlay):
        raise ValueError("Output must not replace input or be inside overlay")
    source_identity = _identity(source_zip)
    manifest_identity = _identity(source_manifest)
    if manifest_identity[2] > 16 * 1024**2:
        raise ValueError("Source streamed manifest exceeds safety limit")
    manifest_bytes = source_manifest.read_bytes()
    if (_digest(source_zip) != source_sha256 or hashlib.sha256(manifest_bytes).hexdigest() != source_manifest_sha256
            or _identity(source_manifest) != manifest_identity):
        raise ValueError("Pinned source ZIP or streamed manifest SHA256 mismatch")
    manifest = json.loads(manifest_bytes)
    overlay_files, excluded_overlay = {}, []
    for path in release.regular_files(overlay):
        name = path.relative_to(overlay).as_posix()
        if _exclude(name):
            excluded_overlay.append(name)
            continue
        _public_path(name)
        overlay_files[name] = path
    required = {"FanyiJiVoiceNext.exe", "_updater/FanyiJiUpdate.exe", "build-info.json"}
    if not required <= set(overlay_files):
        raise ValueError("Fresh main EXE, helper, and build-info are required in overlay")
    build = json.loads(overlay_files["build-info.json"].read_text(encoding="utf-8"))
    if build.get("application") != release.APPLICATION or build.get("version") != version:
        raise ValueError("Overlay build-info product/version mismatch")
    ready = {"schema_version": 1, "application": release.APPLICATION, "version": version, "build_id": build_id}
    generated = {READY: _bytes_json(ready), "release-defaults.json": _bytes_json(release_defaults_payload())}
    excluded, rows, reused, modified = [], {}, [], []
    with zipfile.ZipFile(source_zip) as source, source_zip.open("rb") as stream:
        old = _validate_source(source, manifest)
        retained = {}
        for name, row in old.items():
            if _exclude(name) or name in GENERATED:
                excluded.append(name)
                continue
            _public_path(name)
            if name not in overlay_files:
                retained[name] = row
        _case_paths(set(retained) | set(overlay_files) | set(generated) | {release.UPDATE_MANIFEST})
        for name in release.PUBLIC_CERTIFICATE_FILES & (set(retained) | set(overlay_files)):
            data = overlay_files[name].read_bytes() if name in overlay_files else source.read(name)
            if len(data) > 2 * 1024**2 or b"PRIVATE KEY" in data.upper() or b"-----BEGIN CERTIFICATE-----" not in data:
                raise ValueError("Public certificate contains non-certificate or private data")
            if name != "_internal/certifi/cacert.pem":
                canonical = "_internal/certifi/cacert.pem"
                reference = overlay_files[canonical].read_bytes() if canonical in overlay_files else source.read(canonical)
                if data != reference:
                    raise ValueError("Vendored public certificate differs from canonical bundle")
        with zipfile.ZipFile(output_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as target:
            for name in sorted(retained):
                _copy_raw(source, stream, target, source.getinfo(name))
                rows[name] = {key: retained[name][key] for key in ("path", "size", "sha256")}
                reused.append(name)
            for name, path in sorted(overlay_files.items()):
                if name not in GENERATED:
                    rows[name] = _write_overlay(target, path, name)
                    modified.append(name)
            for name, data in sorted(generated.items()):
                target.writestr(name, data)
                rows[name] = {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            update = {"schema_version": 1, "application": release.APPLICATION, "version": version,
                      "files": [rows[name] for name in sorted(rows)]}
            update_bytes = _bytes_json(update)
            if len(update_bytes) > 2 * 1024**2 or sum(row["size"] for row in rows.values()) > 8 * GIB:
                raise ValueError("Runtime manifest or extraction exceeds updater limit")
            target.writestr(release.UPDATE_MANIFEST, update_bytes)
    if _identity(source_zip) != source_identity:
        raise ValueError("Source ZIP changed during raw assembly")
    if output_zip.stat().st_size > 8 * GIB:
        raise ValueError("Runtime archive exceeds downloader limit")
    if (output_zip.stat().st_size + split_size - 1) // split_size > 16:
        raise ValueError("Runtime archive would exceed the sixteen-part descriptor limit")
    full = update | {"files": update["files"] + [{"path": release.UPDATE_MANIFEST, "size": len(update_bytes),
                                                   "sha256": hashlib.sha256(update_bytes).hexdigest()}]}
    release.write_json(output_zip.parent / "full-manifest.json", full)
    parts, digest = _split(output_zip, split_size, version, repository)
    descriptor = {"schema_version": 1, "application": release.APPLICATION, "platform": "windows-x64",
                  "kind": "runtime-bundle", "version": version, "build_id": build_id,
                  "name": output_zip.name, "size": output_zip.stat().st_size, "sha256": digest, "parts": parts}
    release.write_json(output_zip.parent / "runtime-bundle.json", descriptor)
    report = {"version": version, "build_id": build_id, "archive": descriptor,
              "source_archive_sha256": source_sha256, "source_manifest_sha256": source_manifest_sha256,
              "reused_compressed_entries": len(reused), "reused_compressed_bytes": sum(old[name]["compressed_size"] for name in reused),
              "overlay_entries": modified, "generated_entries": sorted(generated) + [release.UPDATE_MANIFEST],
              "excluded_source_entries": excluded, "excluded_overlay_entries": excluded_overlay,
              "runtime_entries_recompressed": False, "downloaded_model_tree_read": False,
              "model_weights_bundled": False, "source_modified": False, "program_files": len(rows)}
    release.write_json(output_zip.parent / "runtime-assembly-report.json", report)
    checksums = [(digest, output_zip.name), *((part["sha256"], part["name"]) for part in parts)]
    for name in ("runtime-bundle.json", "full-manifest.json", "runtime-assembly-report.json"):
        checksums.append((_digest(output_zip.parent / name), name))
    (output_zip.parent / "SHA256SUMS.txt").write_text("".join(f"{sha}  {name}\n" for sha, name in checksums), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "source-manifest", "overlay", "output-zip"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("source-sha256", "source-manifest-sha256", "version", "build-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--repository", default="humerus-sketch/fanyiji-voice-next")
    parser.add_argument("--split-size", type=int, default=1536 * 1024**2)
    result = assemble_runtime_bundle(**vars(parser.parse_args(argv)))
    print(json.dumps({key: result[key] for key in ("version", "build_id", "reused_compressed_entries", "archive")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
