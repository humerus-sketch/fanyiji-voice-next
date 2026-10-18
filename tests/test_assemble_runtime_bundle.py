"""The runtime assembler retains compressed bytes and emits installable parts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import zipfile

import pytest


@pytest.fixture
def assembler():
    path = Path(__file__).resolve().parents[1] / "tools/assemble_runtime_bundle.py"
    spec = importlib.util.spec_from_file_location("runtime_bundle_assembler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle_inputs(tmp_path):
    source = tmp_path / "measured.zip"
    data = {
        "FanyiJiVoiceNext.exe": b"old executable",
        "_updater/FanyiJiUpdate.exe": b"old helper",
        "build-info.json": b'{"application":"fanyiji-voice-next","version":"0.5.2"}',
        "_internal/library.dll": b"retained runtime library " * 1000,
        "_internal/torch/_inductor/autoheuristic/artifacts/_MMRankingA100.py": b"# shipped torch code\n",
        "_internal/jieba/lac_small/model_baseline/word_emb": b"hidden learned weight",
        "_internal/torch/__pycache__/unused.pyc": b"generated import cache",
        "FanyiJiVoiceNext_models/asr/model.safetensors": b"never copy user weights",
        "_updater/runtime-bootstrap.json": b"obsolete bootstrap description",
        "使用说明.md": "原始公开说明".encode(),
    }
    with zipfile.ZipFile(source, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, content in data.items():
            archive.writestr(name, content)
    with zipfile.ZipFile(source) as archive:
        rows = [{"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                 "compressed_size": archive.getinfo(name).compress_size} for name, content in data.items()]
    manifest = tmp_path / "streamed-source.json"
    manifest.write_text(json.dumps({"files": rows}), encoding="utf-8")
    overlay = tmp_path / "fresh-program"
    (overlay / "_updater").mkdir(parents=True)
    (overlay / "FanyiJiVoiceNext.exe").write_bytes(b"new verified executable")
    (overlay / "_updater/FanyiJiUpdate.exe").write_bytes(b"new verified helper")
    (overlay / "build-info.json").write_text('{"application":"fanyiji-voice-next","version":"0.5.3"}', encoding="utf-8")
    (overlay / "release-defaults.json").write_text('{"private_path":"must never ship"}', encoding="utf-8")
    (overlay / "_updater/runtime-ready.json").write_text('{"build_id":"stale"}', encoding="utf-8")
    output = tmp_path / "assembled"
    output.mkdir()
    return {"source_zip": source, "source_manifest": manifest, "source_sha256": sha(source),
            "source_manifest_sha256": sha(manifest), "overlay": overlay,
            "output_zip": output / "FanyiJiVoiceNext-0.5.3-runtime.zip", "version": "0.5.3",
            "build_id": "voice-ab-0.5.3-test", "repository": "example/voice-next", "split_size": 512}


def compressed_bytes(path, name):
    with zipfile.ZipFile(path) as archive, path.open("rb") as stream:
        info = archive.getinfo(name)
        stream.seek(info.header_offset)
        header = struct.unpack("<4s5H3I2H", stream.read(30))
        stream.seek(header[9] + header[10], 1)
        return stream.read(info.compress_size)


def test_raw_assembly_preserves_runtime_deflate_and_builds_verified_install_manifest(assembler, bundle_inputs):
    args = bundle_inputs
    before_source = sha(args["source_zip"])
    before_overlay = {p.relative_to(args["overlay"]).as_posix(): sha(p)
                      for p in args["overlay"].rglob("*") if p.is_file()}
    result = assembler.assemble_runtime_bundle(**args)
    target = args["output_zip"]
    for name in ("_internal/library.dll", "_internal/torch/_inductor/autoheuristic/artifacts/_MMRankingA100.py", "使用说明.md"):
        assert compressed_bytes(args["source_zip"], name) == compressed_bytes(target, name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert not any("model_baseline" in name or "__pycache__" in name or "_models/" in name for name in names)
        assert "_updater/runtime-bootstrap.json" not in names
        assert archive.read("FanyiJiVoiceNext.exe") == b"new verified executable"
        assert archive.read("_updater/FanyiJiUpdate.exe") == b"new verified helper"
        ready = json.loads(archive.read("_updater/runtime-ready.json"))
        assert ready == {"schema_version": 1, "application": "fanyiji-voice-next", "version": "0.5.3", "build_id": args["build_id"]}
        assert b"private_path" not in archive.read("release-defaults.json")
        manifest = json.loads(archive.read("update-manifest.json"))
        assert {row["path"] for row in manifest["files"]} == names - {"update-manifest.json"}
        for row in manifest["files"]:
            content = archive.read(row["path"])
            assert row["size"] == len(content) and row["sha256"] == hashlib.sha256(content).hexdigest()
    descriptor = json.loads((target.parent / "runtime-bundle.json").read_text(encoding="utf-8"))
    assert descriptor["sha256"] == sha(target) and descriptor["size"] == target.stat().st_size
    assert descriptor["build_id"] == args["build_id"] and descriptor["kind"] == "runtime-bundle"
    parts = descriptor["parts"]
    assert len(parts) > 1 and all(row["size"] <= 512 for row in parts)
    assert b"".join((target.parent / row["name"]).read_bytes() for row in parts) == target.read_bytes()
    for row in parts:
        assert row["sha256"] == sha(target.parent / row["name"])
        assert row["url"] == f"https://github.com/example/voice-next/releases/download/v0.5.3/{row['name']}"
    assert result["runtime_entries_recompressed"] is False
    assert result["model_weights_bundled"] is False
    assert before_source == sha(args["source_zip"])
    assert before_overlay == {p.relative_to(args["overlay"]).as_posix(): sha(p)
                              for p in args["overlay"].rglob("*") if p.is_file()}


@pytest.mark.parametrize("field", ["source_sha256", "source_manifest_sha256"])
def test_pinned_hash_failure_prevents_any_archive_output(assembler, bundle_inputs, field):
    bundle_inputs[field] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        assembler.assemble_runtime_bundle(**bundle_inputs)
    assert not bundle_inputs["output_zip"].exists()


def test_existing_output_is_never_overwritten(assembler, bundle_inputs):
    target = bundle_inputs["output_zip"]
    target.write_bytes(b"keep previous archive")
    with pytest.raises(FileExistsError):
        assembler.assemble_runtime_bundle(**bundle_inputs)
    assert target.read_bytes() == b"keep previous archive"


@pytest.mark.parametrize("relative", ["_internal/private.key", "_internal/weights.gguf", "settings/user.json", "_internal/CON.dll"])
def test_private_or_unsafe_overlay_is_rejected_before_output(assembler, bundle_inputs, relative):
    target = bundle_inputs["overlay"] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name == "CON.dll":
        # Windows reserves this filename, so exercise its lexical boundary.
        with pytest.raises(ValueError):
            assembler._public_path(relative)
        return
    target.write_bytes(b"not public")
    with pytest.raises(ValueError):
        assembler.assemble_runtime_bundle(**bundle_inputs)
    assert not bundle_inputs["output_zip"].exists()


def test_manifest_size_mismatch_is_rejected_before_output(assembler, bundle_inputs):
    path = bundle_inputs["source_manifest"]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["files"][0]["size"] += 1
    path.write_text(json.dumps(value), encoding="utf-8")
    bundle_inputs["source_manifest_sha256"] = sha(path)
    with pytest.raises(ValueError, match="mismatched"):
        assembler.assemble_runtime_bundle(**bundle_inputs)
    assert not bundle_inputs["output_zip"].exists()


@pytest.mark.parametrize("size", [0, -1, 2 * 1024**3, True])
def test_parts_must_fit_the_github_asset_limit(assembler, bundle_inputs, size):
    bundle_inputs["split_size"] = size
    with pytest.raises(ValueError):
        assembler.assemble_runtime_bundle(**bundle_inputs)
    assert not bundle_inputs["output_zip"].exists()


def test_raw_copy_and_central_directory_remain_valid_across_zip64_boundaries(assembler, bundle_inputs, monkeypatch):
    # Exercise ZIP64 local sizes and central offsets without allocating GB data.
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 128)
    assembler.assemble_runtime_bundle(**bundle_inputs)
    target = bundle_inputs["output_zip"]
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert archive.read("_internal/library.dll") == b"retained runtime library " * 1000
    assert b"PK\x06\x06" in target.read_bytes()
    assert compressed_bytes(target, "_internal/library.dll") == compressed_bytes(bundle_inputs["source_zip"], "_internal/library.dll")


def test_local_and_central_name_mismatch_cannot_be_raw_copied(assembler, bundle_inputs):
    source = bundle_inputs["source_zip"]
    with zipfile.ZipFile(source) as archive:
        offset = archive.getinfo("_internal/library.dll").header_offset
    data = bytearray(source.read_bytes())
    data[offset + 30] = ord("X")
    source.write_bytes(data)
    bundle_inputs["source_sha256"] = sha(source)
    with pytest.raises(ValueError, match="local and central"):
        assembler.assemble_runtime_bundle(**bundle_inputs)
    assert not (bundle_inputs["output_zip"].parent / "runtime-bundle.json").exists()


def test_overlay_case_collision_with_retained_parent_is_rejected_before_output(assembler, bundle_inputs):
    target = bundle_inputs["overlay"] / "_internal/TORCH/another.dll"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"case collision")
    with pytest.raises(ValueError, match="Case-colliding"):
        assembler.assemble_runtime_bundle(**bundle_inputs)
    assert not bundle_inputs["output_zip"].exists()


def test_part_count_is_bounded_before_any_parts_or_descriptor_are_emitted(assembler, bundle_inputs):
    bundle_inputs["split_size"] = 1
    with pytest.raises(ValueError, match="sixteen-part"):
        assembler.assemble_runtime_bundle(**bundle_inputs)
    folder = bundle_inputs["output_zip"].parent
    assert not list(folder.glob("*.zip.[0-9][0-9][0-9]"))
    assert not (folder / "runtime-bundle.json").exists()
