"""Pinned metadata and untrusted path validation; never download real models."""
from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from fanyiji_next.model_catalog import (
    FileSpec, ModelSpec, MODEL_CATALOG, MODEL_COMBINATIONS, QWEN_ASR, FUN_ASR,
    TRANSLATE_GEMMA, SILERO_VAD, validate_download_url,
)


REV = "a" * 40


def file_spec(path="model.bin", data=b"abc", **changes):
    args = dict(relative_path=path, url=f"https://huggingface.co/test/repo/resolve/{REV}/{path}",
                size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    args.update(changes)
    return FileSpec(**args)


def model_spec(files=None, **changes):
    args = dict(model_id="test-bf16", directory_name="test-bf16", display_name="Test",
                files=files or (file_spec(),), license_name="Apache-2.0", source_url="https://huggingface.co/test/repo")
    args.update(changes)
    return ModelSpec(**args)


def test_catalog_exact_files_sizes_and_shared_assets():
    assert len(MODEL_CATALOG) == 4
    assert [len(MODEL_CATALOG[k].files) for k in (QWEN_ASR, FUN_ASR, TRANSLATE_GEMMA, SILERO_VAD)] == [7, 10, 1, 1]
    assert [MODEL_CATALOG[k].total_bytes for k in (QWEN_ASR, FUN_ASR, TRANSLATE_GEMMA, SILERO_VAD)] == [
        1576363223, 1987942401, 2489909760, 643854]
    assert MODEL_COMBINATIONS["A"] == (QWEN_ASR, TRANSLATE_GEMMA, SILERO_VAD)
    assert MODEL_COMBINATIONS["B"] == (FUN_ASR, TRANSLATE_GEMMA, SILERO_VAD)
    assert MODEL_CATALOG[TRANSLATE_GEMMA].requires_license_ack
    assert MODEL_CATALOG[TRANSLATE_GEMMA].license_url == "https://ai.google.dev/gemma/terms"
    assert MODEL_CATALOG[SILERO_VAD].files[0].sha256 == "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"


def test_catalog_fixed_revisions_and_weight_hashes():
    assert all("/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c/" in f.url for f in MODEL_CATALOG[QWEN_ASR].files)
    assert all("/272c57b82523ada6fd87095e955f8e29100979ab/" in f.url for f in MODEL_CATALOG[FUN_ASR].files)
    gemma = MODEL_CATALOG[TRANSLATE_GEMMA].files[0]
    assert "/d37868f759d2f817a43d28dd80cf9f982ffa62fd/" in gemma.url
    assert gemma.sha256 == "81200d03e843d2ec1ece6eeafe7d13cb6e5211e1fcd336ade55790b683a08330"


@pytest.mark.parametrize("path", ["", "../x", "a/../x", "/a", "C:/x", "a\\b", "a:b", "a//b", "a/./b",
                                        "a/", "a.", "a ", "CON", "NUL.bin", "com1/x", ".hidden", "a/\nfoo",
                                        "a?b", 'a"b', "a*b", "a|b", "a<b", "a>b", "a\x7fb"])
def test_reject_unsafe_paths(path):
    with pytest.raises(ValueError):
        file_spec(path)


@pytest.mark.parametrize("url", ["http://huggingface.co/test/repo/resolve/" + REV + "/model.bin",
                                      "https://evil.example/test/repo/resolve/" + REV + "/model.bin",
                                      "https://user:password@huggingface.co/test/repo/resolve/" + REV + "/model.bin",
                                      "https://huggingface.co/test/repo/resolve/main/model.bin",
                                      "https://huggingface.co/test/repo/resolve/" + REV + "/other.bin",
                                      "https://huggingface.co/test/repo/resolve/" + REV + "/model.bin?q=secret",
                                      "https://huggingface.co/test/repo/resolve/" + REV + "/model.bin#x",
                                      "https://github.com/other/repo/releases/download/tag/model.bin"])
def test_reject_unpinned_or_untrusted_url(url):
    with pytest.raises(ValueError):
        file_spec(url=url)


@pytest.mark.parametrize("size", [True, False, 1.2, 0, -1, "3"])
def test_reject_invalid_size(size):
    with pytest.raises(ValueError):
        file_spec(size_bytes=size)


@pytest.mark.parametrize("digest", ["x" * 64, "a" * 63, "A" * 64, "", None])
def test_reject_invalid_hash(digest):
    with pytest.raises(ValueError):
        file_spec(sha256=digest)


@pytest.mark.parametrize("names", [("a", "A"), ("a", "a/b"), ("a", "a.part"), ("a", "a.part/b")])
def test_file_sidecar_and_directory_collisions(names):
    with pytest.raises(ValueError):
        model_spec(tuple(file_spec(p) for p in names))


def test_nested_real_model_paths_supported_and_fingerprint_stable():
    spec = model_spec((file_spec("Qwen3-0.6B/config.json"), file_spec("model.pt")))
    assert spec.fingerprint == replace(spec).fingerprint
    assert spec.fingerprint != replace(spec, files=(file_spec("model.pt", b"abcd"),)).fingerprint
    with pytest.raises(FrozenInstanceError):
        spec.model_id = "other"
    with pytest.raises(TypeError):
        MODEL_CATALOG["other"] = spec


@pytest.mark.parametrize("changes", [{"model_id": None}, {"directory_name": "bad"}, {"requires_license_ack": 1},
                                          {"files": [file_spec()]}, {"directory_name": "../bad-bf16"}])
def test_invalid_model(changes):
    with pytest.raises(ValueError):
        model_spec(**changes)


def test_redirect_whitelist_only_allows_https_known_cdn():
    validate_download_url("https://cas-bridge.xethub.hf.co/x?signature=opaque", redirect=True)
    validate_download_url("https://release-assets.githubusercontent.com/x?signature=opaque", redirect=True)
    for url in ("http://cas-bridge.xethub.hf.co/x", "https://cdn.evil.example/x", "https://user@huggingface.co/x"):
        with pytest.raises(ValueError):
            validate_download_url(url, redirect=True)


@pytest.mark.parametrize("host", [
    "cdn-lfs.hf.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co",
    "cas-server.xethub-eu.hf.co", "transfer.xethub.hf.co", "transfer.xethub-eu.hf.co",
    "aws.cdn.hf.co", "us.aws.cdn.hf.co", "us-east-1.aws.cdn.hf.co", "us-west-2.aws.cdn.hf.co",
    "eu-west-3.aws.cdn.hf.co", "ap-southeast-1.aws.cdn.hf.co", "us.gcp.cdn.hf.co",
    "us-east1.us.gcp.cdn.hf.co", "us-central1.us.gcp.cdn.hf.co", "us-west4.us.gcp.cdn.hf.co",
    "europe-west4.us.gcp.cdn.hf.co", "asia-southeast1.us.gcp.cdn.hf.co",
])
def test_documented_storage_hosts_are_exact_and_redirect_only(host):
    validate_download_url(f"https://{host}/xet-bridge/file?X-Amz-Signature=FAKE", redirect=True)
    validate_download_url(f"https://{host}:443/file", redirect=True)
    for url in (f"https://{host}/file", f"https://{host}/file?X-Amz-Signature=FAKE"):
        with pytest.raises(ValueError):
            validate_download_url(url)
    for bad_host in ("evil." + host, host + ".evil.example", host + "."):
        with pytest.raises(ValueError):
            validate_download_url(f"https://{bad_host}/file", redirect=True)


@pytest.mark.parametrize("url", [
    "http://us.aws.cdn.hf.co/file", "https://user:SECRET@us.aws.cdn.hf.co/file",
    "https://@us.aws.cdn.hf.co/file", "https://:@us.aws.cdn.hf.co/file",
    "https://us.aws.cdn.hf.co:444/file", "https://us.aws.cdn.hf.co:abc/file",
    "https://us.aws.cdn.hf.co:/file", "https://us.aws.cdn.hf.co:65536/file",
    "https://us.aws.cdn.hf.co/file\\other", "https://us.aws.cdn.hf.co/file#fragment",
    " https://us.aws.cdn.hf.co/file", "\x00https://us.aws.cdn.hf.co/file",
    "https://us.aws.cdn.hf.co/\nfile", "https://us.aws.cdn.hf.co/\rfile",
    "https://us.aws.cdn.hf.co/\tfile", "https://us.aws.cdn.hf.co/\x7ffile",
    None, 123,
])
def test_cdn_redirect_rejects_malformed_authority_and_raw_controls(url):
    with pytest.raises(ValueError) as caught:
        validate_download_url(url, redirect=True)
    assert "SECRET" not in str(caught.value)


def test_optional_sources_are_pinned_and_do_not_change_content_identity():
    original = file_spec()
    mirror = original.url.replace("huggingface.co", "hf-mirror.com")
    routed = replace(original, sources=(mirror,))
    assert routed.url == original.url and routed.download_urls == (mirror, original.url)
    before, after = model_spec((original,)), model_spec((routed,))
    assert before.fingerprint == after.fingerprint
    assert before.legacy_fingerprint == after.legacy_fingerprint
    assert before.fingerprint != replace(before, files=(file_spec(data=b"changed"),)).fingerprint


@pytest.mark.parametrize("source", [
    "https://hf-mirror.com/test/repo/resolve/main/model.bin",
    "https://hf-mirror.com/other/repo/resolve/" + REV + "/model.bin",
    "https://modelscope.cn/api/v1/models/test/repo/repo?Revision=master&FilePath=model.bin",
    "https://modelscope.cn/api/v1/models/test/repo/repo?Revision=" + REV + "&FilePath=other.bin",
    "https://modelscope.cn/api/v1/models/test/repo/repo?Revision=" + REV + "&FilePath=model.bin&token=FAKE",
    "https://modelscope.cn/api/v1/models/test/repo/repo?Revision=" + REV + "&Revision=" + REV + "&FilePath=model.bin",
    "https://evil.modelscope.cn/api/v1/models/test/repo/repo?Revision=" + REV + "&FilePath=model.bin",
])
def test_optional_sources_cannot_float_revision_or_change_file(source):
    with pytest.raises(ValueError):
        file_spec(sources=(source,))


def test_pinned_official_modelscope_source_supported():
    source = "https://modelscope.cn/api/v1/models/test/repo/repo?Revision=" + REV + "&FilePath=Qwen3-0.6B%2Fconfig.json"
    assert file_spec("Qwen3-0.6B/config.json", sources=(source,)).download_urls[0] == source


def test_real_domestic_sources_keep_original_content_and_file_revisions():
    for file in MODEL_CATALOG[QWEN_ASR].files:
        source = file.download_urls[0]
        assert source.startswith("https://modelscope.cn/api/v1/models/Qwen/Qwen3-ASR-0.6B-hf/repo?")
        revision = ("005b749cd1566dc579fe0238316087e2fc1945c0" if file.relative_path == "chat_template.jinja"
                    else "c4650525d0d40f32f1517abe624a89be04386bef")
        assert "Revision=" + revision in source
        assert file.download_urls[1].startswith("https://hf-mirror.com/")
        assert file.download_urls[-1] == file.url
    gemma = MODEL_CATALOG[TRANSLATE_GEMMA].files[0]
    assert gemma.download_urls[0].startswith(
        "https://modelscope.cn/api/v1/models/prithivMLmods/translategemma-4b-it-f32-GGUF/repo?")
    assert "Revision=331bd735819210f93a62833b306dd8fa4b70b074" in gemma.download_urls[0]
    assert gemma.sha256 == "81200d03e843d2ec1ece6eeafe7d13cb6e5211e1fcd336ade55790b683a08330"


def test_modelscope_storage_host_is_exact_and_redirect_only():
    host = "cdn-lfs-cn-1.modelscope.cn"
    validate_download_url(f"https://{host}/asset?Signature=FAKE", redirect=True)
    for url in (f"https://{host}/asset", f"http://{host}/asset", f"https://{host}:444/asset",
                f"https://evil.{host}/asset", f"https://{host}.evil.example/asset",
                f"https://@{host}/asset", f"https://{host}/asset#fragment"):
        with pytest.raises(ValueError):
            validate_download_url(url, redirect=not url == f"https://{host}/asset")


def test_fun_only_uses_modelscope_for_files_with_identical_bytes():
    for file in MODEL_CATALOG[FUN_ASR].files:
        source = file.download_urls[0]
        if file.relative_path == "model.pt":
            assert "hf-mirror.com" in source
            assert not any("modelscope.cn" in url for url in file.download_urls)
            assert file.size_bytes == 1971149431
            assert file.sha256 == "55ae0d2fee369f0f11cce0795f6927934ad17cf11b278a7e56a51272074160bb"
        else:
            assert source.startswith("https://modelscope.cn/api/v1/models/FunAudioLLM/Fun-ASR-Nano-2512/repo?")
            revision = {"config.yaml": "16795894cff13ca4f89b301dd08c22f60351e844",
                        "configuration.json": "39f6a4d23e678bee2a471da2f74224a314b2fafc",
                        "multilingual.tiktoken": "9e733102544707cd890655071ee3f66269355165"}.get(
                            file.relative_path, "c819b4dcbf5f76bc86f9cf488213606ca1832715")
            assert "Revision=" + revision in source


def test_silero_domestic_mirrors_keep_the_original_vad_and_github_fallback():
    file = MODEL_CATALOG[SILERO_VAD].files[0]
    assert file.download_urls[-1] == file.url
    assert file.url == "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    assert file.size_bytes == 643854
    assert file.sha256 == "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
    assert len(file.sources) == 2
    assert "/xnnehang/k2-fsa-silero-vad/repo?Revision=e64bab140550f46d724405a77ce842d2095feeef" in file.sources[0]
    assert "/liaowenbin/silero_vad/repo?Revision=a9cffd6df29686d5e36fbe9d3baf6ef62d9c16a9" in file.sources[1]
    assert all(source.endswith("&FilePath=silero_vad.onnx") for source in file.sources)


@pytest.mark.parametrize("sources", [["https://hf-mirror.com/x"], ([],), (None,), ("same", "same")])
def test_sources_must_be_a_small_unique_string_tuple(sources):
    with pytest.raises(ValueError):
        file_spec(sources=sources)
