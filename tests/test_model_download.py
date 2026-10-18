"""Hermetic streaming/resume tests. Fake transport never downloads model weights."""
from dataclasses import replace
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from fanyiji_next.model_catalog import FileSpec, ModelSpec
from fanyiji_next.model_download import (
    ModelDownloadManager, ModelDownloadError, DownloadCancelled, ModelDownloadBusy, ModelLicenseRequired,
    _TrustedRedirect,
    _HttpTransport,
)


DATA = b"0123456789abcdef"


def spec_for(files=None, ack=False):
    entries = files or {"sub/model.bin": DATA}
    files = tuple(FileSpec(p, "https://huggingface.co/test/repo/resolve/" + "a" * 40 + "/" + p,
                           len(d), hashlib.sha256(d).hexdigest()) for p, d in entries.items())
    return ModelSpec("test-bf16", "test-bf16", "Test", files, "Test", "https://huggingface.co/test/repo", ack)


class Response:
    def __init__(self, data=DATA, status=200, headers=None, fail_after=None):
        self.stream = io.BytesIO(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))} if headers is None else headers
        self.fail_after = fail_after
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self, size):
        if self.fail_after is not None and self.stream.tell() >= self.fail_after:
            raise OSError("secret-token: interrupted")
        return self.stream.read(size)


class Transport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def open(self, url, *, headers, timeout):
        self.calls.append((url, headers, timeout))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def manager(tmp_path, transport=None, spec=None, **kwargs):
    spec = spec or spec_for()
    kwargs.setdefault("attempts_per_source", 1)
    kwargs.setdefault("retry_backoff", 0)
    return ModelDownloadManager(tmp_path / "RenamedApp_models", catalog={spec.model_id: spec},
                                transport=transport or Transport(), chunk_size=4, **kwargs)


def paths(mgr, spec=None):
    spec = spec or spec_for()
    final = mgr.model_root / spec.directory_name / spec.files[0].relative_path
    return final, final.with_name(final.name + ".part"), mgr.model_root / spec.directory_name / ".model-ready.json"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_inspect_missing_is_read_only_and_does_not_create_root(tmp_path):
    mgr = manager(tmp_path)
    state = mgr.inspect("test-bf16")
    assert not state.ready and state.verified_bytes == state.partial_bytes == 0
    assert state.remaining_bytes == len(DATA)
    assert not mgr.model_root.exists()


def test_download_verifies_marker_progress_and_skips_completed(tmp_path):
    response = Response()
    transport = Transport(response)
    mgr = manager(tmp_path, transport)
    updates = []
    result = mgr.download("test-bf16", progress=updates.append)
    final, partial, marker = paths(mgr)
    assert result == mgr.model_root / "test-bf16"
    assert final.read_bytes() == DATA and not partial.exists()
    assert json.loads(marker.read_text())["manifest_sha256"] == spec_for().fingerprint
    assert mgr.inspect("test-bf16").ready and response.closed
    assert [u.downloaded_bytes for u in updates] == sorted(u.downloaded_bytes for u in updates)
    assert updates[-1].status == "complete" and updates[-1].downloaded_bytes == len(DATA)
    mgr.download("test-bf16")
    assert len(transport.calls) == 1


def test_valid_existing_files_without_marker_only_writes_marker(tmp_path):
    mgr = manager(tmp_path)
    final, _, marker = paths(mgr)
    write(final, DATA)
    state = mgr.inspect("test-bf16")
    assert not state.ready and state.verified_bytes == len(DATA) and state.remaining_bytes == 0
    mgr.download("test-bf16")
    assert marker.exists() and mgr.inspect("test-bf16").ready
    assert mgr.transport.calls == []


def test_tampered_same_size_file_repaired_and_wrong_marker_rebuilt(tmp_path):
    mgr = manager(tmp_path, Transport(Response(), Response()))
    final, _, marker = paths(mgr)
    mgr.download("test-bf16")
    final.write_bytes(b"x" * len(DATA))
    assert not mgr.inspect("test-bf16").ready
    mgr.download("test-bf16")
    assert final.read_bytes() == DATA
    marker.write_text("{}")
    assert not mgr.inspect("test-bf16").ready
    mgr.download("test-bf16")
    assert mgr.inspect("test-bf16").ready and len(mgr.transport.calls) == 2


def test_resume_strict_206_and_partial_estimate(tmp_path):
    transport = Transport(Response(DATA[5:], 206, {"Content-Length": "11", "Content-Range": "bytes 5-15/16"}))
    mgr = manager(tmp_path, transport)
    final, partial, _ = paths(mgr)
    write(partial, DATA[:5])
    state = mgr.inspect("test-bf16")
    assert not state.ready and state.partial_bytes == 5 and state.remaining_bytes == 11
    mgr.download("test-bf16")
    assert transport.calls[0][1]["Range"] == "bytes=5-"
    assert final.read_bytes() == DATA and mgr.inspect("test-bf16").ready


def test_range_ignored_200_restarts_safely_without_append(tmp_path):
    mgr = manager(tmp_path, Transport(Response()))
    final, partial, _ = paths(mgr)
    write(partial, b"wrong")
    updates = []
    mgr.download("test-bf16", progress=updates.append)
    assert mgr.transport.calls[0][1]["Range"] == "bytes=5-"
    assert final.read_bytes() == DATA
    assert next(u for u in updates if u.status == "downloading").downloaded_bytes == 0


@pytest.mark.parametrize("response", [
    Response(DATA[5:], 206, {"Content-Range": "bytes 0-10/16"}),
    Response(DATA[5:], 206, {"Content-Range": "bytes 5-15/17"}),
    Response(DATA[5:], 206, {"Content-Range": "bytes 5-14/16"}),
    Response(DATA[5:], 206, {"Content-Range": "bytes 5-15/*"}),
    Response(DATA, 200, {"Content-Length": "17"}),
    Response(DATA, 200, {"Content-Encoding": "gzip"}),
    Response(DATA, 416),
])
def test_bad_range_length_encoding_retains_original_partial(tmp_path, response):
    mgr = manager(tmp_path, Transport(response))
    _, partial, marker = paths(mgr)
    write(partial, DATA[:5])
    with pytest.raises(ModelDownloadError):
        mgr.download("test-bf16")
    assert partial.read_bytes() == DATA[:5] and not marker.exists()
    assert not mgr.inspect("test-bf16").ready


def test_network_interruption_retains_partial_then_resumes_without_token_leak(tmp_path):
    mgr = manager(tmp_path, Transport(Response(fail_after=8),
                  Response(DATA[8:], 206, {"Content-Range": "bytes 8-15/16", "Content-Length": "8"})))
    final, partial, marker = paths(mgr)
    with pytest.raises(ModelDownloadError) as error:
        mgr.download("test-bf16")
    assert "secret-token" not in str(error.value)
    assert partial.read_bytes() == DATA[:8] and not marker.exists()
    mgr.download("test-bf16")
    assert mgr.transport.calls[-1][1]["Range"] == "bytes=8-"
    assert final.read_bytes() == DATA


def test_cancel_before_start_and_mid_download_retains_partial(tmp_path):
    event = threading.Event()
    event.set()
    mgr = manager(tmp_path, Transport(Response()))
    with pytest.raises(DownloadCancelled):
        mgr.download("test-bf16", cancel_event=event)
    assert not mgr.model_root.exists() and not mgr.transport.calls
    event.clear()

    def progress(update):
        if update.status == "downloading" and update.downloaded_bytes >= 4:
            event.set()

    with pytest.raises(DownloadCancelled):
        mgr.download("test-bf16", cancel_event=event, progress=progress)
    _, part, marker = paths(mgr)
    assert part.read_bytes() == DATA[:4] and not marker.exists()


@pytest.mark.parametrize("body,headers", [(b"x" * 16, None), (DATA[:8], {"Content-Length": "16"}),
                                          (DATA + b"extra", {"Content-Length": "16"})])
def test_bad_body_never_ready_and_full_bad_part_can_be_retried(tmp_path, body, headers):
    mgr = manager(tmp_path, Transport(Response(body, headers=headers), Response()))
    with pytest.raises(ModelDownloadError):
        mgr.download("test-bf16")
    assert not mgr.inspect("test-bf16").ready and not paths(mgr)[2].exists()
    mgr.download("test-bf16")
    assert mgr.inspect("test-bf16").ready


def test_full_valid_partial_promoted_without_network(tmp_path):
    mgr = manager(tmp_path)
    final, part, _ = paths(mgr)
    write(part, DATA)
    mgr.download("test-bf16")
    assert final.read_bytes() == DATA and mgr.transport.calls == []


def test_license_required_before_creating_files_or_network(tmp_path):
    mgr = manager(tmp_path, Transport(Response()), spec_for(ack=True))
    for ack in (False, 1, "true"):
        with pytest.raises(ModelLicenseRequired):
            mgr.download("test-bf16", license_acknowledged=ack)
    assert not mgr.model_root.exists() and not mgr.transport.calls
    mgr.download("test-bf16", license_acknowledged=True)
    assert mgr.inspect("test-bf16").ready


def test_same_model_lock_and_release_after_exception(tmp_path):
    mgr = manager(tmp_path, Transport(Response()))
    other = manager(tmp_path, Transport(Response()))
    with mgr._lock(spec_for()):
        with pytest.raises(ModelDownloadBusy):
            other.download("test-bf16")
    other.download("test-bf16")
    assert other.inspect("test-bf16").ready


def test_download_lock_blocks_another_process(tmp_path):
    mgr = manager(tmp_path)
    script = """
import os, sys
stream = open(sys.argv[1], 'r+b')
try:
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print('locked')
else:
    print('acquired')
finally:
    stream.close()
"""
    with mgr._lock(spec_for()):
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script,
                                 str(mgr.model_root / "test-bf16" / ".download.lock")],
                                capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == "locked", result.stderr


def test_final_verification_catches_callback_mutation(tmp_path):
    mgr = manager(tmp_path, Transport(Response()))
    final, _, marker = paths(mgr)

    def progress(update):
        if update.status == "verified":
            final.write_bytes(b"x" * len(DATA))

    with pytest.raises(ModelDownloadError, match="变化"):
        mgr.download("test-bf16", progress=progress)
    assert not marker.exists() and not mgr.inspect("test-bf16").ready


@pytest.mark.parametrize("location", ["root", "ancestor", "model", "nested", "partial", "marker"])
def test_symlink_paths_cannot_escape_root(tmp_path, location):
    mgr = manager(tmp_path, Transport(Response()))
    outside = tmp_path / "outside"
    outside.mkdir()
    final, part, marker = paths(mgr)
    links = {"root": mgr.model_root, "ancestor": tmp_path / "linked-parent", "model": final.parent.parent,
             "nested": final.parent, "partial": part, "marker": marker}
    link = links[location]
    link.parent.mkdir(parents=True, exist_ok=True)
    destination = outside if location not in {"partial", "marker"} else outside / "untouched"
    if location in {"partial", "marker"}:
        destination.write_bytes(b"keep")
    try:
        link.symlink_to(destination, target_is_directory=location not in {"partial", "marker"})
    except OSError:
        pytest.skip("Windows account cannot create symbolic links")
    if location == "ancestor":
        mgr = ModelDownloadManager(link / "child_models", catalog={"test-bf16": spec_for()}, transport=Transport(Response()))
    with pytest.raises(ModelDownloadError, match="链接|重解析"):
        mgr.download("test-bf16")
    assert sorted(p.name for p in outside.iterdir()) == (["untouched"] if location in {"partial", "marker"} else [])
    if destination.is_file():
        assert destination.read_bytes() == b"keep"


def test_partial_hardlink_cannot_modify_other_file(tmp_path):
    mgr = manager(tmp_path, Transport(Response()))
    _, part, _ = paths(mgr)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"keep")
    part.parent.mkdir(parents=True)
    os.link(outside, part)
    with pytest.raises(ModelDownloadError, match="硬链接"):
        mgr.download("test-bf16")
    assert outside.read_bytes() == b"keep" and not mgr.transport.calls


def test_marker_write_failure_does_not_claim_ready_or_leave_temp(tmp_path, monkeypatch):
    mgr = manager(tmp_path, Transport(Response()))
    original_replace = os.replace

    def fail_marker(src, dst):
        if str(dst).endswith(".model-ready.json"):
            raise OSError("cannot write marker")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_marker)
    with pytest.raises(ModelDownloadError):
        mgr.download("test-bf16")
    assert not mgr.inspect("test-bf16").ready
    assert list((mgr.model_root / "test-bf16").glob("*.tmp")) == []


@pytest.mark.parametrize("code", [401, 403, 404, 500])
def test_http_errors_hide_url_tokens_and_body(tmp_path, code):
    error = HTTPError("https://evil/?token=SECRET", code, "SECRET", {}, None)
    mgr = manager(tmp_path, Transport(error))
    with pytest.raises(ModelDownloadError) as caught:
        mgr.download("test-bf16")
    assert "SECRET" not in str(caught.value) and "https://" not in str(caught.value)


def test_redirect_rejects_untrusted_host_without_exposing_signed_url():
    with pytest.raises(ModelDownloadError) as caught:
        _TrustedRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example/?token=SECRET")
    assert "SECRET" not in str(caught.value)


def test_real_http_transport_resumes_through_current_hf_cdn_and_checks_full_sha(tmp_path):
    transport = _HttpTransport()
    calls = []
    class RedirectingOpener:
        def open(self, request, timeout):
            calls.append(request)
            redirected = _TrustedRedirect().redirect_request(request, None, 302, "Found", {},
                "https://us.aws.cdn.hf.co/xet-bridge/file?X-Amz-Signature=FAKE")
            assert redirected.get_header("Range") == "bytes=5-"
            assert redirected.get_header("Accept-encoding") == "identity"
            assert timeout == 10
            return Response(DATA[5:], 206, {"Content-Length": "11", "Content-Range": "bytes 5-15/16"})
    transport._opener = RedirectingOpener()
    mgr = manager(tmp_path, transport)
    final, part, _ = paths(mgr)
    write(part, DATA[:5])
    mgr.download("test-bf16")
    assert len(calls) == 1 and final.read_bytes() == DATA
    assert mgr.inspect("test-bf16").ready


@pytest.mark.parametrize("target", [
    "http://us.aws.cdn.hf.co/file?token=SECRET", "https://evil.us.aws.cdn.hf.co/file?token=SECRET",
    "https://us.aws.cdn.hf.co.evil.example/file?token=SECRET", "https://@us.aws.cdn.hf.co/file?token=SECRET",
    "https://user:SECRET@us.aws.cdn.hf.co/file", "https://us.aws.cdn.hf.co:444/file?token=SECRET",
])
def test_redirect_guard_still_blocks_untrusted_variants_without_token_leak(target):
    request = Request(spec_for().files[0].url)
    with pytest.raises(ModelDownloadError) as caught:
        _TrustedRedirect().redirect_request(request, None, 302, "Found", {}, target)
    assert "SECRET" not in str(caught.value) and "https://" not in str(caught.value)


def test_http_transport_rejects_direct_cdn_or_malformed_source_before_network():
    transport = _HttpTransport()
    class MustNotOpen:
        def open(self, *args, **kwargs):
            raise AssertionError("Network was reached for an invalid model source")
    transport._opener = MustNotOpen()
    for url in ("https://us.aws.cdn.hf.co/file?token=SECRET", "http://huggingface.co/test",
                "https://huggingface.co/test\\other", "https://huggingface.co:/test"):
        with pytest.raises(ValueError):
            transport.open(url, headers={}, timeout=10)


def test_unknown_model_and_malformed_catalog_rejected(tmp_path):
    mgr = manager(tmp_path)
    with pytest.raises(ModelDownloadError):
        mgr.inspect("../../other")
    with pytest.raises(ModelDownloadError):
        ModelDownloadManager(tmp_path, catalog={"test": object()})
    spec = spec_for()
    with pytest.raises(ModelDownloadError):
        ModelDownloadManager(tmp_path, catalog={spec.model_id: spec, "other": replace(spec, model_id="other")})


def mirrored_spec():
    spec = spec_for()
    file = spec.files[0]
    return replace(spec, files=(replace(file, sources=(file.url.replace("huggingface.co", "hf-mirror.com"),)),))


def test_interruption_is_retried_and_resumed_on_the_same_source(tmp_path):
    transport = Transport(Response(fail_after=8), Response(DATA[8:], 206,
        {"Content-Range": "bytes 8-15/16", "Content-Length": "8"}))
    mgr = manager(tmp_path, transport, mirrored_spec(), attempts_per_source=2)
    updates = []
    mgr.download("test-bf16", progress=updates.append)
    assert len(transport.calls) == 2 and all("hf-mirror.com" in row[0] for row in transport.calls)
    assert transport.calls[1][1]["Range"] == "bytes=8-"
    assert any(row.status == "retrying" and row.attempt == 1 for row in updates)
    assert mgr.inspect("test-bf16").ready


def test_network_failures_exhaust_source_then_switch_without_secret_leak(tmp_path):
    transport = Transport(OSError("URL?token=SECRET"), TimeoutError("SECRET"), Response())
    mgr = manager(tmp_path, transport, mirrored_spec(), attempts_per_source=2)
    updates = []
    mgr.download("test-bf16", progress=updates.append)
    assert ["hf-mirror.com" in call[0] for call in transport.calls] == [True, True, False]
    assert any(row.status == "switching_source" and row.source_name == "Hugging Face" for row in updates)
    assert all("SECRET" not in row.source_name for row in updates)
    assert mgr.inspect("test-bf16").ready


def test_source_switch_resumes_a_matching_partial(tmp_path):
    transport = Transport(Response(fail_after=8), Response(DATA[8:], 206,
        {"Content-Range": "bytes 8-15/16", "Content-Length": "8"}))
    mgr = manager(tmp_path, transport, mirrored_spec())
    mgr.download("test-bf16")
    assert "hf-mirror.com" in transport.calls[0][0] and "huggingface.co" in transport.calls[1][0]
    assert transport.calls[1][1]["Range"] == "bytes=8-"
    assert paths(mgr)[0].read_bytes() == DATA


def test_poisoned_prefix_never_passes_and_good_next_source_restarts_clean(tmp_path):
    transport = Transport(Response(b"xxxxxxxx" + DATA[8:], fail_after=8), OSError("network"),
        Response(DATA[8:], 206, {"Content-Range": "bytes 8-15/16", "Content-Length": "8"}), Response())
    mgr = manager(tmp_path, transport, mirrored_spec(), attempts_per_source=2)
    mgr.download("test-bf16")
    final, part, _ = paths(mgr)
    assert len(transport.calls) == 4
    assert transport.calls[2][1]["Range"] == "bytes=8-"
    assert "Range" not in transport.calls[3][1]
    bad = list(part.parent.glob(part.name + ".bad-*"))
    assert len(bad) == 1 and bad[0].read_bytes() == b"xxxxxxxx" + DATA[8:]
    assert final.read_bytes() == DATA and mgr.inspect("test-bf16").ready


def test_full_sha_failure_quarantines_bad_partial_and_uses_next_source(tmp_path):
    mgr = manager(tmp_path, Transport(Response(b"x" * 16), Response()), mirrored_spec(), attempts_per_source=2)
    mgr.download("test-bf16")
    assert len(mgr.transport.calls) == 2
    _, part, _ = paths(mgr)
    assert next(part.parent.glob(part.name + ".bad-*")).read_bytes() == b"x" * 16
    assert mgr.inspect("test-bf16").ready


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_http_statuses_have_bounded_same_source_retry(tmp_path, status):
    transport = Transport(HTTPError("https://host/?token=SECRET", status, "SECRET", {}, None), Response())
    mgr = manager(tmp_path, transport, mirrored_spec(), attempts_per_source=2)
    mgr.download("test-bf16")
    assert len(transport.calls) == 2 and transport.calls[0][0] == transport.calls[1][0]


def test_cancel_during_retry_backoff_is_not_swallowed(tmp_path):
    event = threading.Event()
    transport = Transport(Response(fail_after=4), Response())
    mgr = manager(tmp_path, transport, mirrored_spec(), attempts_per_source=2, retry_backoff=2)
    def progress(update):
        if update.status == "retrying":
            event.set()
    with pytest.raises(DownloadCancelled):
        mgr.download("test-bf16", cancel_event=event, progress=progress)
    assert len(transport.calls) == 1 and paths(mgr)[1].read_bytes() == DATA[:4]


def test_range_ignored_by_new_source_replaces_instead_of_appending(tmp_path):
    mgr = manager(tmp_path, Transport(OSError("offline"), Response()), mirrored_spec())
    write(paths(mgr)[1], b"wrong")
    mgr.download("test-bf16")
    assert mgr.transport.calls[1][1]["Range"] == "bytes=5-"
    assert paths(mgr)[0].read_bytes() == DATA


def test_existing_invalid_full_partial_is_preserved_but_no_longer_stuck(tmp_path):
    mgr = manager(tmp_path, Transport(Response()))
    _, part, _ = paths(mgr)
    write(part, b"x" * len(DATA))
    mgr.download("test-bf16")
    assert next(part.parent.glob(part.name + ".bad-*")).read_bytes() == b"x" * len(DATA)
    assert mgr.inspect("test-bf16").ready


def test_legacy_marker_requires_full_sha_and_migrates_without_network(tmp_path):
    spec = mirrored_spec()
    mgr = manager(tmp_path, spec=spec)
    final, _, marker = paths(mgr, spec)
    write(final, DATA)
    write(marker, json.dumps({"schema_version": 1, "model_id": spec.model_id,
                             "manifest_sha256": spec.legacy_fingerprint}).encode())
    assert mgr.inspect(spec.model_id).ready
    final.write_bytes(b"x" * len(DATA))
    assert not mgr.inspect(spec.model_id).ready
    final.write_bytes(DATA)
    mgr.download(spec.model_id)
    assert not mgr.transport.calls
    assert json.loads(marker.read_text())["manifest_sha256"] == spec.fingerprint


def test_disk_failure_is_not_retried_as_a_network_failure(tmp_path, monkeypatch):
    mgr = manager(tmp_path, Transport(Response(), Response()), mirrored_spec(), attempts_per_source=2)
    def disk_error(descriptor):
        raise OSError("disk full")
    monkeypatch.setattr(os, "fsync", disk_error)
    with pytest.raises(ModelDownloadError):
        mgr.download("test-bf16")
    assert len(mgr.transport.calls) == 1 and not paths(mgr)[2].exists()
