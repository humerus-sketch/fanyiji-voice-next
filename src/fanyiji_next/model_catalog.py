"""Pinned external model assets. Importing this catalog never accesses a network."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from urllib.parse import parse_qs, unquote, urlencode, urlsplit


_SILERO_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
# Current exact storage/CDN hosts come from Hugging Face's download docs and
# https://huggingface.co/.well-known/meta.json (2026-10-19); retain the previously
# supported legacy hosts. These remain redirect-only, never catalog entry URLs.
_HF_REDIRECT_HOSTS = frozenset({
    "cdn-lfs.huggingface.co", "cdn-lfs.hf.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co",
    "cas-bridge.xethub.hf.co", "cas-server.xethub.hf.co", "cas-server.xethub-eu.hf.co",
    "transfer.xethub.hf.co", "transfer.xethub-eu.hf.co",
    "aws.cdn.hf.co", "us.aws.cdn.hf.co", "us-east-1.aws.cdn.hf.co", "us-west-2.aws.cdn.hf.co",
    "eu-west-3.aws.cdn.hf.co", "ap-southeast-1.aws.cdn.hf.co",
    "us.gcp.cdn.hf.co", "us-east1.us.gcp.cdn.hf.co", "us-central1.us.gcp.cdn.hf.co",
    "us-west4.us.gcp.cdn.hf.co", "europe-west4.us.gcp.cdn.hf.co", "asia-southeast1.us.gcp.cdn.hf.co",
})
# Observed from ModelScope's fixed-revision download API for the catalogued
# TranslateGemma file (2026-10-19). Keep storage hosts redirect-only and exact.
_MS_REDIRECT_HOSTS = frozenset({"cdn-lfs-cn-1.modelscope.cn"})


def validate_relative_path(value: str) -> None:
    if (not isinstance(value, str) or not value or any(c in value for c in '\\:<>"|?*')
            or any(ord(c) < 32 or ord(c) == 127 for c in value) or value.startswith("/")):
        raise ValueError("模型文件路径无效。")
    parts = value.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (str(PurePosixPath(value)) != value or any(p in {"", ".", ".."} or p.endswith((".", " "))
            or p.split(".")[0].upper() in reserved or p.startswith(".") for p in parts)):
        raise ValueError("模型文件路径无效。")


def validate_download_url(value: str, *, redirect: bool = False) -> None:
    try:
        # urlsplit silently removes some controls; validate the supplied string
        # first so malformed input cannot be normalized into a trusted URL.
        if (not isinstance(value, str) or "\\" in value
                or any(ord(c) <= 32 or ord(c) == 127 for c in value)):
            raise ValueError
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
        allowed = host in {"huggingface.co", "hf-mirror.com", "modelscope.cn", "www.modelscope.cn"} or value == _SILERO_URL
        if redirect:
            allowed = (allowed or host in _HF_REDIRECT_HOSTS or host in _MS_REDIRECT_HOSTS
                       or host == "release-assets.githubusercontent.com")
        if (parsed.scheme != "https" or not allowed or parsed.username is not None or parsed.password is not None
                or parsed.port not in {None, 443} or parsed.netloc.endswith(":")
                or parsed.fragment or (parsed.query and not redirect and host not in {"modelscope.cn", "www.modelscope.cn"})):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("模型下载地址不在允许的 HTTPS 来源中。") from None


@dataclass(frozen=True)
class FileSpec:
    relative_path: str
    url: str
    size_bytes: int
    sha256: str
    sources: tuple[str, ...] = ()

    def __post_init__(self):
        validate_relative_path(self.relative_path)
        validate_download_url(self.url)
        parts = unquote(urlsplit(self.url).path).strip("/").split("/")
        pinned_hf = (urlsplit(self.url).hostname == "huggingface.co" and len(parts) >= 5 and parts[2] == "resolve"
                     and re.fullmatch(r"[0-9a-f]{40}", parts[3]) and "/".join(parts[4:]) == self.relative_path)
        pinned_silero = self.url == _SILERO_URL and self.relative_path == "silero_vad.onnx"
        if (not (pinned_hf or pinned_silero) or type(self.size_bytes) is not int
                or self.size_bytes <= 0 or not isinstance(self.sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)):
            raise ValueError("模型文件清单的版本、大小或校验值无效。")
        if (not isinstance(self.sources, tuple) or len(self.sources) > 4
                or any(not isinstance(source, str) for source in self.sources)
                or len(set(self.sources)) != len(self.sources)):
            raise ValueError("模型备用来源清单无效。")
        for source in self.sources:
            validate_download_url(source)
            parsed = urlsplit(source)
            source_parts = unquote(parsed.path).strip("/").split("/")
            mirror = (parsed.hostname in {"huggingface.co", "hf-mirror.com"}
                      and source_parts == parts and pinned_hf)
            query = parse_qs(parsed.query, keep_blank_values=True)
            modelscope = (parsed.hostname in {"modelscope.cn", "www.modelscope.cn"}
                          and len(source_parts) == 6 and source_parts[:3] == ["api", "v1", "models"]
                          and source_parts[5] == "repo" and set(query) == {"Revision", "FilePath"}
                          and len(query["Revision"]) == len(query["FilePath"]) == 1
                          and re.fullmatch(r"[0-9a-f]{40}", query["Revision"][0])
                          and query["FilePath"][0] == self.relative_path)
            if not (mirror or modelscope or source == self.url):
                raise ValueError("模型备用来源必须固定版本并对应同一文件。")

    @property
    def download_urls(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.sources, self.url)))


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    directory_name: str
    display_name: str
    files: tuple[FileSpec, ...]
    license_name: str
    source_url: str
    requires_license_ack: bool = False
    license_url: str = ""

    def __post_init__(self):
        if (not isinstance(self.model_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.model_id)
                or not isinstance(self.directory_name, str)
                or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*-(?:bf16|q4_k_m|onnx)", self.directory_name)
                or not isinstance(self.files, tuple) or not self.files
                or any(not isinstance(f, FileSpec) for f in self.files)
                or type(self.requires_license_ack) is not bool):
            raise ValueError("模型清单无效。")
        validate_relative_path(self.directory_name)
        names = [f.relative_path.casefold() for f in self.files]
        sidecars = [name + ".part" for name in names]
        occupied = names + sidecars
        if (len(occupied) != len(set(occupied))
                or any(a.startswith(b + "/") for a in occupied for b in occupied if a != b)):
            raise ValueError("模型文件路径重复或冲突。")

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    @property
    def fingerprint(self) -> str:
        # Download routes may evolve without invalidating already verified model
        # contents. Old markers are accepted only after a complete SHA recheck.
        body = {"model_id": self.model_id, "directory_name": self.directory_name,
                "files": [{"path": f.relative_path, "size": f.size_bytes, "sha256": f.sha256}
                          for f in self.files]}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @property
    def legacy_fingerprint(self) -> str:
        body = {"model_id": self.model_id, "directory_name": self.directory_name,
                "files": [{"path": f.relative_path, "url": f.url, "size": f.size_bytes, "sha256": f.sha256}
                          for f in self.files]}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _files(repo, revision, rows, *, modelscope_repo=None, modelscope_revision=None, revision_overrides=None):
    files = []
    for path, size, digest in rows:
        sources = []
        if modelscope_repo:
            ms_revision = (revision_overrides or {}).get(path, modelscope_revision)
            if ms_revision:  # No matching domestic bytes: retain the fixed HF routes.
                sources.append(f"https://modelscope.cn/api/v1/models/{modelscope_repo}/repo?" + urlencode(
                    {"Revision": ms_revision, "FilePath": path}))
        sources.append(f"https://hf-mirror.com/{repo}/resolve/{revision}/{path}")
        files.append(FileSpec(path, f"https://huggingface.co/{repo}/resolve/{revision}/{path}",
                              size, digest, tuple(sources)))
    return tuple(files)


QWEN_ASR = "qwen3-asr-0.6b-hf-bf16"
FUN_ASR = "fun-asr-nano-2512-bf16"
TRANSLATE_GEMMA = "translategemma-4b-it-q4_k_m"
SILERO_VAD = "silero-vad-onnx"

_qwen = ModelSpec(QWEN_ASR, QWEN_ASR, "Qwen3-ASR 0.6B · BF16", _files(
    "Qwen/Qwen3-ASR-0.6B-hf", "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c", (
        ("chat_template.jinja", 1434, "f50e6b694fbf4a683206e37869990d68333fe95d285730f084c838a34b0d98c2"),
        ("config.json", 2398, "9eecf6f1b383e343889c2e6010e632590fa57d4bc678e151c7d6a160a0dfb04a"),
        ("generation_config.json", 165, "9939fc9388b79bd70757f938b87381e817173d6a6158f5af6506c0b73e775c3c"),
        ("model.safetensors", 1564928088, "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35"),
        ("processor_config.json", 487, "bc0b230081b44e629dd5b9045b78495615c1831b4b9f4cffe97bd37e82a6156a"),
        ("tokenizer.json", 11429653, "fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05"),
        ("tokenizer_config.json", 998, "945e980986de2ca7768f3326bfdbb4fbea3406f972b8ae0be233089f2b253c11"),
    ), modelscope_repo="Qwen/Qwen3-ASR-0.6B-hf",
    modelscope_revision="c4650525d0d40f32f1517abe624a89be04386bef",
    # This template changed separately; the older commit has different bytes.
    revision_overrides={"chat_template.jinja": "005b749cd1566dc579fe0238316087e2fc1945c0"}),
    "Apache-2.0", "https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf")

_fun = ModelSpec(FUN_ASR, FUN_ASR, "Fun-ASR-Nano 800M · BF16", _files(
    "FunAudioLLM/Fun-ASR-Nano-2512", "272c57b82523ada6fd87095e955f8e29100979ab", (
        ("Qwen3-0.6B/config.json", 726, "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
        ("Qwen3-0.6B/generation_config.json", 239, "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
        ("Qwen3-0.6B/merges.txt", 1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
        ("Qwen3-0.6B/tokenizer.json", 11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        ("Qwen3-0.6B/tokenizer_config.json", 9732, "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
        ("Qwen3-0.6B/vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
        ("config.yaml", 3140, "daed38ea6484f5650fb32cbd9069b9aa13880acaf2bcb1f0bf4be2712837917c"),
        ("configuration.json", 398, "b64a3a55d35bcbe2cf4d31f2d3ef25a423d3ba2ebff203298c27fa055f3c7612"),
        ("model.pt", 1971149431, "55ae0d2fee369f0f11cce0795f6927934ad17cf11b278a7e56a51272074160bb"),
        ("multilingual.tiktoken", 907395, "747979631e813193436aabcff7c1c235d37de8097b71c563ec8b63b7a515c718"),
    ), modelscope_repo="FunAudioLLM/Fun-ASR-Nano-2512",
    modelscope_revision="c819b4dcbf5f76bc86f9cf488213606ca1832715",
    revision_overrides={
        "config.yaml": "16795894cff13ca4f89b301dd08c22f60351e844",
        "configuration.json": "39f6a4d23e678bee2a471da2f74224a314b2fafc",
        "multilingual.tiktoken": "9e733102544707cd890655071ee3f66269355165",
        # Available ModelScope weights have different bytes and cannot be mixed.
        "model.pt": None}),
    "Apache-2.0", "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512")

_gemma = ModelSpec(TRANSLATE_GEMMA, TRANSLATE_GEMMA, "TranslateGemma 4B · Q4_K_M（第三方量化）", _files(
    "mradermacher/translategemma-4b-it-GGUF", "d37868f759d2f817a43d28dd80cf9f982ffa62fd", (
        ("translategemma-4b-it.Q4_K_M.gguf", 2489909760, "81200d03e843d2ec1ece6eeafe7d13cb6e5211e1fcd336ade55790b683a08330"),
    ), modelscope_repo="prithivMLmods/translategemma-4b-it-f32-GGUF",
    modelscope_revision="331bd735819210f93a62833b306dd8fa4b70b074"),
    "Gemma", "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF", True,
    "https://ai.google.dev/gemma/terms")

_silero = ModelSpec(SILERO_VAD, SILERO_VAD, "Silero VAD · ONNX（共享语音检测）", (
    FileSpec("silero_vad.onnx", _SILERO_URL, 643854,
             "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6", (
                 # Pinned community mirrors; both complete files match the original SHA.
                 "https://modelscope.cn/api/v1/models/xnnehang/k2-fsa-silero-vad/repo?"
                 "Revision=e64bab140550f46d724405a77ce842d2095feeef&FilePath=silero_vad.onnx",
                 "https://modelscope.cn/api/v1/models/liaowenbin/silero_vad/repo?"
                 "Revision=a9cffd6df29686d5e36fbe9d3baf6ef62d9c16a9&FilePath=silero_vad.onnx")),
), "MIT", "https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html")

MODEL_CATALOG = MappingProxyType({m.model_id: m for m in (_qwen, _fun, _gemma, _silero)})
MODEL_COMBINATIONS = MappingProxyType({"A": (QWEN_ASR, TRANSLATE_GEMMA, SILERO_VAD),
                                      "B": (FUN_ASR, TRANSLATE_GEMMA, SILERO_VAD)})
