# Model and runtime notices

This directory preserves original upstream license texts alongside their source URLs and SHA256 digests in `sources.json`. The project does not replace or relicense those terms. The notices and model files have separate licenses from the application. No endorsement by the upstream authors is implied.

## Included in the public package

| Component | Attribution and source | License and local format |
| --- | --- | --- |
| Whisper large-v3 | OpenAI, [Whisper](https://github.com/openai/whisper), [upstream model](https://huggingface.co/openai/whisper-large-v3), [CTranslate2 conversion](https://huggingface.co/Systran/faster-whisper-large-v3/tree/edaa852ec7e145841d8ffdb056a99866b5f0a478) | MIT; see `Whisper-MIT-LICENSE.txt`. The package uses the existing CTranslate2 conversion; the default recognition compute type is `int8_float16`. Conversion source revision and original file hashes are retained in `Whisper-conversion-provenance.json`. |
| SenseVoiceSmall | Alibaba Group / FunAudioLLM; [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), [sherpa-onnx model package](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tree/2365baeacb507f821a0c8120fcee3d484dba7a07) | FunASR Model Open Source License Agreement Version 1.1; see the complete `SenseVoice-FunASR-MODEL-LICENSE.txt` and the original reference notice. Packaged as the existing int8 ONNX model; its model license must not be confused with a runtime software license. |
| Hy-MT2-1.8B Q8_0 | Tencent Hunyuan; [official GGUF model repository](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | Apache License 2.0; see `Hy-MT2-Apache-2.0-LICENSE.txt`. The official `Hy-MT2-1.8B-Q8_0.gguf` file is included without local weight edits. |
| M2M100 418M | Facebook, Inc. and its affiliates; [official model repository](https://huggingface.co/facebook/m2m100_418M/tree/55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636) | MIT; see `M2M100-MIT-LICENSE.txt`. The package retains its existing CTranslate2 model and SentencePiece tokenizer. |
| NLLB-200-distilled-1.3B | Meta / Facebook; [official model repository](https://huggingface.co/facebook/nllb-200-distilled-1.3B/tree/7be3e24664b38ce1cac29b8aeed6911aa0cf0576) | **CC-BY-NC-4.0: NonCommercial use only** under that license. See the complete `NLLB-CC-BY-NC-4.0-LICENSE.txt`. Local technical changes: conversion using CTranslate2 4.8.2 and Transformers 4.57.6 with `int8_float16` quantization; tokenizer retained. Source revision and conversion details are recorded in `NLLB-conversion-provenance.json`. |
| GTCRN | Rong Xiaobin; [official GTCRN project](https://github.com/Xiaobin-Rong/gtcrn), [sherpa-onnx integration](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.8/scripts/gtcrn/README.md), [packaged model source](https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx) | MIT; see `GTCRN-MIT-LICENSE.txt`. Existing streaming ONNX model is included without local weight edits. |
| Silero VAD | Silero Team; [official project](https://github.com/snakers4/silero-vad) | MIT; see `Silero-VAD-MIT-LICENSE.txt`. Existing `silero_vad.onnx` is included. |
| llama.cpp b11696 runtime | The ggml authors; [fixed upstream release](https://github.com/ggml-org/llama.cpp/tree/b11696) | MIT; see `llama-cpp-b11696-MIT-LICENSE.txt`. The runtime's separate LLVM OpenMP and other bundled library notices also remain in the portable runtime and application license directories. |

The original model READMEs and notices that were present locally remain in their corresponding packaged model directories. NVIDIA CUDA/cuDNN, Python, Qt/PySide6 and other dependency notices are collected by the program build into its `licenses/` directory; the license of a model does not replace those library licenses.

## MDX is not included in this public package

The user requested that the public release omit `Kim_Vocal_2.onnx`. The inventory therefore marks MDX Kim Vocal 2 as excluded, even though the application retains support for loading it. The default `MDX→GTCRN` setting is preserved; users must supply an appropriately licensed model themselves or explicitly select another processing mode. The program does not silently switch its default.

For source attribution and reference, [UVR's official README License section](https://github.com/Anjok07/ultimatevocalremovergui#license) discusses its models, the [official model download table](https://github.com/Anjok07/ultimatevocalremovergui/blob/master/gui_data/model_manual_download.json) lists Kim Vocal 2, and [UVR's source](https://github.com/Anjok07/ultimatevocalremovergui/blob/master/gui_data/constants.py) identifies the [TRvlvr model repository](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models) used by the existing local model. `UVR-historical-2022-MIT-LICENSE.txt` is an unchanged historical upstream license snapshot at commit `0fd6b751dbbe9da9c96660d0b4ec2b07108727b7`; it is retained as a dated reference and is not a claim that this public package contains the MDX weights.

## 中文说明

公开包包含识别、翻译、GTCRN 与 VAD 运行模型，不包含 MDX 模型、训练用原始权重、用户设置或录音。NLLB 使用 CC-BY-NC-4.0，须遵守非商业用途等条款；SenseVoice 的模型许可证以完整 FunASR Model License 1.1 为准。各模型、运行库遵循各自原始许可，本应用的许可证不会覆盖它们。官方原文文件保持下载或现有文件的原始字节，来源及校验值见 `sources.json`。
