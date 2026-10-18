# 翻译姬 · 语音版 NEXT

Windows x64 本地语音字幕工具，支持麦克风、电脑声音、本次会话滚轮回看、手动导出、窗口透明度和软件内更新。程序启动后等待手动操作，不自动录音，不自动下载模型。

## A/B 方案与模型下载

本版只提供以下两套新方案，两个方案可以同时下载、在停止采集后切换：

| 方案 | 语音识别 | 翻译 |
|---|---|---|
| A | Qwen3-ASR-0.6B-hf BF16 | TranslateGemma 4B Q4_K_M |
| B | Fun-ASR-Nano-2512 BF16 | 同一份 TranslateGemma 4B Q4_K_M |

首次安装包和后续更新包都不附带模型权重，包括识别、翻译、VAD、GTCRN 或 MDX 权重。模型下载与程序更新分开；切换 A/B 时共用翻译模型，不复制两份。此前 Whisper、SenseVoice、Hy-MT2、NLLB、M2M100 等模型属于旧方案，不再作为新安装的下载选项。

1. 点击主窗口右下角 **模型下载 · A/B**。
2. 查看固定模型目录与许可说明，选择 **下载方案 A** 或 **下载方案 B**。TranslateGemma 的第三方量化文件遵循 Gemma 条款，需要先阅读并确认。
3. 下载窗口显示文件进度、来源和重试状态，支持取消与后续续传。源失败时先重试，再自动切换备用源；文件完成后按固定版本、大小和 SHA256 校验，不完整或校验失败的模型不会标记为可用。
4. 停止采集后点击 **使用方案 A/B**。仅下载不切换当前方案，也不会开启采音。
5. 确认声音来源和目标语言，点击 **开始翻译**。

模型首次下载需要网络，识别与翻译在本地运行。启动和正常本地推理不应触发后台补下载。下载源固定到指定 revision；不执行下载仓库的任意 Python 代码。

Qwen、TranslateGemma、Silero 和 Fun-ASR 的配套小文件优先使用固定版本的 ModelScope 国内源。Fun-ASR 当前选定的 `model.pt` 尚未找到 SHA256 完全一致的 ModelScope 文件，保留 HF-Mirror 与 Hugging Face 顺序回退；不会为了换源替换不同权重。HF-Mirror 可能继续重定向至 Hugging Face，国内可达性仍取决于实际网络，不能保证所有地区可用。新安装默认方案 A，已明确选择的方案 B 会保留。

## 固定后缀、跟随主文件名的目录

固定的是 **`_models` 后缀**，前缀是主程序文件名（不含 `.exe`），目录位于主程序同级。目录不会跟随当前工作目录或安装包解压文件夹名称改变。

```text
FanyiJiVoiceNext.exe
FanyiJiVoiceNext_models/
  qwen3-asr-0.6b-hf-bf16/
  fun-asr-nano-2512-bf16/
  translategemma-4b-it-q4_k_m/
  silero-vad-onnx/
_internal/
_updater/
runtime/
licenses/
```

例如将主文件改名为 `直播字幕.exe`，对应目录就是 `直播字幕_models`。已有下载若需要继续复用，应将原模型目录一并改为新名称；程序不会为了主文件改名而自动删除旧目录或再次复制模型。

源码入口 `run.py` 对应项目根目录的 `run_models/`。诊断工具显式传入的 `--model-root` 仍用于指定诊断输入，不能据此改变 GUI 下载目录规则。设置窗口显示实际配置与模型根目录，便于确认使用的是哪一份。

## 采集、字幕与阅读

电脑声音使用 WASAPI loopback，不需要虚拟声卡。两路可独立启用和选择设备，只有点击开始后才加载模型、开启所选设备。停止后等待资源释放，再切换模型或设备；不会为切换而清空已经显示的字幕。

本次窗口内保留全部会话字幕。默认跟随最新内容；向上滚动即可回看，回看时新句不会强制把位置拉回底部。点击 **返回最新** 恢复跟随。需要留存时手动导出；关闭窗口后不自动恢复本次字幕，也不会后台保存真实字幕正文。

单句翻译失败时以原文显示。普通错误提示和错误弹窗统一在三秒后关闭，同一错误重复到达不延长当前显示时间；许可和操作确认仍需手动处理。错误自动消失不会删除原文或字幕，也不把失败句标记为翻译成功。

顶部 **透明** 按钮控制整个主窗口的透明效果，**阅读设置 → 窗口透明度** 可调整 0%–70%。声音与模型设置窗口保持不透明，阅读调整不重启采集。

识别模型支持哪些语言与 GUI 的翻译目标是两件事。自动语言结果必须随当前片段传递，不能因为目标为中文就把所有声音强制按日语识别。对极短句、歌声、混合语种或不明确语言，不能把模型分数当成人工确认结果。

## 程序更新与模型保留

在窗口右下角点击 **检查更新 → 下载更新 → 安装并重启**。开始安装前须停止采集。更新清单只包含程序和运行库，拒绝 `models/`、任意 `*_models/` 目录及模型权重；安装不能覆盖或删除已经下载的模型、用户设置或录音。

更新和模型下载互相独立。模型完成一次校验后，后续程序更新无需重新下载同一套权重。新主窗口成功打开并运行 3 秒后，后台自动清理本次更新的旧程序备份、暂存文件和下载包，保留小型更新日志；失败、取消或未成功启动时保留回退文件。清理不加载模型，不扫描删除其它安装目录。源码模式只检查更新，不覆盖源码安装。

0.5.3 首次引入完整 A/B 运行库，旧版本先安装兼容引导补丁，重启后在软件内继续下载运行库分卷，逐卷及合并后均校验 SHA256，再安装重启。运行库包含 PyTorch/CUDA 等程序依赖，不包含模型权重；需为下载、解压和事务备份预留空间。下载中断可以重开软件继续，后续普通程序更新无需再次下载相同运行库。

旧版发布的 `full.7z.*` 分卷包含旧模型，属于历史发布，不是本版推荐安装方式；新版本不会撤回或改写这些历史附件。

## 来源与许可

- A 识别：[Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)，Apache-2.0。
- B 识别：[FunAudioLLM/Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)，Apache-2.0。
- 翻译原模型：[google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it)。下载文件为 [mradermacher 的第三方 Q4_K_M 量化](https://huggingface.co/mradermacher/translategemma-4b-it-GGUF)，遵守 [Gemma 条款](https://ai.google.dev/gemma/terms)，不能当作 Apache 或 MIT 模型。

软件保存固定 revision、文件长度、SHA256 和来源信息。模型能力与作者评测不等于本机直播准确率，量化文件也不因公开下载而自动具有不同于原模型的许可。

## 源码环境与打包

项目使用独立 Python 3.12 环境。现有 `.venv` 的运行库不为新模型试验做隐式大版本升级；A/B 所需的 PyTorch、Transformers、FunASR 环境单独准备。不同环境的安装或构建成功不等于 GPU 推理或准确率验证通过。

- 单元测试：`.venv/Scripts/python.exe -m pytest -q`
- 不采音界面检查：`.venv/Scripts/python.exe run.py --smoke-test --report artifacts/voice/ui.json`
- 运行库检查：`.venv/Scripts/python.exe run.py --check-runtime --report artifacts/voice/runtime.json`
- 仅界面与旧兼容运行库构建（不包含 A/B 推理依赖）：`.venv/Scripts/python.exe tools/build.py --edition voice --with-updater`
- A/B 运行库完整构建：在已经装好全部依赖的独立环境中运行 `python tools/build.py --edition voice --with-updater --with-ab-engines`。

`--with-ab-engines` 在构建前检查所需分发包 metadata，显式收集代码、原生运行库和许可证，不收集模型数据；FunASR 注册器需要的 `.py` 实现与 `version.txt` 通过精确 hook 保留，属于运行代码。缺少依赖时直接失败，不把未打包的引擎报告为可用。构建与发布打包都拒绝带入 ONNX、GGUF、safetensors、PT、PTH 等权重；SentencePiece 的四个精确路径 Unicode 归一化 `.bin` 查表属于运行数据，不能据此放行其它 `.bin`。

`tools/package_release.py` 制作无模型便携目录与程序更新 ZIP；旧 `--models` 和 `--exclude-mdx` 参数只为旧构建脚本兼容保留，不读取或携带模型。`tools/package_patch.py` 保持 schema-1 覆盖更新兼容；基线模型只保留，不重读、重哈希或复制进补丁。构建始终使用新输出目录，不覆盖已有发布。

## 验证边界与历史记录

A/B 是保留的候选配置，不因下载、文件校验或启动成功而宣称已经通过“不漏句、准确性不降且说完后三秒内出译文”的联合验收。识别语言分数、多个 ASR 结果一致、译文非空或无运行报错均不能替代有可信逐字稿或实际听辨的准确性核验。

此前 Q8 残句扩写、否定与指代错误、GTCRN 混音改变识别、演唱时出现观看感谢语等失败记录仍保留：[Q8 与同源识别报告](artifacts/q8-current-repair-20261018/result.md)、[GTCRN 同源对照](artifacts/gtcrn-preserve-r2-20261017/validation.md)。清理模型不删除源码、用户配置、录音或这些对照记录。旧模型实验与历史耗时不能当作新 A/B 组合的验收结果。
