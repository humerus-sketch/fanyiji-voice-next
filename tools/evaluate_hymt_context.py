"""Run explicit, public text fixtures through old and current Hy-MT policies."""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def old_policy(runtime, case):
    from fanyiji_next.hymt_translation import PARAMETERS
    language = "日语" if case["direction"] == "zh_to_ja" else "中文"
    messages = ([{"role": "system", "content": "以下是已经说过的前两句，仅供理解当前句，不要翻译这些背景信息：\n"
                  + "\n".join(case["previous"][-2:])}] if case["previous"] else [])
    messages.append({"role": "user", "content": f"将以下文本翻译为 {language}，注意只需要输出翻译后的结果，不要额外解释：\n"+case["current"]})
    rendered = runtime.post_json("/apply-template", {"messages": messages, "add_generation_prompt": True})["prompt"]
    result = runtime.post_json("/completion", {**PARAMETERS, "temperature": .7, "stream": False, "prompt": rendered})
    return result["content"].strip(), {k: result.get(k) for k in ("stop_type", "truncated", "timings")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--compare-old", action="store_true")
    parser.add_argument("--whisper-resident", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a prior evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
    from fanyiji_next.models import AppSettings
    from fanyiji_next.settings import with_local_models
    from fanyiji_next.hymt_translation import HyTranslationEngine, CONTEXT_RULE, PARAMETERS
    from fanyiji_next.translation_guard import candidate_issues
    settings = replace(with_local_models(AppSettings(), ROOT / "models"), recognition_backend="whisper",
                       recognition_device="cuda", recognition_compute_type="int8_float16", translation_backend="hymt")
    report = {"fixture":str(args.fixtures.resolve()),
              "fixture_sha256":hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
              "repeats":args.repeats,"whisper_resident":args.whisper_resident,
              "context_rule":CONTEXT_RULE,"parameters":PARAMETERS,"rows":[]}
    engine = recognizer = None
    try:
        if args.whisper_resident:
            from fanyiji_next.recognition_engines import create_recognition_engine
            import numpy as np
            import wave
            recognizer = create_recognition_engine(settings)
            with wave.open(str(ROOT/"models/self_test/zh.wav"),"rb") as wav:
                samples = np.frombuffer(wav.readframes(wav.getnframes()),dtype="<i2").astype(np.float32)/32768
            report["whisper_warmup_text"] = recognizer.recognize(samples,language="zh")
        started = time.perf_counter()
        engine = HyTranslationEngine(settings)
        report.update(load_ms=(time.perf_counter()-started)*1000,backend_info=engine.backend_info)
        # Fixed development-only warmup; held-out contents do not select prompts.
        for _ in range(2):
            engine.translate("会议将在10分钟后开始。","zh_to_ja",context_id="warmup")
            engine.translate("会議は10分後に始まります。","ja_to_zh",context_id="warmup")
        for repeat in range(args.repeats):
            for index,case in enumerate(fixture["cases"]):
                conditions = ["optimized","old_system"] if args.compare_old else ["optimized"]
                if (index+repeat)%2:
                    conditions.reverse()
                for condition in conditions:
                    row={"id":case["id"],"direction":case["direction"],"condition":condition,"repeat":repeat}
                    generated = []
                    generate = engine._generate
                    def record_generation(messages, **generation_options):
                        output, diagnostics = generate(messages, **generation_options)
                        generated.append(output)
                        return output, diagnostics
                    engine._generate = record_generation
                    started=time.perf_counter()
                    try:
                        if condition=="old_system":
                            output,diag=old_policy(engine.runtime,case)
                        else:
                            # Fixture history represents source utterances that
                            # already happened; never inject target references.
                            engine._history.clear()
                            engine._history[(case["id"],case["direction"])]=deque(case["previous"][-2:],maxlen=2)
                            output=engine.translate(case["current"],case["direction"],context_id=case["id"])
                            diag=deepcopy(engine.last_diagnostics)
                        row.update(output=output,diagnostics=diag,ok=True,
                                   candidate_issues=list(candidate_issues(case["current"],output,case["direction"])))
                    except Exception as exc:
                        row.update(ok=False,error={"type":type(exc).__name__,"message":str(exc)},
                                   diagnostics=deepcopy(engine.last_diagnostics) if condition=="optimized" else {})
                    finally:
                        engine._generate = generate
                    # These are public, fictional fixtures only. Production
                    # diagnostics never store source or generated subtitle text.
                    if generated:
                        row["generated_candidates"] = generated
                    row["elapsed_ms"]=(time.perf_counter()-started)*1000
                    report["rows"].append(row)
                    print(json.dumps({k:row[k] for k in ["id","condition","ok","elapsed_ms"]}),flush=True)
    except BaseException as exc:
        report["error"]={"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        for name, resource in (("engine",engine),("recognizer",recognizer)):
            if resource is None:
                continue
            try:
                if name=="engine":
                    log=engine.runtime.log_path
                    if log is not None and log.is_file():
                        shutil.copy2(log,args.output.with_suffix(".server.log"))
                resource.close()
            except Exception as cleanup:
                report.setdefault("cleanup_errors",[]).append({"resource":name,"type":type(cleanup).__name__})
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="")
    print(json.dumps({"report":str(args.output),"rows":len(report["rows"])}))


if __name__=="__main__":
    main()
