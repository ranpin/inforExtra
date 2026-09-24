"""用指定模型跑旧管线（巨型 prompt）提取，与 gold 对比。

旧提取函数从 git main 分支逐字加载；新版 openai SDK 的 ChatCompletion
可迭代会触发旧代码的流式误判，用 SimpleNamespace 包装响应绕过
（不改旧代码本身）。

用法: .venv/bin/python eval_llm_baseline.py [模型名] [--in 输入.xlsx] [--out 输出.xlsx]
默认: qwen3.8-max-aliyun，输入为 510 条扩充集
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from contextlib import redirect_stdout

import pandas as pd

MODEL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else "qwen3.8-max-aliyun"
NO_THINKING = "--no-thinking" in sys.argv  # Qwen3 系推理模型关思考（平台透传 enable_thinking）


def arg(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


IN_FILE = arg("--in", "test/测试集_扩充10x.xlsx")
OUT_FILE = arg("--out", f"test/results/扩充510_{MODEL}"
                        f"{'_关思考' if NO_THINKING else ''}.xlsx")
CALL_GAP = float(os.getenv("EVAL_CALL_GAP", "1.0"))  # 平台 QPM 限流保护

os.environ["OPENAI_EXTRACTION_MODEL"] = MODEL  # model_config 在 import 时读取

# 从 git 历史加载旧管线模块（巨型 prompt 逐字一致）
_src = subprocess.run(["git", "show", "main:qwen_chat.py"],
                      capture_output=True, text=True, check=True).stdout
_tmp = tempfile.mkdtemp(prefix="eval_llm_")
_path = os.path.join(_tmp, "old_qwen_chat.py")
with open(_path, "w", encoding="utf-8") as f:
    f.write(_src)
_spec = importlib.util.spec_from_file_location("old_qwen_chat", _path)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

usages = []
_orig_create = mod.client.chat.completions.create


def safe_create(*args, **kwargs):
    if NO_THINKING:
        extra = dict(kwargs.get("extra_body") or {})
        extra["enable_thinking"] = False
        kwargs["extra_body"] = extra
    resp = _orig_create(*args, **kwargs)
    usage = getattr(resp, "usage", None)
    if usage is not None:
        usages.append((getattr(usage, "prompt_tokens", 0),
                       getattr(usage, "completion_tokens", 0)))
    # SimpleNamespace 不可迭代 → 旧代码的 hasattr(__iter__) 走非流式分支
    return types.SimpleNamespace(choices=resp.choices, usage=usage)


mod.client.chat.completions.create = safe_create

from compare_results import read_rows  # noqa: E402


def main():
    # 预检：模型名在平台上是否可用
    try:
        r = _orig_create(model=MODEL, stream=False, max_tokens=16,
                         messages=[{"role": "user", "content": "你好，回复OK"}])
        print(f"预检通过: {MODEL} -> {r.choices[0].message.content!r}")
    except Exception as e:  # noqa: BLE001
        print(f"预检失败（模型 {MODEL} 在平台不可用？）: {e}")
        sys.exit(1)

    inputs = [a for a, _ in read_rows(IN_FILE) if a and a != "用户输入"]
    print(f"输入 {len(inputs)} 条, 调用间隔 {CALL_GAP}s, "
          f"思考模式: {'关' if NO_THINKING else '开'}")

    results, times, empty = [], [], 0
    t_start = time.perf_counter()
    for i, text in enumerate(inputs):
        t0 = time.perf_counter()
        buf = io.StringIO()
        with redirect_stdout(buf):
            entities = mod.extract_entities_with_model(text)
        times.append(time.perf_counter() - t0)
        results.append({"用户输入": text,
                        "JSON输出": json.dumps(entities, ensure_ascii=False)})
        if not entities.get("persons"):
            empty += 1
        if (i + 1) % 50 == 0:
            recent = times[-50:]
            print(f"  {i + 1}/{len(inputs)} 完成, 近 50 条均值 {sum(recent)/len(recent):.2f}s")
        time.sleep(CALL_GAP)

    os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
    pd.DataFrame(results).to_excel(OUT_FILE, index=False)
    ts = sorted(times)
    n = len(ts)
    print(f"\n完成: {OUT_FILE}  (总 wall {time.perf_counter()-t_start:.0f}s)")
    print(f"延迟 p50={ts[n//2]:.2f}s p95={ts[int(n*0.95)]:.2f}s "
          f"mean={sum(ts)/n:.2f}s max={ts[-1]:.2f}s")
    print(f"空结果（含合法空与错误）: {empty}/{n}")
    if usages:
        print(f"tokens 均值: prompt={sum(u[0] for u in usages)/len(usages):.0f} "
              f"completion={sum(u[1] for u in usages)/len(usages):.0f}")
    print(f"\n下一步: .venv/bin/python compare_results.py {OUT_FILE} "
          f"test/测试集_扩充10x_gold.xlsx")


if __name__ == "__main__":
    main()
