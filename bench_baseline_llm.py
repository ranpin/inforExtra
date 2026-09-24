"""旧管线（Qwen-2.5-Omni-7B + 巨型 prompt）提取调用的性能基线。

前置：zbxctl llm sidecar 已在本机 13984 端口运行（边车转发到平台模型服务，
与旧管线 10.23 测试时的部署形态一致）。

旧提取函数从 git main 分支动态加载（逐字一致，不复制粘贴）；
token 用量通过包装 client.chat.completions.create 捕获（不改旧逻辑）。

口径：
- 只测「提取调用」（对话调用在新旧管线中相同，不属于对比范围）
- 同一台机器、同一批 51 条测试话语、预热 3 条、正式 2 轮
- 调用间固定间隔 CALL_GAP 秒以避开平台 QPM 限流（不计入延迟）

用法: .venv/bin/python bench_baseline_llm.py
产物: stdout 表格 + test/results/原始51_omni7b_性能.json
"""

import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

WARMUP = 3
ROUNDS = 2
CALL_GAP = float(os.getenv("BENCH_CALL_GAP", "1.0"))


def load_old_extractor():
    """从 git main 分支加载旧版 extract_entities_with_model（含巨型 prompt）。"""
    src = subprocess.run(
        ["git", "show", "main:qwen_chat.py"],
        capture_output=True, text=True, check=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    ).stdout
    tmpdir = tempfile.mkdtemp(prefix="bench_baseline_")
    path = os.path.join(tmpdir, "old_qwen_chat.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("old_qwen_chat", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    usages = []
    orig_create = mod.client.chat.completions.create

    def wrapped(*args, **kwargs):
        resp = orig_create(*args, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            usages.append({
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            })
        return resp

    mod.client.chat.completions.create = wrapped
    return mod.extract_entities_with_model, usages, mod


def load_inputs():
    from compare_results import read_rows
    return [a for a, _ in read_rows("test/测试集.xlsx") if a and a != "用户输入"]


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    extract, usages, mod = load_old_extractor()
    inputs = load_inputs()
    print(f"旧管线基线: 模型={mod.EXTRACTION_MODEL} temp={mod.EXTRACTION_TEMPERATURE} "
          f"max_tokens={mod.EXTRACTION_MAX_TOKENS}")
    print(f"测试集: {len(inputs)} 条, 预热 {WARMUP}, 正式 {ROUNDS} 轮, 调用间隔 {CALL_GAP}s")

    for s in inputs[:WARMUP]:
        extract(s)
        time.sleep(CALL_GAP)
    usages.clear()  # 预热不计入 token 统计

    times = []
    failures = 0
    t_wall = time.perf_counter()
    for r in range(ROUNDS):
        for s in inputs:
            t0 = time.perf_counter()
            result = extract(s)
            times.append(time.perf_counter() - t0)
            if not result.get("persons") and "错误" in json.dumps(result, ensure_ascii=False):
                failures += 1
            time.sleep(CALL_GAP)
    wall = time.perf_counter() - t_wall

    report = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline": "old-llm-extraction (main branch, mega-prompt)",
            "model": mod.EXTRACTION_MODEL,
            "endpoint": "localhost:13984 sidecar -> platform",
            "warmup": WARMUP,
            "rounds": ROUNDS,
            "call_gap_s": CALL_GAP,
        },
        "n_calls": len(times),
        "wall_seconds_incl_gap": round(wall, 1),
        "latency_p50_s": round(percentile(times, 0.50), 3),
        "latency_p95_s": round(percentile(times, 0.95), 3),
        "latency_mean_s": round(statistics.mean(times), 3),
        "latency_min_s": round(min(times), 3),
        "latency_max_s": round(max(times), 3),
    }
    if usages:
        report["tokens_mean_prompt"] = round(
            statistics.mean(u["prompt_tokens"] for u in usages), 1)
        report["tokens_mean_completion"] = round(
            statistics.mean(u["completion_tokens"] for u in usages), 1)

    print("\n--- old-llm-extraction ---")
    for k, v in report.items():
        if k != "meta":
            print(f"  {k}: {v}")

    os.makedirs("test/results", exist_ok=True)
    with open("test/results/原始51_omni7b_性能.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n原始数据已写入 test/results/原始51_omni7b_性能.json")


if __name__ == "__main__":
    main()
