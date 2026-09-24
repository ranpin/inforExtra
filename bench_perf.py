"""open-jev 本地后端性能基准（E1 端到端延迟 / E2 批吞吐 / E3 一次性成本 / E4 扩展特性）。

口径：
- 同一台机器、同一进程内对比 seq（串行分块前向）与 batch（批量前向）两个变体
- 预热 WARMUP 条（排除 MPS 图编译等首次开销），正式 ROUNDS 轮
- 延迟报 p50/p95/mean；分阶段计时来自 jev_extract.PERF
- 旧 LLM 基线与 TypeSafe 官方 API 不在本脚本范围（端点/key 不可得），
  结果只描述 open-jev 本地后端的绝对性能

用法: .venv/bin/python bench_perf.py
产物: stdout 表格 + test/results/原始51_<backend>_性能.json（原始数据）
"""

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time

BACKEND = os.getenv("BENCH_BACKEND", "openjev")  # openjev | laya
os.environ.setdefault("JEV_BACKEND", BACKEND)

import jev_extract  # noqa: E402
from jev_extract import build_questions, generate_candidates  # noqa: E402
from compare_results import read_rows  # noqa: E402

WARMUP = 3
ROUNDS = 2
SCALING_REPS = 3


def load_inputs():
    rows = read_rows("test/测试集.xlsx")
    return [a for a, _ in rows if a and a != "用户输入"]


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def fresh_backend(batched=False):
    """重建后端（模型重新加载），返回 (backend, 加载秒数)。"""
    if BACKEND == "openjev":
        os.environ["JEV_OPENJEV_BATCHED"] = "1" if batched else "0"
    jev_extract._backend = None
    t0 = time.perf_counter()
    b = jev_extract.get_backend()
    return b, time.perf_counter() - t0


def run_suite(inputs):
    """E1+E2：预热后跑 ROUNDS 轮全量，返回逐条计时与阶段记录。"""
    for s in inputs[:WARMUP]:
        jev_extract.extract_entities(s)
    times, stages = [], []
    t_wall = time.perf_counter()
    for _ in range(ROUNDS):
        for s in inputs:
            t0 = time.perf_counter()
            jev_extract.extract_entities(s)
            times.append(time.perf_counter() - t0)
            stages.append(dict(jev_extract.PERF))
    wall = time.perf_counter() - t_wall
    return times, stages, wall


def summarize(name, times, stages, wall, load_s):
    n = len(times)
    row = {
        "variant": name,
        "load_seconds": round(load_s, 2),
        "n_utterances": n,
        "wall_seconds": round(wall, 1),
        "throughput_per_min": round(n / wall * 60, 1),
        "latency_p50_s": round(percentile(times, 0.50), 3),
        "latency_p95_s": round(percentile(times, 0.95), 3),
        "latency_mean_s": round(statistics.mean(times), 3),
        "stage_mean_s": {
            k: round(statistics.mean([s[k] for s in stages]), 3)
            for k in ("t_candidates", "t_round1", "t_round2")
        },
        "mean_candidates": round(statistics.mean([s["n_candidates"] for s in stages]), 1),
        "mean_questions_r1": round(statistics.mean([s["n_questions_r1"] for s in stages]), 1),
        "mean_chunks_r1": round(statistics.mean([s["n_chunks_r1"] for s in stages]), 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
    }
    print(f"\n--- {name} ---")
    for k, v in row.items():
        print(f"  {k}: {v}")
    return row


def run_scaling(backend, state, cands_full):
    """E4：固定 state，问题数递增，测 ask() 延迟（seq vs batch 由 backend.batched 控制）。"""
    out = {}
    for k in (2, 6, 12, len(cands_full)):
        cands = cands_full[:k]
        qs = build_questions(cands)
        backend.ask(state, qs)  # 预热
        ts = []
        for _ in range(SCALING_REPS):
            t0 = time.perf_counter()
            backend.ask(state, qs)
            ts.append(time.perf_counter() - t0)
        out[len(qs)] = {
            "mean_s": round(statistics.mean(ts), 3),
            "chunks": backend.last_chunks,
        }
    return out


def main():
    inputs = load_inputs()
    print(f"后端: {BACKEND}  测试集: {len(inputs)} 条话语, 预热 {WARMUP}, 正式 {ROUNDS} 轮")

    model_name = (jev_extract.OPEN_JEV_MODEL if BACKEND == "openjev"
                  else f"{jev_extract.LAYA_MODEL}/{jev_extract.LAYA_SUBFOLDER}")
    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": BACKEND,
            "machine": platform.machine(),
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip(),
            "python": sys.version.split()[0],
            "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
            "model": model_name,
            "candidate_filter": jev_extract.CANDIDATE_FILTER,
            "warmup": WARMUP,
            "rounds": ROUNDS,
        },
        "suites": [],
        "scaling": {},
    }

    variants = []
    if BACKEND == "openjev":
        b_seq, load_s = fresh_backend(batched=False)
        times, stages, wall = run_suite(inputs)
        results["suites"].append(summarize("openjev-seq", times, stages, wall, load_s))
        variants.append(("seq", b_seq))
        b_bat, load_s = fresh_backend(batched=True)
        times, stages, wall = run_suite(inputs)
        results["suites"].append(summarize("openjev-batch", times, stages, wall, load_s))
        variants.append(("batch", b_bat))
    else:
        b, load_s = fresh_backend()
        times, stages, wall = run_suite(inputs)
        results["suites"].append(summarize(BACKEND, times, stages, wall, load_s))
        variants.append((BACKEND, b))

    # E4: 扩展特性（取候选最多的一条话语）
    state = max(inputs, key=lambda s: len(generate_candidates(s)))
    cands_full = generate_candidates(state)
    print(f"\nE4 扩展特性: state={state!r} 候选={len(cands_full)}")
    for name, backend in variants:
        if BACKEND == "openjev":
            backend.batched = (name == "batch")
        sc = run_scaling(backend, state, cands_full)
        results["scaling"][name] = {"state": state, "by_questions": sc}
        print(f"  {name}: " + "  ".join(
            f"{q}问={v['mean_s']}s({v['chunks']}块)" for q, v in sc.items()))

    out_path = f"test/results/原始51_{BACKEND}_性能.json"
    os.makedirs("test/results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n原始数据已写入 {out_path}")


if __name__ == "__main__":
    main()
