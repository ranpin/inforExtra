/**
 * 展示页 live 推理集成层（esbuild 打包为 docs/assets/laya-live.bundle.js）。
 *
 * 模型源：HF repo aac6fef/laya-multilingual-mlx（fp16 ~678MB，Apache-2.0，
 * convaiinnovations/laya-multilingual 的 MLX 导出），revision 钉死（URL 不可变）。
 *
 * 为什么走 URL 直连（resolve/<commit>/ + Range 请求）而不是 Hub repo ID：
 * Chromium Cache API 单条目上限实测在 200~650MB 之间（650MB 在
 * headless 默认/--unlimited-storage/持久 profile 三种模式全部 put 失败，
 * quota 11.3GB 非瓶颈），643.8MB 的 model.safetensors 走 Hub 路径必炸；
 * URL 路径不经 Cache API——代价是无持久缓存，每次启用重新下载（页面已如实标注）。
 *
 * 国内可达性：探活 hf.co（473B 小文件，4s 超时），不可达自动切 hf-mirror.com
 * （其 CORS 回显 Origin 且 expose Accept-Ranges/Content-Range，跨域 Range 已验证）。
 *
 * 提取管线：web/pipeline.js（jev_extract.py v4 的逐函数移植，51 条对拍
 * 50/51 逐字节一致；唯一分歧是阈值边界翻转且浏览器侧语义更对——
 * 『我右边坐的是日产的陈总』Python noul<0.5 漏检、浏览器 ≥0.5 正确提取）。
 */
import { load } from "@johnhenry/laya";
import { extractEntities, PERF } from "./pipeline.js";

const REPO = "aac6fef/laya-multilingual-mlx";
// 钉死 revision：不可变 URL（可复现；浏览器 HTTP 缓存若命中可加速重复访问）
const COMMIT = "f2b4faf51023039425946074e2cf1361d2db11d5";
const HF_CO_BASE = `https://huggingface.co/${REPO}/resolve/${COMMIT}/`;
const HF_MIRROR_BASE = `https://hf-mirror.com/${REPO}/resolve/${COMMIT}/`;
const PROBE_FILE = "rl_agent_config.json";
/** 资产总量（model 643.8 + tokenizer 34.4 + 配置零头），进度显示用。 */
export const MODEL_TOTAL_MB = 678;

let agent = null;
let loadingPromise = null;

export function isReady() {
  return agent !== null;
}

/** 探活 hf.co；不可达返回 hf-mirror base。 */
async function pickBase() {
  try {
    const res = await fetch(HF_CO_BASE + PROBE_FILE, {
      signal: AbortSignal.timeout(4000),
    });
    if (res.ok) return HF_CO_BASE;
  } catch (_) { /* 不可达 → 镜像 */ }
  return HF_MIRROR_BASE;
}

/** 带字节计数的 fetch 包装：URL 路径无 onProgress，从响应体流上计数。 */
function progressFetch(onProgress) {
  let loaded = 0;
  return async (input, init) => {
    const res = await fetch(input, init);
    if (!res.ok || !res.body) return res;
    const url = typeof input === "string" ? input : input.url;
    const file = url.split("/").pop().split("?")[0];
    const counter = new TransformStream({
      transform(chunk, controller) {
        loaded += chunk.byteLength;
        if (onProgress) onProgress({ file, loaded });
        controller.enqueue(chunk);
      },
    });
    return new Response(res.body.pipeThrough(counter), {
      status: res.status,
      statusText: res.statusText,
      headers: res.headers,
    });
  };
}

/** 加载模型（幂等：重复调用返回同一 promise）。onProgress({file, loaded})。 */
export async function loadModel(onProgress) {
  if (agent) return agent;
  if (!loadingPromise) {
    loadingPromise = (async () => {
      const base = await pickBase();
      agent = await load(base, { fetch: progressFetch(onProgress) });
      return agent;
    })();
  }
  return loadingPromise;
}

/** 单句提取：返回 {persons, elapsedMs, perf}。 */
export async function extract(text) {
  if (!agent) throw new Error("model not loaded");
  const t0 = performance.now();
  const r = await extractEntities(agent, text);
  return {
    persons: r.persons,
    elapsedMs: performance.now() - t0,
    perf: { nCandidates: PERF.nCandidates, nQ1: PERF.nQ1, nQ2: PERF.nQ2 },
  };
}

export function backendInfo() {
  if (!agent) return null;
  const b = agent.backend;
  return {
    backend: (b && (b.name || (b.constructor && b.constructor.name))) || "?",
    dtype: agent.dtype || "?",
  };
}
