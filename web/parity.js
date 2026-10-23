/**
 * 端到端对拍：浏览器内用 q8 模型跑 JS 管线（pipeline.js），
 * 与 Python 管线在 51 条真实种子集上的输出逐条 JSON 全等比较。
 * gold = web/public/parity51.json（由 test/results/原始51_laya强制选择v4.xlsx 生成）。
 * 结果挂 window.__PARITY，完成置 window.__DONE。
 */
import { load } from "@johnhenry/laya";
import { extractEntities } from "./pipeline.js";

const logEl = document.getElementById("log");
const log = (s) => {
  logEl.textContent += s + "\n";
  console.log(s);
};

async function main() {
  const data = await (await fetch("/parity51.json")).json();
  log(`sentences: ${data.length}`);
  log("loading model (laya-multilingual q8, self-hosted)...");
  const t0 = performance.now();
  const agent = await load(location.origin + "/models/laya-q8/", {
    onProgress: (p) => {
      try {
        logEl.textContent = "loading... " + JSON.stringify(p).slice(0, 140) + "\n";
      } catch (_) { /* ignore */ }
    },
  });
  const loadSeconds = (performance.now() - t0) / 1000;
  const backendName =
    (agent.backend && (agent.backend.name || agent.backend.constructor.name)) || "?";
  log(`loaded in ${loadSeconds.toFixed(1)}s backend=${backendName} dtype=${agent.dtype || "?"}`);

  let ok = 0;
  const mismatches = [];
  const t1 = performance.now();
  for (const item of data) {
    const r = await extractEntities(agent, item.text);
    if (JSON.stringify(r.persons) === JSON.stringify(item.expected)) {
      ok += 1;
    } else {
      mismatches.push({ text: item.text, expected: item.expected, got: r.persons });
    }
  }
  const inferSeconds = (performance.now() - t1) / 1000;
  window.__PARITY = {
    ok, total: data.length, mismatches,
    loadSeconds, inferSeconds, backendName,
  };
  window.__DONE = true;
  log(`parity: ${ok}/${data.length} in ${inferSeconds.toFixed(1)}s ` +
      `(${(inferSeconds / data.length * 1000).toFixed(0)}ms/sentence)`);
  for (const m of mismatches.slice(0, 12)) {
    log(`MISMATCH ${m.text}\n  exp=${JSON.stringify(m.expected)}\n  got=${JSON.stringify(m.got)}`);
  }
}

main().catch((e) => {
  log("ERROR: " + String((e && e.stack) || e));
  window.__DONE = true;
});
