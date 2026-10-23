/**
 * 端到端对拍：浏览器内跑 JS 管线（pipeline.js），与 Python 管线在 51 条
 * 真实种子集上的输出逐条 JSON 全等比较。模型走 live.js 的 URL 直连路径
 * （fp16 ~678MB，hf.co 探活失败自动切 hf-mirror）——与展示页发货路径一致。
 * gold = web/public/parity51.json（由 v5 Python 管线对 test/测试集.xlsx 重跑生成）。
 * 结果挂 window.__PARITY，完成置 window.__DONE。
 */
import { loadModel } from "./live.js";
import { extractEntities } from "./pipeline.js";

const logEl = document.getElementById("log");
const log = (s) => {
  logEl.textContent += s + "\n";
  console.log(s);
};

async function main() {
  const data = await (await fetch("/parity51.json")).json();
  log(`sentences: ${data.length}`);
  log("loading model (laya-multilingual fp16, URL route via live.js)...");
  const t0 = performance.now();
  const agent = await loadModel((p) => {
    if (p && p.loaded !== undefined) {
      try {
        logEl.textContent = `loading... ${p.file || ""} ${(p.loaded / 1e6).toFixed(0)}MB\n`;
      } catch (_) { /* ignore */ }
    }
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
