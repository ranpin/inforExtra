"""座舱人员信息提取 demo 服务。

零额外依赖：只用标准库 http.server 提供页面与 /api/extract 接口，
提取逻辑复用仓库根的 jev_extract.py（laya 本地后端，单条前向 ~50ms）。

启动（在仓库根目录）：
    python demo/server.py
然后浏览器打开 http://127.0.0.1:8000

可用环境变量覆盖：DEMO_HOST / DEMO_PORT，以及 jev_extract 自身的
JEV_BACKEND / JEV_CANDIDATE_FILTER / JEV_ASSEMBLY 等（默认已置为 laya 最优配置）。
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
sys.path.insert(0, str(ROOT))

# laya 最优配置（见 docs/architecture.md「后端配置」）；setdefault 允许外部覆盖
os.environ.setdefault("JEV_BACKEND", "laya")
os.environ.setdefault("JEV_CANDIDATE_FILTER", "surname")
os.environ.setdefault("JEV_ASSEMBLY", "forced")
# HF 主站不可达的环境里强制走本地缓存，否则首次加载会卡在联网检查
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import jev_extract  # noqa: E402

HOST = os.getenv("DEMO_HOST", "127.0.0.1")
PORT = int(os.getenv("DEMO_PORT", "8000"))

# laya 后端非线程安全，且 demo 是单人使用，串行化即可
_LOCK = threading.Lock()
_WARMUP_MS = None


def warmup():
    """启动即加载模型并跑一条，避免首个请求等 ~4s 冷启动。"""
    t0 = time.perf_counter()
    with _LOCK:
        jev_extract.extract_entities("副驾是日产的小王")
    return (time.perf_counter() - t0) * 1000


class Handler(BaseHTTPRequestHandler):
    server_version = "jevCabinDemo/1.0"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                self._send(200, (DEMO_DIR / "index.html").read_bytes(),
                           "text/html; charset=utf-8")
            except OSError as e:
                self._send(500, json.dumps({"error": str(e)}))
        elif path == "/api/health":
            self._send(200, json.dumps({
                "ok": True,
                "backend": jev_extract.JEV_BACKEND,
                "model": jev_extract.LAYA_MODEL,
                "subfolder": jev_extract.LAYA_SUBFOLDER,
                "assembly": jev_extract.ASSEMBLY_MODE,
                "candidate_filter": jev_extract.CANDIDATE_FILTER,
                "warmup_ms": round(_WARMUP_MS, 1) if _WARMUP_MS else None,
            }, ensure_ascii=False))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/extract":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, json.dumps({"error": "invalid json body"}))
            return

        text = (payload.get("text") or "").strip()
        if not text:
            self._send(200, json.dumps({"persons": [], "elapsed_ms": 0.0,
                                        "perf": {}}, ensure_ascii=False))
            return

        t0 = time.perf_counter()
        try:
            with _LOCK:
                result = jev_extract.extract_entities(text)
                perf = dict(jev_extract.PERF)
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"},
                                       ensure_ascii=False))
            return
        elapsed = (time.perf_counter() - t0) * 1000

        self._send(200, json.dumps({
            "persons": result.get("persons", []),
            "elapsed_ms": round(elapsed, 1),
            "perf": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in perf.items()},
        }, ensure_ascii=False))

    def log_message(self, fmt, *args):  # 静音默认访问日志
        pass


def main():
    global _WARMUP_MS
    print(f"[demo] 后端={jev_extract.JEV_BACKEND} 组装={jev_extract.ASSEMBLY_MODE} "
          f"候选过滤={jev_extract.CANDIDATE_FILTER}")
    print("[demo] 正在加载 laya 模型（首次约 3-5s）...")
    _WARMUP_MS = warmup()
    print(f"[demo] 模型就绪，预热耗时 {_WARMUP_MS:.0f}ms")
    print(f"[demo] 打开 http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
