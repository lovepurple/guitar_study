#!/usr/bin/env python3
"""
CyberFret 本地启动器 & 同源 LLM API 代理（零外部依赖，兼容 Python 3.9+）
作用：
1. 在 http://127.0.0.1:8080 托管当前目录（MacBook/Windows 下麦克风与音高检测权限更稳定）
2. 将 /api/v1/* 反向代理到上游 OpenAI 兼容服务，解决浏览器 CORS 跨域与证书问题
"""

import http.server
import os
import ssl
import sys
import urllib.error
import urllib.request
import webbrowser

PORT = int(os.environ.get("CYBERFRET_PORT", "8080"))
UPSTREAM_BASE = os.environ.get("CYBERFRET_UPSTREAM", "https://sub2api.guangze-song.top").rstrip("/")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


class StudioHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self._proxy_request("POST")
        else:
            self.send_error(404, "Not Found")

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy_request("GET")
        elif self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/code_artifact.html")
            self.end_headers()
        else:
            super().do_GET()

    def _proxy_request(self, method: str):
        target_path = self.path[len("/api"):]  # e.g. /v1/chat/completions
        target_url = f"{UPSTREAM_BASE}{target_path}"
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else None

        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth

        req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=60) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(f'{{"error":{{"message":"Proxy error: {e}"}}}}'.encode("utf-8"))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), StudioHandler)
    url = f"http://127.0.0.1:{PORT}/code_artifact.html"
    print(f"[CyberFret] AI Guitar Studio started: {url}")
    print(f"[CyberFret] Upstream proxy: /api/* -> {UPSTREAM_BASE}/*")
    print("Press Ctrl+C to stop.")
    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止服务。")
