"""Route and API smoke tests for the arcade server."""
import json
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from input_bridge import get_bridge  # noqa: E402
from server import Handler  # noqa: E402

ROUTES = [
    "/", "/subway", "/leveldevil",
    "/css/style.css", "/js/arcade.js", "/js/hand-runner-bridge.js",
    "/health", "/api/status",
]


def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def post_json(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def main():
    port = 8765
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.6)

    base = f"http://127.0.0.1:{port}"
    fails = []
    for path in ROUTES:
        try:
            code, _, _ = fetch(base + path)
            if code != 200:
                fails.append(f"{path} -> HTTP {code}")
        except Exception as e:
            fails.append(f"{path} -> {e}")

    code, data = post_json(base + "/api/key", {"game": "leveldevil", "key": "left", "state": "tap"})
    if code != 200 or not data.get("ok"):
        fails.append(f"/api/key leveldevil -> {data}")

    code, data = post_json(base + "/api/key", {"game": "subway", "key": "up", "state": "down"})
    if code != 200 or not data.get("ok"):
        fails.append(f"/api/key subway -> {data}")

    code, data = post_json(base + "/api/key", {"game": "subway", "key": "bogus", "state": "tap"})
    if code != 200 or data.get("ok"):
        fails.append(f"/api/key bogus key should fail -> {data}")

    code, data = post_json(base + "/api/prepare", {"game": "leveldevil"})
    if code != 200 or "ok" not in data:
        fails.append(f"/api/prepare leveldevil -> {data}")

    httpd.shutdown()
    if fails:
        print("FAIL:")
        for f in fails:
            print(" ", f)
        sys.exit(1)
    print(f"OK: {len(ROUTES)} routes + /api/key + /api/prepare")


if __name__ == "__main__":
    main()
