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

    bridge = get_bridge()
    for game, action in [("subway", "left"), ("leveldevil", "jump")]:
        try:
            data = bridge.dispatch(game, action)
            if "ok" not in data:
                fails.append(f"dispatch {game}/{action} missing ok field")
        except Exception as e:
            fails.append(f"dispatch {game}/{action} -> {e}")

    httpd.shutdown()
    if fails:
        print("FAIL:")
        for f in fails:
            print(" ", f)
        sys.exit(1)
    print(f"OK: {len(ROUTES)} routes + /api/action (subway, leveldevil)")


if __name__ == "__main__":
    main()
