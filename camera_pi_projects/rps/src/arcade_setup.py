"""One-click Neon Arcade setup — BlueStacks, ADB, game APKs."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APK_DIR = ROOT / "tools" / "apks"
CONF_PATHS = [
    Path(r"C:\ProgramData\BlueStacks_nxt\bluestacks.conf"),
    Path(r"C:\ProgramData\BlueStacks\bluestacks.conf"),
]

GAME_APKS = {
    "subway": {
        "package": "com.kiloo.subwaysurf",
        "filename": "subway.apk",
        "urls": [
            "https://d.apkpure.com/b/APK/com.kiloo.subwaysurf?version=latest",
            "https://d.apkpure.com/b/XAPK/com.kiloo.subwaysurf?version=latest",
        ],
    },
}

_APK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 10; SM-G988N) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
    ),
    "Referer": "https://apkpure.com/",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
}


def _run(cmd: list[str], timeout: float = 600.0) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()
    except Exception as e:
        return 1, str(e)


def find_bluestacks() -> Path | None:
    for p in [
        Path(r"C:\Program Files\BlueStacks_nxt\HD-Player.exe"),
        Path(r"C:\Program Files\BlueStacks\Bluestacks.exe"),
    ]:
        if p.exists():
            return p
    return None


def bluestacks_running() -> bool:
    code, out = _run(["tasklist", "/FI", "IMAGENAME eq HD-Player.exe"], timeout=5)
    return code == 0 and "HD-Player.exe" in out


INSTALLER = ROOT / "tools" / "BlueStacksFullInstaller.exe"
INSTALLER_URL = (
    "https://ak-build.bluestacks.com/public/app-player/windows/nxt/"
    "5.22.166.1003/e0cbf0a49445273dc3c94ced970fd7d4/FullInstaller/x64/"
    "BlueStacksFullInstaller_5.22.166.1003_amd64_native.exe"
)


def _download_installer() -> dict:
    if INSTALLER.exists() and INSTALLER.stat().st_size > 50_000_000:
        return {"ok": True, "path": str(INSTALLER), "note": "cached"}
    INSTALLER.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(INSTALLER_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = resp.read()
        if len(data) < 50_000_000:
            return {"ok": False, "error": f"installer too small ({len(data)} bytes)"}
        INSTALLER.write_bytes(data)
        return {"ok": True, "path": str(INSTALLER), "bytes": len(data)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def install_bluestacks() -> dict:
    if find_bluestacks():
        return {"ok": True, "note": "already installed"}
    if INSTALLER.exists() and INSTALLER.stat().st_size > 50_000_000:
        code, out = _run([str(INSTALLER), "-s"], timeout=900)
        if code == 0 or find_bluestacks():
            for _ in range(90):
                if find_bluestacks():
                    return {"ok": True, "note": "installed via local installer", "log": out}
                time.sleep(5)
    dl = _download_installer()
    if dl.get("ok"):
        code, out = _run([str(INSTALLER), "-s"], timeout=900)
        if code == 0 or find_bluestacks():
            for _ in range(90):
                if find_bluestacks():
                    return {"ok": True, "note": "installed via downloaded installer", "log": out}
                time.sleep(5)
    winget = shutil.which("winget")
    if winget:
        code, out = _run([
            winget, "install", "--id", "BlueStack.BlueStacks", "-e",
            "--accept-package-agreements", "--accept-source-agreements", "--silent",
        ], timeout=900)
        if code == 0 or "already installed" in out.lower():
            for _ in range(60):
                if find_bluestacks():
                    return {"ok": True, "note": "installed via winget"}
                time.sleep(5)
    return {"ok": False, "error": dl.get("error") or "BlueStacks install did not finish in time"}


def _conf_path() -> Path | None:
    for p in CONF_PATHS:
        if p.exists():
            return p
    return None


def _read_conf() -> dict[str, str]:
    path = _conf_path()
    if not path:
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        key, val = line.split("=", 1)
        data[key.strip()] = val.strip().strip('"')
    return data


def enable_adb_in_config() -> dict:
    path = _conf_path()
    if not path:
        return {"ok": False, "error": "bluestacks.conf not found yet"}
    text = path.read_text(encoding="utf-8", errors="ignore")
    changed = False
    for key, val in [
        ("bst.enable_adb_access", "1"),
        ("bst.enable_adb_remote_access", "1"),
    ]:
        pat = rf'^({re.escape(key)}=)"[^"]*"'
        if re.search(pat, text, flags=re.M):
            new = re.sub(pat, rf'\1"{val}"', text, count=1, flags=re.M)
            if new != text:
                text = new
                changed = True
        else:
            text += f'\n{key}="{val}"\n'
            changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return {"ok": True, "path": str(path), "changed": changed}


def adb_ports_from_config() -> list[str]:
    conf = _read_conf()
    ports: set[str] = set()
    for key, val in conf.items():
        if "adb_port" in key and val.isdigit():
            ports.add(val)
    if not ports:
        ports = {"5555", "5556"}
    return [f"127.0.0.1:{p}" for p in sorted(ports)]


def _adb_bin() -> str | None:
    bundled = ROOT / "tools" / "platform-tools" / "adb.exe"
    if bundled.exists():
        return str(bundled)
    return shutil.which("adb")


def _package_installed(adb: str, package: str) -> bool:
    code, out = _run([adb, "shell", "pm", "path", package], timeout=8)
    return code == 0 and "package:" in out


def _download_apk(urls: str | list[str], dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return {"ok": True, "path": str(dest), "note": "cached"}
    if isinstance(urls, str):
        urls = [urls]
    errors = []
    for url in urls:
        req = urllib.request.Request(url, headers=_APK_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            if len(data) < 500_000:
                errors.append(f"{url}: too small ({len(data)} bytes)")
                continue
            dest.write_bytes(data)
            return {"ok": True, "path": str(dest), "bytes": len(data), "url": url}
        except Exception as e:
            errors.append(f"{url}: {e}")
    return {"ok": False, "error": "; ".join(errors) or "all download URLs failed"}


def install_game_apk(game: str, adb: str | None = None) -> dict:
    if game not in GAME_APKS:
        return {"ok": False, "error": f"unknown game {game}"}
    adb = adb or _adb_bin()
    if not adb:
        return {"ok": False, "error": "adb missing"}
    meta = GAME_APKS[game]
    for pkg in meta.get("packages", [meta["package"]]):
        if _package_installed(adb, pkg):
            return {"ok": True, "note": "already installed", "package": pkg}
    dest = APK_DIR / meta["filename"]
    urls = meta.get("urls") or [meta.get("url", "")]
    dl = _download_apk(urls, dest)
    if not dl.get("ok"):
        return dl
    code, out = _run([adb, "install", "-r", str(dest)], timeout=180)
    if code == 0 and "Success" in out:
        return {"ok": True, "package": meta["package"], "apk": str(dest)}
    return {"ok": False, "error": out or f"adb install failed ({code})"}


def start_bluestacks() -> dict:
    exe = find_bluestacks()
    if not exe:
        return {"ok": False, "error": "BlueStacks not installed"}
    if bluestacks_running():
        return {"ok": True, "note": "already running"}
    subprocess.Popen(
        [str(exe)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(exe.parent),
    )
    return {"ok": True, "path": str(exe)}


def connect_adb(retries: int = 8) -> dict:
    adb = _adb_bin()
    if not adb:
        return {"ok": False, "error": "adb missing"}
    logs = []
    ports = adb_ports_from_config()
    for attempt in range(retries):
        for port in ports + [
            "127.0.0.1:62001", "127.0.0.1:62025",
            "127.0.0.1:62026", "127.0.0.1:62027",
        ]:
            code, out = _run([adb, "connect", port], timeout=4)
            logs.append(f"{port}: {out or code}")
        code, out = _run([adb, "devices"], timeout=5)
        if code == 0 and any("\tdevice" in ln for ln in out.splitlines()[1:]):
            return {"ok": True, "devices": out, "log": logs}
        time.sleep(3)
    return {"ok": False, "error": "ADB not connected", "log": logs}


def full_setup(games: list[str] | None = None) -> dict:
    """Install BlueStacks, enable ADB, connect, sideload games."""
    games = games or ["subway"]
    steps: list[dict] = []

    inst = install_bluestacks()
    steps.append({"step": "install_bluestacks", **inst})
    if not inst.get("ok"):
        return {"ok": False, "steps": steps, "error": inst.get("error")}

    start = start_bluestacks()
    steps.append({"step": "start_bluestacks", **start})
    if not start.get("ok"):
        return {"ok": False, "steps": steps, "error": start.get("error")}

    if not bluestacks_running():
        time.sleep(25)

    adb_cfg = enable_adb_in_config()
    steps.append({"step": "enable_adb", **adb_cfg})
    if adb_cfg.get("changed"):
        _run(["taskkill", "/IM", "HD-Player.exe", "/F"], timeout=10)
        time.sleep(3)
        start = start_bluestacks()
        steps.append({"step": "restart_bluestacks", **start})
        time.sleep(30)

    conn = connect_adb(retries=10)
    steps.append({"step": "connect_adb", **conn})
    if not conn.get("ok"):
        return {"ok": False, "steps": steps, "error": conn.get("error")}

    adb = _adb_bin()
    apk_ok = True
    for game in games:
        apk = install_game_apk(game, adb)
        steps.append({"step": f"install_{game}", **apk})
        if not apk.get("ok"):
            apk_ok = False

    return {"ok": apk_ok, "steps": steps}


def main():
    import json
    print("Neon Arcade setup — installing BlueStacks, ADB, games…")
    result = full_setup()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
