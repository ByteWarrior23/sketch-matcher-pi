/* Virtual trackpad — control panel for the BACKEND trackpad.
 *
 * All hand detection and cursor control run in the Python server
 * (src/gesture_trackpad.py). Because it is not tied to this browser tab,
 * the real OS cursor keeps moving and clicking on ANY window you switch to
 * (Chrome, other websites, games) — even when this tab is in the background.
 *
 * This page only starts/stops the backend and shows its MJPEG preview.
 *
 * Gestures (index finger controls the cursor; the thumb is the mode switch):
 *   - index finger up (others down) ... move the cursor (1:1 aim)
 *   - index + middle spread ............ scroll up / down
 *   - thumb stretched out .............. freezes the cursor, enters button
 *     mode
 *   - curl index down (thumb out) ...... left-click (twice = double-click)
 *   - curl middle down (thumb out) ..... right-click
 *   - open palm ........................ release — no events, your real
 *     trackpad is always in charge
 */
"use strict";

const Trackpad = (() => {
  const $ = (id) => document.getElementById(id);

  const MODE_TEXT = { idle: "RELEASED", move: "MOVE", scroll: "SCROLL", buttons: "BUTTONS" };
  let pollTimer = null;

  async function api(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return res.json();
  }

  function setStatus(msg, level) {
    const el = $("statusMsg");
    const dot = $("statusDot");
    if (el) el.textContent = msg;
    if (dot) dot.className = "status-dot" + (level ? " " + level : "");
  }

  async function refreshStatus() {
    let st;
    try {
      const res = await fetch("/api/trackpad/status");
      st = await res.json();
    } catch (_) {
      setStatus("trackpad status unavailable", "err");
      return;
    }
    if (!st || !st.ok) return;

    const hudMove = $("hudMove");
    const hudConf = $("hudConf");
    if (hudMove) {
      hudMove.textContent = st.hand_seen
        ? (MODE_TEXT[st.mode] || String(st.mode || "idle").toUpperCase())
        : "NO HAND";
    }
    if (hudConf) {
      if (st.hand_seen) {
        const p = st.pose || {};
        if (p && Object.keys(p).length) {
          const t = st.thumb || {};
          hudConf.textContent =
            `scroll=${p.scroll ? "Y" : "N"} point=${p.point ? "Y" : "N"} ` +
            `open=${p.open ? "Y" : "N"} thumb=${p.thumb ? "Y" : "N"} ` +
            `(d45=${t.d45 || 0} palm=${t.d_palm || 0}) ` +
            `ts=${p.thumb_streak || 0} streak=${p.streak || 0}`;
        } else {
          hudConf.textContent = "hand seen — gesture to control";
        }
      } else {
        hudConf.textContent = "show your hand to the camera";
      }
    }

    const last = $("lastAction");
    if (last) last.textContent = st.last_action || "—";

    const dot = $("cursorDot");
    if (dot && st.cursor) {
      dot.style.left = (st.cursor.x * 100) + "%";
      dot.style.top = (st.cursor.y * 100) + "%";
    }

    setStatus(
      st.running
        ? "Backend running — switch to any window and gesture there"
        : "Stopped",
      st.running ? "ok" : ""
    );
    syncToggleLabel(st.running);
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(refreshStatus, 250);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function showPreview() {
    const img = $("camFeed");
    if (img) {
      img.src = "/api/trackpad/preview";
      img.style.display = "block";
    }
  }

  function hidePreview() {
    const img = $("camFeed");
    if (img) {
      img.removeAttribute("src");
      img.style.display = "none";
    }
  }

  async function toggleCamera() {
    let st = null;
    try {
      const res = await fetch("/api/trackpad/status");
      st = await res.json();
    } catch (_) { /* fall through */ }
    if (st && st.running) {
      await stopSession();
    } else {
      await startSession();
    }
  }

  function syncToggleLabel(running) {
    const tg = $("camToggleBtn");
    if (!tg) return;
    tg.textContent = running ? "Trackpad ON" : "Trackpad OFF";
    tg.classList.toggle("btn-primary", running);
    tg.classList.toggle("btn-ghost", !running);
  }

  async function startSession() {
    $("landing").classList.add("hidden");
    $("camError").classList.add("hidden");
    $("loading").classList.remove("hidden");

    const r = await api("/api/trackpad/start", {});
    if (r && r.ok) {
      $("loading").classList.add("hidden");
      $("hud").classList.remove("hidden");
      $("playPanel").classList.remove("hidden");
      showPreview();
      setStatus("Backend trackpad running — switch to any window and gesture there", "ok");
      startPolling();
      syncToggleLabel(true);
    } else {
      $("loading").classList.add("hidden");
      $("camError").classList.remove("hidden");
      const p = $("camError").querySelector("p");
      if (p) p.textContent = (r && r.error) || "Trackpad failed to start";
    }
  }

  async function stopSession() {
    await api("/api/trackpad/stop", {});
    stopPolling();
    hidePreview();
    $("playPanel").classList.add("hidden");
    $("hud").classList.add("hidden");
    $("landing").classList.remove("hidden");
    setStatus("Stopped", "");
    syncToggleLabel(false);
  }

  // If the backend is already running (e.g. page reload), restore the UI.
  async function resumeIfRunning() {
    let st;
    try {
      const res = await fetch("/api/trackpad/status");
      st = await res.json();
    } catch (_) {
      return;
    }
    if (st && st.ok && st.running) {
      $("landing").classList.add("hidden");
      $("hud").classList.remove("hidden");
      $("playPanel").classList.remove("hidden");
      showPreview();
      startPolling();
    }
  }

  function bind() {
    $("startBtn").addEventListener("click", () => {
      Sfx.start();
      startSession();
    });
    $("stopBtn").addEventListener("click", () => stopSession());
    $("retryBtn").addEventListener("click", () => startSession());

    const tog = $("camToggleBtn");
    if (tog) {
      tog.addEventListener("click", toggleCamera);
    }

    const rc = $("recenterBtn");
    if (rc) {
      rc.addEventListener("click", async () => {
        await api("/api/trackpad/recenter", {});
        setStatus("Cursor centered", "ok");
      });
    }

    resumeIfRunning();
  }

  return { bind };
})();
