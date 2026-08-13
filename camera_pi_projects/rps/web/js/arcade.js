/* Shared Neon Arcade utilities — camera, UI states, API helpers. */
"use strict";

if (location.hostname === "127.0.0.1" || location.hostname === "0.0.0.0" || location.hostname === "localhost") {
  if (location.hostname !== "localhost") {
    location.replace("http://localhost:" + location.port + location.pathname + location.search);
  }
}

const Arcade = (() => {
  const $ = (id) => document.getElementById(id);

  async function startCamera(videoEl, opts) {
    opts = opts || {};
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return { ok: false, error: "Camera API not available — use Chrome/Edge on localhost" };
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: opts.width || 1280 },
          height: { ideal: opts.height || 720 },
          facingMode: "user",
        },
        audio: false,
      });
      videoEl.srcObject = stream;
      await videoEl.play();
      return { ok: true, stream };
    } catch (e) {
      return { ok: false, error: (e && e.message) || "Camera blocked or in use" };
    }
  }

  function stopCamera(stream) {
    if (!stream) return;
    stream.getTracks().forEach((t) => t.stop());
  }

  function setPhase(ids, phase) {
    const map = ids || {};
    ["landing", "loading", "playing", "error", "camError"].forEach((k) => {
      const el = map[k] || $(k);
      if (el) el.classList.toggle("hidden", k !== phase);
    });
  }

  async function api(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return res.json();
  }

  async function status() {
    const res = await fetch("/api/status");
    return res.json();
  }

  function showLoader(el, msg) {
    if (!el) return;
    el.textContent = msg || "Loading…";
    el.classList.remove("hidden");
  }

  return { $, startCamera, stopCamera, setPhase, api, status, showLoader };
})();
