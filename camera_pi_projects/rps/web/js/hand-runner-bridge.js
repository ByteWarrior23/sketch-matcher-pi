/* Hand-zone runner controller — Subway Surfers and Level Devil.
 *
 * Runs MediaPipe HandLandmarker DIRECTLY in the browser (GPU/WASM, full-res
 * frames, no network round-trip). Hand POSITION drives both games — there is
 * no motion/swipe detection, so returning your hand to center never fires an
 * unwanted command. Only the final action is POSTed to the server.
 *
 * Reliability measures:
 *   - only the highest-scoring hand with score >= 0.5 is used
 *   - zones engage/release with hysteresis (no flicker at boundaries)
 *   - holding a zone repeats that action; center (rest) stops it
 */
"use strict";

const HandRunner = (() => {
  const $ = (id) => document.getElementById(id);

  // ---- shared zone thresholds (mirrored frame x) ----
  const Z_LEFT = { on: 0.24, off: 0.45 };
  const Z_RIGHT = { on: 0.76, off: 0.55 };
  const Z_UP = { on: 0.42, off: 0.50 };
  const Z_DOWN = { on: 0.62, off: 0.55 };

  // ---- swipe-based control for Subway Surfers (poki.com web) ----
  // The index fingertip is tracked; a quick flick fires one arrow key. A swipe
  // only counts when it starts from the neutral center zone, so returning your
  // hand to center after an up swipe never fires the down (roll) command.
  const SWIPE_MIN_MOVE = 0.14, SWIPE_MAX_MS = 400, SWIPE_COOLDOWN_MS = 250;
  const NEUTRAL = { x0: 0.30, x1: 0.70, y0: 0.30, y1: 0.70 };
  const SUBWAY_KEYS = { left: "left", right: "right", up: "up", down: "down" };
  const sw = { anchor: null, locked: false, lastFire: -1e9 };

  function swReset() {
    sw.anchor = null;
    sw.locked = false;
  }

  function subwaySwipe(fx, fy, t) {
    const inNeutral = fx > NEUTRAL.x0 && fx < NEUTRAL.x1 &&
                      fy > NEUTRAL.y0 && fy < NEUTRAL.y1;
    if (sw.locked && inNeutral) {
      sw.locked = false;
      sw.anchor = { x: fx, y: fy, t };
      return null;
    }
    if (inNeutral) {
      sw.anchor = { x: fx, y: fy, t };
      return null;
    }
    if (sw.locked || !sw.anchor) return null;
    if (t - sw.anchor.t > SWIPE_MAX_MS) return null;
    if (t - sw.lastFire < SWIPE_COOLDOWN_MS) return null;
    const dx = fx - sw.anchor.x, dy = fy - sw.anchor.y;
    const adx = Math.abs(dx), ady = Math.abs(dy);
    if (Math.max(adx, ady) < SWIPE_MIN_MOVE) return null;
    sw.locked = true;
    sw.lastFire = t;
    if (adx > ady * 1.3) return dx > 0 ? "right" : "left";
    if (ady > adx * 1.3) return dy > 0 ? "down" : "up";
    sw.locked = false;
    return null;
  }

  function subwayTap(zone) {
    const key = SUBWAY_KEYS[zone];
    Arcade.api("/api/key", { game: "subway", key, state: "tap" }).then((res) => {
      const last = $("lastSent");
      if (last) {
        last.textContent = (res && res.ok)
          ? (zone.toUpperCase() + " SENT")
          : "key failed: " + ((res && res.error) || "no response");
      }
    }).catch(() => { const last = $("lastSent"); if (last) last.textContent = "key failed: network error"; });
  }

  // ---- position-based (joystick) control for Level Devil ----
  // Side zones: far left/right edges move; the CENTER is neutral (stop).
  // Within a side, hand height splits the zone: natural height = direction
  // only, raised clearly = direction + jump (Space tapped repeatedly).
  const JUMP_REPEAT_MS = 280;
  const joy = { left: false, right: false, jump: false, repeat: false };
  let jumpTimer = null;

  // Jump repeat (side + raised hand): taps Space repeatedly so the character
  // keeps jumping while running. Plain center jump is a simple hold instead.
  function startJumpRepeat() {
    if (jumpTimer) { clearTimeout(jumpTimer); jumpTimer = null; }
    joy.repeat = true;
    const tap = () => {
      if (!joy.jump) return;
      sendKey("jump", "up");
      sendKey("jump", "down");
      jumpTimer = setTimeout(tap, JUMP_REPEAT_MS);
    };
    jumpTimer = setTimeout(tap, JUMP_REPEAT_MS);
  }

  function stopJumpRepeat() {
    if (jumpTimer) { clearTimeout(jumpTimer); jumpTimer = null; }
    joy.repeat = false;
    if (joy.jump) { sendKey("jump", "up"); joy.jump = false; }
  }

  function sendKey(key, state) {
    Arcade.api("/api/key", { game: gameId, key, state }).then((res) => {
      const last = $("lastSent");
      if (last) {
        last.textContent = (res && res.ok)
          ? (key.toUpperCase() + " " + (state === "down" ? "HOLD" : "release"))
          : "key failed: " + ((res && res.error) || "no response");
      }
    }).catch(() => {
      const last = $("lastSent");
      if (last) last.textContent = "key failed: network error";
    });
  }

  function releaseKeys() {
    stopJumpRepeat();
    swReset();
    for (const k of ["left", "right", "jump"]) {
      if (joy[k]) sendKey(k, "up");
      joy[k] = false;
    }
  }

  // ---- index-finger double-tap = click (Level Devil) ----
  // Uses the index fingertip (landmark 8) position RELATIVE to the palm
  // center, so raising/lowering the whole arm never looks like a tap — only
  // curling the index finger down then up does. Two taps within 500 ms fire.
  const TAP = { base: null, down: false, lastTap: -1e9 };
  function updateTap(relY8, t) {
    if (TAP.base === null) TAP.base = relY8;
    TAP.base += 0.05 * (relY8 - TAP.base);
    if (!TAP.down && relY8 - TAP.base > 0.12) {
      TAP.down = true;
    } else if (TAP.down && relY8 - TAP.base < 0.05) {
      TAP.down = false;
      if (t - TAP.lastTap < 500) {
        sendKey("click", "tap");
        TAP.lastTap = -1e9;
      } else {
        TAP.lastTap = t;
      }
    }
  }

  let phase = "idle";
  let gameId = "";
  let landmarker = null;
  let rafId = 0;
  let preview = null;
  let pctx = null;
  let lastVideoTime = -1;

  function setStatus(msg, level) {
    const el = $("statusText");
    const dot = $("statusDot");
    if (el) el.textContent = msg;
    if (dot) dot.className = "status-dot" + (level ? " " + level : "");
  }

  function setStep(msg) {
    const el = $("adbStatus");
    if (el) el.textContent = msg;
  }

  async function refresh() {
    try {
      const st = await Arcade.status();
      setStep(st.message || "Checking…");
      if (st.connected) setStatus("Connected — play in BlueStacks window", "ok");
      else if (st.bluestacks_running) setStatus("BlueStacks running — connect ADB", "warn");
      else if (st.bluestacks_installed) setStatus("Starting BlueStacks…", "warn");
      else setStatus("Installing BlueStacks automatically…", "warn");
      return st;
    } catch (_) {
      setStatus("Server offline", "err");
      return null;
    }
  }

  async function startBlueStacks() {
    setStatus("Starting BlueStacks…", "warn");
    const r = await Arcade.api("/api/bluestacks/start", {});
    await refresh();
    setStatus(r.ok ? "BlueStacks starting…" : (r.error || "Failed"), r.ok ? "warn" : "err");
    return r;
  }

  async function connect() {
    setStatus("Connecting ADB…", "warn");
    const r = await Arcade.api("/api/connect", {});
    await refresh();
    setStatus(r.ok ? "Emulator connected" : (r.error || "Failed"), r.ok ? "ok" : "err");
    return r;
  }

  async function prepare(id) {
    setStatus("Preparing real game (auto setup)…", "warn");
    setStep("Installing/connecting BlueStacks, loading game — first run may take several minutes…");
    const r = await Arcade.api("/api/prepare", { game: id });
    await refresh();
    if (r.ok) {
      setStatus("Real game launched — switch to BlueStacks window", "ok");
      setStep("Play in the BlueStacks window. Hand swipes here send game swipes.");
    } else {
      setStatus(r.error || "Setup failed", "err");
      setStep(r.error || "See steps below");
    }
    return r;
  }

  // ---------- hand landmarking (in-browser MediaPipe) ----------
  const PALM_IDX = [0, 5, 9, 13, 17];
  const CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
  ];

  function palmCenter(landmarks) {
    let sx = 0, sy = 0;
    for (const i of PALM_IDX) { sx += landmarks[i].x; sy += landmarks[i].y; }
    return [sx / PALM_IDX.length, sy / PALM_IDX.length];
  }

  function drawHand(landmarks) {
    if (!pctx || !preview) return;
    const W = preview.width, H = preview.height;
    pctx.clearRect(0, 0, W, H);
    if (!landmarks) return;
    const X = (i) => (1 - landmarks[i].x) * W;
    const Y = (i) => landmarks[i].y * H;
    pctx.strokeStyle = "#39ff14";
    pctx.lineWidth = 2;
    pctx.lineJoin = "round";
    for (const [a, b] of CONNECTIONS) {
      pctx.beginPath();
      pctx.moveTo(X(a), Y(a));
      pctx.lineTo(X(b), Y(b));
      pctx.stroke();
    }
    const [px, py] = palmCenter(landmarks);
    pctx.fillStyle = "#ffd400";
    pctx.beginPath();
    pctx.arc((1 - px) * W, py * H, 5, 0, Math.PI * 2);
    pctx.fill();
  }

  async function initModel() {
    const { FilesetResolver, HandLandmarker } = await import("../vendor/vision_bundle.mjs");
    const vision = await FilesetResolver.forVisionTasks(location.origin + "/vendor/wasm/");
    const opts = {
      baseOptions: {
        modelAssetPath: location.origin + "/hand_model.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    };
    try {
      return await HandLandmarker.createFromOptions(vision, opts);
    } catch (_) {
      opts.baseOptions.delegate = "CPU";
      return await HandLandmarker.createFromOptions(vision, opts);
    }
  }

  function onHand(results) {
    if (phase !== "playing") return;
    const hudMove = $("hudMove");
    const hudConf = $("hudConf");
    const hands = results.landmarks || [];
    const scores = results.handedness || [];
    let best = -1, bestScore = 0;
    for (let i = 0; i < hands.length; i++) {
      const sc = (scores[i] && scores[i][0]) ? scores[i][0].score : 0;
      if (sc >= 0.5 && sc > bestScore) { bestScore = sc; best = i; }
    }
    if (best < 0) {
      if (hudMove) hudMove.textContent = "NO HAND";
      if (hudConf) hudConf.textContent = "move your hand in the camera";
      drawHand(null);
      releaseKeys();
      return;
    }
    const lm = hands[best];
    drawHand(lm);
    if (hudConf) hudConf.textContent = "hand seen (" + bestScore.toFixed(2) + ")";
    const [px, py] = palmCenter(lm);
    const now = performance.now();

    if (gameId === "leveldevil") {
      const x = 1 - px;  // mirrored to match the preview
      if (joy.left) {
        if (x > Z_LEFT.off) { sendKey("left", "up"); joy.left = false; }
      } else if (x < Z_LEFT.on) {
        sendKey("left", "down"); joy.left = true;
      }
      if (joy.right) {
        if (x < Z_RIGHT.off) { sendKey("right", "up"); joy.right = false; }
      } else if (x > Z_RIGHT.on) {
        sendKey("right", "down"); joy.right = true;
      }
      // Height within the active side: raise the hand clearly above the
      // shoulder/head level to also jump (taps Space repeatedly while held).
      const onSide = joy.left || joy.right;
      if (onSide) {
        const wantJump = joy.jump ? py < Z_UP.off : py < Z_UP.on;
        if (wantJump && !joy.jump) {
          joy.jump = true;
          joy.repeat = false;
          sendKey("jump", "down");
          startJumpRepeat();
        } else if (wantJump && joy.jump && !joy.repeat) {
          startJumpRepeat();
        } else if (!wantJump && joy.jump) {
          stopJumpRepeat();
        }
      } else {
        // Center: raised hand = plain jump hold (no direction).
        stopJumpRepeat();
        const wantJump = joy.jump ? py < Z_UP.off : py < Z_UP.on;
        if (wantJump !== joy.jump) {
          sendKey("jump", wantJump ? "down" : "up");
          joy.jump = wantJump;
        }
      }
      // Index-finger double-tap = click.
      updateTap(lm[8].y - py, now);
      if (hudMove) {
        const parts = [];
        if (joy.left) parts.push("LEFT");
        if (joy.right) parts.push("RIGHT");
        if (joy.jump) parts.push("JUMP");
        hudMove.textContent = parts.length ? parts.join("+") : "CENTER";
      }
      return;
    }

    // Subway Surfers (poki.com web): flick the index finger to send an arrow
    // key. Flicks only fire from the center, so lowering your hand back to
    // center never triggers roll.
    const fx = 1 - lm[8].x;  // mirrored index fingertip
    const swipe = subwaySwipe(fx, lm[8].y, now);
    if (swipe) subwayTap(swipe);
    if (hudMove) {
      hudMove.textContent = sw.locked
        ? (swipe ? swipe.toUpperCase() + " SENT" : "return to center")
        : "flick to move";
    }
  }

  function loop() {
    const video = $("video");
    if (landmarker && video && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      try {
        onHand(landmarker.detectForVideo(video, performance.now()));
      } catch (_) { /* skip frame */ }
    }
    rafId = requestAnimationFrame(loop);
  }

  function stopLoop() {
    cancelAnimationFrame(rafId);
    releaseKeys();
  }

  window.addEventListener("blur", releaseKeys);

  async function startSession(id) {
    gameId = id;
    phase = "loading";
    $("landing").classList.add("hidden");
    $("loading").classList.remove("hidden");
    $("camError").classList.add("hidden");

    preview = $("preview");
    if (preview) {
      pctx = preview.getContext("2d");
      preview.style.display = "block";
    }

    const cam = await Arcade.startCamera($("video"), { width: 640, height: 480 });
    if (!cam.ok) {
      $("loading").classList.add("hidden");
      $("camError").classList.remove("hidden");
      return;
    }

    $("loading").classList.add("hidden");
    $("hud").classList.remove("hidden");
    $("playPanel").classList.remove("hidden");
    const hudConf = $("hudConf");
    const hudMove = $("hudMove");
    if (hudConf) hudConf.textContent = "loading hand model…";
    if (hudMove) hudMove.textContent = "…";
    try {
      landmarker = await initModel();
    } catch (e) {
      if (hudConf) hudConf.textContent = "model load failed: " + e.message;
      phase = "idle";
      return;
    }
    phase = "playing";

    rafId = requestAnimationFrame(loop);
    prepare(id).catch(() => {});
  }

  function bind(id) {
    gameId = id;
    $("startBtn").addEventListener("click", () => { Sfx.start(); startSession(id); });
    $("retryBtn").addEventListener("click", () => location.reload());
    if ($("reloadBtn")) $("reloadBtn").addEventListener("click", () => location.reload());
    if ($("bsBtn")) $("bsBtn").addEventListener("click", startBlueStacks);
    if ($("connectBtn")) $("connectBtn").addEventListener("click", connect);
    if ($("launchBtn")) $("launchBtn").addEventListener("click", () => prepare(id));
    refresh();
    setInterval(refresh, 10000);
  }

  return { bind, refresh, prepare, stopClassify: stopLoop };
})();
