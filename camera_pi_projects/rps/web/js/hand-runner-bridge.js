/* Hand-swipe runner controller — BlueStacks runner games (Subway/Temple Run)
 * driven by moving your hand.
 *
 * Runs MediaPipe HandLandmarker DIRECTLY in the browser (GPU/WASM, full-res
 * frames, no network round-trip) for fast, accurate swipe detection — the
 * same model and approach used by the official mediapipe-samples web demos.
 * Only the final swipe action is POSTed to the server (/api/action).
 *
 * Anti-false-positive measures (matching MediaPipe's guidance):
 *   - only the highest-scoring hand with score >= 0.5 is used
 *   - a hand must persist for CONFIRM_FRAMES consecutive frames
 *   - swipes need peak displacement + a velocity gate (rejects drift/jitter)
 *   - EMA smoothing of the palm position
 */
"use strict";

const HandRunner = (() => {
  const $ = (id) => document.getElementById(id);
  const SWIPE_ACTIONS = { up: "jump", down: "duck", left: "left", right: "right" };
  const SWIPE_LABELS = { up: "SWIPE UP", down: "SWIPE DOWN", left: "SWIPE LEFT", right: "SWIPE RIGHT" };

  // ---- in-browser swipe tracker (ported from server _SwipeTracker) ----
  // Subway/Temple Run: each swipe is ONE discrete command (lane change), then
  // nothing until the next swipe. Short cooldown = fast consecutive moves.
  const WIN_MS = 450, MIN_MOVE = 0.05, MIN_SPAN_MS = 60, COOLDOWN_MS = 280,
        MAX_DROPOUT_MS = 400, CONFIRM_FRAMES = 1, MIN_SPEED = 0.2, EMA_ALPHA = 0.7;

  function makeTracker() {
    return {
      pts: [], streak: 0, ema: null, lastFire: -1e9, lastSeen: -1e9,
      reset() {
        this.pts = []; this.streak = 0; this.ema = null; this.lastFire = -1e9; this.lastSeen = -1e9;
      },
      tick(t) {
        if (this.pts.length && t - this.lastSeen > MAX_DROPOUT_MS) {
          this.pts = []; this.streak = 0; this.ema = null;
        }
        this.lastSeen = t;
      },
      update(x, y, t) {
        if (this.pts.length && t - this.lastSeen > MAX_DROPOUT_MS) {
          this.pts = []; this.streak = 0; this.ema = null;
        }
        this.lastSeen = t;
        this.streak += 1;
        if (this.streak < CONFIRM_FRAMES) return null;
        if (this.ema === null) this.ema = [x, y];
        else this.ema = [EMA_ALPHA * x + (1 - EMA_ALPHA) * this.ema[0],
                         EMA_ALPHA * y + (1 - EMA_ALPHA) * this.ema[1]];
        [x, y] = this.ema;
        this.pts.push([t, x, y]);
        this.pts = this.pts.filter((p) => t - p[0] <= WIN_MS);
        if (this.pts.length < 3) return null;
        const t0 = this.pts[0][0], x0 = this.pts[0][1], y0 = this.pts[0][2];
        if (t - t0 < MIN_SPAN_MS) return null;
        let dx = 0, dy = 0, bestT = t0;
        for (const [pt, px, py] of this.pts.slice(1)) {
          if (Math.abs(px - x0) > Math.abs(dx)) { dx = px - x0; bestT = pt; }
          if (Math.abs(py - y0) > Math.abs(dy)) { dy = py - y0; bestT = pt; }
        }
        if (Math.abs(dx) < MIN_MOVE && Math.abs(dy) < MIN_MOVE) return null;
        if (t - this.lastFire < COOLDOWN_MS) return null;
        const peak = Math.max(Math.abs(dx), Math.abs(dy));
        if (peak / ((bestT - t0) / 1000) < MIN_SPEED) return null;
        const swipe = Math.abs(dx) > Math.abs(dy)
          ? (dx > 0 ? "right" : "left")
          : (dy > 0 ? "down" : "up");
        this.lastFire = t;
        this.pts = [];
        return swipe;
      }
    };
  }
  const tracker = makeTracker();

  // ---- position-based (joystick) control for Level Devil ----
  // Side zones: far left/right edges move; the CENTER is neutral (stop).
  // Within a side, hand height splits the zone: bottom->mid = direction only,
  // mid->up = direction + jump (Space tapped repeatedly while held).
  const JOY_L_ON = 0.24, JOY_L_OFF = 0.45;
  const JOY_R_ON = 0.76, JOY_R_OFF = 0.55;
  const JUMP_ON = 0.42, JUMP_OFF = 0.50;
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

  const JOYSTICK_GAMES = new Set(["leveldevil"]);

  function onHand(results) {
    if (phase !== "playing") return;
    const hudMove = $("hudMove");
    const hudConf = $("hudConf");
    const last = $("lastSent");
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
      if (JOYSTICK_GAMES.has(gameId)) releaseKeys();
      else tracker.tick(performance.now());
      return;
    }
    const lm = hands[best];
    drawHand(lm);
    if (hudConf) hudConf.textContent = "hand seen (" + bestScore.toFixed(2) + ")";
    const [px, py] = palmCenter(lm);
    const now = performance.now();

    if (JOYSTICK_GAMES.has(gameId)) {
      const x = 1 - px;  // mirrored to match the preview
      if (joy.left) {
        if (x > JOY_L_OFF) { sendKey("left", "up"); joy.left = false; }
      } else if (x < JOY_L_ON) {
        sendKey("left", "down"); joy.left = true;
      }
      if (joy.right) {
        if (x < JOY_R_OFF) { sendKey("right", "up"); joy.right = false; }
      } else if (x > JOY_R_ON) {
        sendKey("right", "down"); joy.right = true;
      }
      // Height within the active side: raise the hand clearly above the
      // shoulder/head level to also jump (taps Space repeatedly while held).
      const onSide = joy.left || joy.right;
      if (onSide) {
        const wantJump = joy.jump ? py < JUMP_OFF : py < JUMP_ON;
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
        const wantJump = joy.jump ? py < JUMP_OFF : py < JUMP_ON;
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

    const swipe = tracker.update(1 - px, py, now);
    if (swipe && SWIPE_ACTIONS[swipe]) {
      const label = SWIPE_LABELS[swipe];
      if (hudMove) hudMove.textContent = label;
      Arcade.api("/api/action", { game: gameId, action: SWIPE_ACTIONS[swipe] })
        .then((res) => {
          if (last) {
            last.textContent = (res && res.ok)
              ? label + " SENT to game"
              : "send failed: " + ((res && res.error) || "no response");
          }
        })
        .catch(() => { if (last) last.textContent = "send failed: network error"; });
    } else if (hudMove) {
      hudMove.textContent = "READY — swipe your hand";
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
    tracker.reset();
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
