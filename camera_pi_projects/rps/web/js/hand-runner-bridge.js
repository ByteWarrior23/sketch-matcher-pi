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

  // ---- index-finger joystick for Subway Surfers (poki.com web) ----
  // Keep the hand still and steer with the index finger. The fingertip's
  // resting spot is tracked as the center; moving the fingertip clear of the
  // center fires that direction's key once, and returning to center re-arms.
  const JOY_DEADZONE = 0.07, JOY_REARM_MS = 60;
  const SUBWAY_KEYS = { left: "left", right: "right", up: "up", down: "down" };
  const jz = { cx: 0.5, cy: 0.5, armed: true, lastFire: -1e9 };

  function swReset() {
    jz.armed = true;
  }

  function subwayJoystick(fx, fy, t) {
    const dx = fx - jz.cx, dy = fy - jz.cy;
    const adx = Math.abs(dx), ady = Math.abs(dy);
    if (adx < JOY_DEADZONE && ady < JOY_DEADZONE) {
      // Idle: re-anchor the center to the finger's resting spot and re-arm.
      // The center is FROZEN while a move is in progress, so it never drifts
      // toward the finger mid-move (which caused random double/triple fires).
      jz.cx += 0.1 * (fx - jz.cx);
      jz.cy += 0.1 * (fy - jz.cy);
      jz.armed = true;
      return null;
    }
    if (!jz.armed || t - jz.lastFire < JOY_REARM_MS) return null;
    let dir = null;
    if (adx > ady * 1.5) dir = dx > 0 ? "right" : "left";
    else if (ady > adx * 1.5) dir = dy > 0 ? "down" : "up";
    if (!dir) return null;
    jz.armed = false;
    jz.lastFire = t;
    return dir;
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
  // Whole-hand (palm) position drives the game exactly like the original
  // working build: far left/right edges run, center stops, and hand height
  // jumps. The arrow key is HELD while the hand stays in a side zone, so the
  // character runs at the game's own full speed.
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

  // Key events must reach the server IN ORDER (a "down" must be processed
  // before its "up"), or the game gets stuck/erratically moving keys. Chain
  // every POST so each is sent only after the previous one completed.
  let keyChain = Promise.resolve();
  function sendKey(key, state) {
    const p = keyChain.then(() =>
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
      })
    );
    keyChain = p.catch(() => {});
    return p;
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

  async function prepare(id) {
    setStatus("Opening game…", "warn");
    const r = await Arcade.api("/api/prepare", { game: id });
    if (r.ok) {
      setStatus("Game launched — click inside it once, then steer", "ok");
    } else {
      setStatus(r.error || "Launch failed", "err");
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
    const fx = (1 - landmarks[8].x) * W, fy = landmarks[8].y * H;
    pctx.fillStyle = "#00e5ff";
    pctx.beginPath();
    pctx.arc(fx, fy, 4, 0, Math.PI * 2);
    pctx.fill();
    if (gameId === "subway") {
      const cx = (1 - jz.cx) * W, cy = jz.cy * H;
      pctx.strokeStyle = "rgba(255,255,255,0.9)";
      pctx.lineWidth = 1.5;
      pctx.beginPath();
      pctx.moveTo(cx - 14, cy); pctx.lineTo(cx + 14, cy);
      pctx.moveTo(cx, cy - 14); pctx.lineTo(cx, cy + 14);
      pctx.stroke();
    }
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
      // Whole-hand control (as in the original working build): mirrored palm
      // x drives left/right, palm height (py) drives jump. Keys are HELD
      // while the hand stays in a zone, center = release/stop.
      const x = 1 - px;   // mirrored to match the preview
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

    // Subway Surfers (poki.com web): steer with the index fingertip. Its rest
    // spot is the center; moving it clear of the center fires an arrow key.
    const fx = 1 - lm[8].x;  // mirrored index fingertip
    const move = subwayJoystick(fx, yFlip ? 1 - lm[8].y : lm[8].y, now);
    if (move) subwayTap(move);
    if (hudMove) {
      hudMove.textContent = !jz.armed
        ? (move ? move.toUpperCase() + " SENT" : "return index to center")
        : "move index";
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

  let yFlip = localStorage.getItem("arcade_yflip2") === "1";

  function bind(id) {
    gameId = id;
    $("startBtn").addEventListener("click", () => { Sfx.start(); startSession(id); });
    $("retryBtn").addEventListener("click", () => location.reload());
    if ($("reloadBtn")) $("reloadBtn").addEventListener("click", () => location.reload());
    if ($("launchBtn")) $("launchBtn").addEventListener("click", () => prepare(id));
    const yBtn = $("yFlipBtn");
    if (yBtn) {
      yBtn.checked = yFlip;
      yBtn.addEventListener("change", () => {
        yFlip = yBtn.checked;
        localStorage.setItem("arcade_yflip2", yFlip ? "1" : "0");
      });
    }
  }

  return { bind, prepare, stopClassify: stopLoop };
})();
