/* Rock Paper Scissors — browser game logic.
 * Camera -> canvas -> JPEG -> POST /classify -> server runs the int8 TFLite model.
 */
"use strict";

const MOVES = { rock: "\u270A", paper: "\u270B", scissors: "\u270C\uFE0F" };
const WIN = { rock: "scissors", paper: "rock", scissors: "paper" }; // WIN[a] = what a beats
const MIN_CONF = 0.7;
const HOLD = 5;           // consecutive agreeing frames to register a move
const CLASSIFY_MS = 150;  // ~6.6 fps

const $ = (id) => document.getElementById(id);
const video = $("video");
const preview = $("preview");
const pctx = preview.getContext("2d");

const state = {
  phase: "idle",       // idle | countdown | playing | revealed
  countdown: 0,
  timer: null,
  pending: null,
  holdCount: 0,
  history: [],         // last player moves (for counter strategy)
  score: { you: 0, pi: 0, draw: 0 },
  round: 0,
  camera: null,
  classifyTimer: null,
};

/* ---------- camera ---------- */
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("camError").classList.remove("hidden");
    return false;
  }
  try {
    state.camera = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = state.camera;
    await video.play();
    return true;
  } catch (e) {
    $("camError").classList.remove("hidden");
    return false;
  }
}

/* ---------- overlay ---------- */
function showOverlay(big, sub) {
  $("overlayBig").textContent = big;
  $("overlaySub").textContent = sub || "";
  $("overlay").classList.remove("hidden");
}
function hideOverlay() {
  $("overlay").classList.add("hidden");
}

/* ---------- classify a frame ---------- */
async function classifyNow() {
  if (video.readyState < 2) return;
  // center-crop to a square so the hand fills the frame like training data
  const vw = video.videoWidth, vh = video.videoHeight;
  const size = Math.min(vw, vh);
  const sx = (vw - size) / 2, sy = (vh - size) / 2;
  pctx.clearRect(0, 0, preview.width, preview.height);
  pctx.drawImage(video, sx, sy, size, size, 0, 0, 224, 224);

  preview.toBlob(async (blob) => {
    try {
      const res = await fetch("/classify", { method: "POST", body: blob });
      if (!res.ok) return;
      const r = await res.json();
      onPrediction(r);
    } catch (_) { /* server restarting */ }
  }, "image/jpeg", 0.85);
}

/* ---------- prediction -> game ---------- */
function onPrediction(r) {
  if (state.phase !== "playing") return;

  $("hudMove").textContent = MOVES[r.label] || "?";
  $("hudConf").textContent = `${Math.round(r.conf * 100)}%`;

  if (r.conf >= MIN_CONF) {
    if (state.pending === r.label) {
      state.holdCount += 1;
    } else {
      state.pending = r.label;
      state.holdCount = 1;
    }
    if (state.holdCount >= HOLD) {
      registerMove(r.label);
    }
  } else {
    state.pending = null;
    state.holdCount = 0;
    $("hudMove").textContent = "?";
    $("hudConf").textContent = "no gesture";
  }
}

function piMove() {
  if (state.history.length === 0 || Math.random() < 0.25) {
    return ["rock", "paper", "scissors"][Math.floor(Math.random() * 3)];
  }
  // counter: beat the player's most common recent move
  const counts = {};
  state.history.forEach((m) => (counts[m] = (counts[m] || 0) + 1));
  const most = Object.keys(counts).reduce((a, b) => (counts[a] > counts[b] ? a : b));
  return WIN[most];
}

function registerMove(playerMove) {
  state.phase = "revealed";
  clearInterval(state.classifyTimer);
  $("hud").classList.add("hidden");

  const pi = piMove();
  state.history.push(playerMove);
  if (state.history.length > 4) state.history.shift();
  state.round += 1;

  let outcome;
  if (playerMove === pi) outcome = "draw";
  else if (WIN[playerMove] === pi) outcome = "you";
  else outcome = "pi";
  state.score[outcome] += 1;

  // a beat of silence for dramatic pause, then reveal
  showOverlay("\u2026", "Machine is thinking");
  setTimeout(() => {
    hideOverlay();
    showResult(playerMove, pi, outcome);
  }, 700);
}

/* ---------- result UI ---------- */
function showResult(you, pi, outcome) {
  $("youMove").textContent = MOVES[you];
  $("piMove").textContent = MOVES[pi];

  const texts = { you: "YOU WIN", pi: "MACHINE WINS", draw: "DRAW" };
  const rt = $("resultText");
  rt.textContent = texts[outcome];
  rt.className = "result-text " + (outcome === "you" ? "win" : outcome === "pi" ? "lose" : "draw");

  const youCard = document.querySelector(".player-card:first-of-type");
  const piCard = document.querySelector(".player-card:last-of-type");
  youCard.classList.toggle("win-card", outcome === "you");
  piCard.classList.toggle("win-card", outcome === "pi");
  youCard.classList.toggle("lose-card", outcome === "pi");
  piCard.classList.toggle("lose-card", outcome === "you");

  $("roundNum").textContent = state.round;
  $("scoreYou").textContent = state.score.you;
  $("scorePi").textContent = state.score.pi;
  $("scoreDraw").textContent = state.score.draw;
  $("result").classList.remove("hidden");
}

/* ---------- countdown ---------- */
async function startRound() {
  $("result").classList.add("hidden");
  hideOverlay();
  $("hud").classList.add("hidden");
  state.pending = null;
  state.holdCount = 0;

  state.phase = "countdown";
  const nums = ["3", "2", "1", "\u2703\uFE0F"];
  for (let i = 0; i < nums.length; i++) {
    showOverlay(nums[i], i === 3 ? "SHOW YOUR MOVE!" : "Get ready\u2026");
    await new Promise((r) => setTimeout(r, 800));
  }
  hideOverlay();
  $("hud").classList.remove("hidden");
  state.phase = "playing";
  state.classifyTimer = setInterval(classifyNow, CLASSIFY_MS);
  classifyNow();
}

/* ---------- wiring ---------- */
$("startBtn").addEventListener("click", async () => {
  $("landing").classList.add("hidden");
  if (!(await startCamera())) {
    $("landing").classList.remove("hidden");
    return;
  }
  startRound();
});

$("retryBtn").addEventListener("click", async () => {
  $("camError").classList.add("hidden");
  if (await startCamera()) startRound();
});

$("nextBtn").addEventListener("click", () => {
  stopClassify();
  startRound();
});

$("resetBtn").addEventListener("click", () => {
  state.score = { you: 0, pi: 0, draw: 0 };
  state.history = [];
  $("scoreYou").textContent = "0";
  $("scorePi").textContent = "0";
  $("scoreDraw").textContent = "0";
  $("roundNum").textContent = "0";
});

function stopClassify() {
  if (state.classifyTimer) clearInterval(state.classifyTimer);
  state.classifyTimer = null;
}
