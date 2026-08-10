/* RPS vs the Machine — match system (first to 3 = 5 rounds) */
"use strict";

const MOVES = { rock: "\u270A", paper: "\u270B", scissors: "\u270C\uFE0F" };
const BEATS = { rock: "scissors", paper: "rock", scissors: "paper" };
const MIN_CONF = 0.7;
const HOLD = 4;
const CLASSIFY_MS = 140;
const WIN_TARGET = 3;

const $ = (id) => document.getElementById(id);
const video = $("video");
const preview = $("preview");
const pctx = preview.getContext("2d");

const state = {
  phase: "idle",
  pending: null,
  holdCount: 0,
  history: [],
  score: { you: 0, pi: 0, draw: 0 },
  streak: 0,
  round: 0,
  camera: null,
  classifyTimer: null,
  noHandFrames: 0,
};

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

function showOverlay(big, sub) {
  $("overlayBig").textContent = big;
  $("overlaySub").textContent = sub || "";
  $("overlay").classList.remove("hidden");
}
function hideOverlay() { $("overlay").classList.add("hidden"); }

function classifyNow() {
  if (video.readyState < 2) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  const size = Math.min(vw, vh);
  const sx = (vw - size) / 2, sy = (vh - size) / 2;
  pctx.clearRect(0, 0, preview.width, preview.height);
  pctx.drawImage(video, sx, sy, size, size, 0, 0, 224, 224);
  preview.toBlob(async (blob) => {
    try {
      const res = await fetch("/classify", { method: "POST", body: blob });
      if (!res.ok) return;
      onPrediction(await res.json());
    } catch (_) {}
  }, "image/jpeg", 0.9);
}

function onPrediction(r) {
  if (state.phase !== "playing") return;

  if (!r.detected) {
    state.noHandFrames += 1;
    if (state.noHandFrames >= 3) {
      state.pending = null;
      state.holdCount = 0;
      $("hudMove").textContent = "👋";
      $("hudConf").textContent = "show your hand";
    }
    return;
  }
  state.noHandFrames = 0;
  $("hudMove").textContent = MOVES[r.label] || "?";
  $("hudConf").textContent = `${Math.round(r.conf * 100)}%`;

  if (r.conf >= MIN_CONF) {
    if (state.pending === r.label) { state.holdCount += 1; }
    else { state.pending = r.label; state.holdCount = 1; }
    if (state.holdCount >= HOLD) registerMove(r.label);
  } else {
    state.pending = null;
    state.holdCount = 0;
    $("hudMove").textContent = "👋";
    $("hudConf").textContent = "hold still";
  }
}

function aiPredict() {
  const h = state.history;
  if (h.length < 2) return null;
  const last = h[h.length - 1];
  const votes = { rock: 0, paper: 0, scissors: 0 };
  votes[last] += 3;
  const prev = h[h.length - 2];
  votes[prev] += 2;
  const bigrams = {};
  for (let i = 0; i < h.length - 1; i++) {
    const key = h[i] + ">" + h[i + 1];
    bigrams[key] = (bigrams[key] || 0) + 1;
  }
  if (bigrams[prev + ">" + last]) votes[last] += bigrams[prev + ">" + last] * 2;
  let best = "rock";
  for (const k of Object.keys(votes)) if (votes[k] > votes[best]) best = k;
  return best;
}

function aiMove() {
  const pred = aiPredict();
  if (pred && Math.random() < 0.55) return BEATS[pred];
  return ["rock", "paper", "scissors"][Math.floor(Math.random() * 3)];
}

function registerMove(playerMove) {
  state.phase = "revealed";
  clearInterval(state.classifyTimer);
  state.classifyTimer = null;
  $("hud").classList.add("hidden");
  Sfx.thinking();

  const pi = aiMove();
  state.history.push(playerMove);
  if (state.history.length > 12) state.history.shift();
  state.round += 1;

  let outcome;
  if (playerMove === pi) outcome = "draw";
  else if (BEATS[playerMove] === pi) outcome = "you";
  else outcome = "pi";
  state.score[outcome] += 1;
  state.streak = outcome === "you" ? state.streak + 1 : 0;

  showOverlay("…", "Machine is thinking");
  setTimeout(() => {
    hideOverlay();
    showResult(playerMove, pi, outcome);
  }, 650);
}

function showResult(you, pi, outcome) {
  $("youMove").textContent = MOVES[you];
  $("piMove").textContent = MOVES[pi];
  $("youLabel").textContent = you;
  $("piLabel").textContent = pi;

  const texts = { you: "YOU WIN", pi: "MACHINE WINS", draw: "DRAW" };
  const sub = { you: `${you} beats ${pi}`, pi: `${pi} beats ${you}`, draw: `${you} ties ${pi}` };
  const rt = $("resultText");
  rt.textContent = texts[outcome];
  $("resultSub").textContent = sub[outcome];
  rt.className = "result-text " + (outcome === "you" ? "win" : outcome === "pi" ? "lose" : "draw");

  const pills = document.querySelectorAll(".player-pill");
  pills[0].classList.toggle("win-card", outcome === "you");
  pills[1].classList.toggle("win-card", outcome === "pi");
  pills[0].classList.toggle("lose-card", outcome === "pi");
  pills[1].classList.toggle("lose-card", outcome === "you");

  $("roundNum").textContent = state.round;
  $("scoreYou").textContent = state.score.you;
  $("scorePi").textContent = state.score.pi;
  $("scoreDraw").textContent = state.score.draw;
  $("streakNum").textContent = state.streak > 1 ? `🔥 Win streak ×${state.streak}` : "";

  if (outcome === "you") Sfx.win(); else if (outcome === "pi") Sfx.lose(); else Sfx.draw();
  renderPips();
  $("result").classList.remove("hidden");

  if (state.score.you >= WIN_TARGET || state.score.pi >= WIN_TARGET) {
    setTimeout(endMatch, 1200);
  }
}

function endMatch() {
  $("result").classList.add("hidden");
  const youWon = state.score.you >= WIN_TARGET;
  $("matchText").textContent = youWon ? "YOU TAKE THE MATCH!" : "MACHINE TAKES THE MATCH";
  $("matchText").className = "match-text " + (youWon ? "win" : "lose");
  $("matchScore").textContent = `${state.score.you} — ${state.score.pi}`;
  $("matchBanner").classList.remove("hidden");
  if (youWon) Sfx.win(); else Sfx.lose();
}

function renderPips() {
  const pips = $("roundPips");
  pips.innerHTML = "";
  for (let i = 0; i < WIN_TARGET * 2 - 1; i++) {
    const p = document.createElement("span");
    p.className = "pip";
    if (i < state.score.you) p.classList.add("you");
    else if (i < state.score.you + state.score.draw) p.classList.add("draw");
    else if (i < state.score.you + state.score.draw + state.score.pi) p.classList.add("pi");
    pips.appendChild(p);
  }
}

async function startRound() {
  $("result").classList.add("hidden");
  $("matchBanner").classList.add("hidden");
  hideOverlay();
  $("hud").classList.add("hidden");
  state.pending = null;
  state.holdCount = 0;
  state.noHandFrames = 0;

  state.phase = "countdown";
  const nums = ["3", "2", "1", "✂️"];
  for (let i = 0; i < nums.length; i++) {
    showOverlay(nums[i], i === 3 ? "SHOW YOUR MOVE!" : "Get ready…");
    if (i < 3) Sfx.count(); else Sfx.go();
    await new Promise((r) => setTimeout(r, 750));
  }
  hideOverlay();
  $("hud").classList.remove("hidden");
  state.phase = "playing";
  state.classifyTimer = setInterval(classifyNow, CLASSIFY_MS);
  classifyNow();
}

$("startBtn").addEventListener("click", async () => {
  $("landing").classList.add("hidden");
  Sfx.start();
  if (!(await startCamera())) { $("landing").classList.remove("hidden"); return; }
  startRound();
});
$("retryBtn").addEventListener("click", async () => {
  $("camError").classList.add("hidden");
  if (await startCamera()) startRound();
});
$("nextBtn").addEventListener("click", () => { Sfx.click(); stopClassify(); startRound(); });
$("rematchBtn").addEventListener("click", () => {
  Sfx.click();
  state.score = { you: 0, pi: 0, draw: 0 };
  state.history = [];
  state.streak = 0;
  state.round = 0;
  $("scoreYou").textContent = "0"; $("scorePi").textContent = "0"; $("scoreDraw").textContent = "0";
  $("roundNum").textContent = "1";
  $("streakNum").textContent = "";
  renderPips();
  startRound();
});
$("resetBtn").addEventListener("click", () => {
  Sfx.click();
  state.score = { you: 0, pi: 0, draw: 0 };
  state.history = [];
  state.streak = 0;
  $("scoreYou").textContent = "0"; $("scorePi").textContent = "0"; $("scoreDraw").textContent = "0";
  $("roundNum").textContent = "1";
  $("streakNum").textContent = "";
  renderPips();
});

function stopClassify() {
  if (state.classifyTimer) clearInterval(state.classifyTimer);
  state.classifyTimer = null;
}
