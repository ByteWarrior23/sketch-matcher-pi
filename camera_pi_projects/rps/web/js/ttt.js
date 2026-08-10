/* Tic Tac Toe — 2 player, settable rounds, sound */
"use strict";

const $ = (id) => document.getElementById(id);
let WINS_NEEDED = 1;
let current = "X";
let board = Array(9).fill(null);
let score = { X: 0, O: 0 };
let round = 0;

function nameOf(sym) {
  const p1 = $("p1Name").value.trim() || "Player X";
  const p2 = $("p2Name").value.trim() || "Player O";
  return sym === "X" ? p1 : p2;
}

function renderScore() {
  $("tttP1Label").textContent = nameOf("X");
  $("tttP2Label").textContent = nameOf("O");
  $("tttP1").textContent = score.X;
  $("tttP2").textContent = score.O;
  $("tttRound").textContent = round;
  renderPips();
}

function renderPips() {
  const pips = $("roundPips");
  pips.innerHTML = "";
  for (let i = 0; i < WINS_NEEDED * 2 - 1; i++) {
    const p = document.createElement("span");
    p.className = "pip";
    if (i < score.X) p.classList.add("you");
    else if (i < score.X + score.O) p.classList.add("pi");
    pips.appendChild(p);
  }
}

function turnUI() {
  $("tttTurnName").textContent = nameOf(current);
  $("tttTurnSym").textContent = current;
  $("tttTurnDot").className = "turn-dot " + (current === "X" ? "dot-x" : "dot-o");
  $("tttStatus").textContent = "";
}

function resetBoard() {
  board = Array(9).fill(null);
  document.querySelectorAll(".cell").forEach((c) => {
    c.textContent = "";
    c.classList.remove("x", "o", "win-cell");
  });
  current = round % 2 === 0 ? "X" : "O"; // alternate who starts
  turnUI();
}

function checkWin(b) {
  const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
  for (const [a,bb,c] of lines) {
    if (b[a] && b[a] === b[bb] && b[a] === b[c]) return { winner: b[a], line: [a,bb,c] };
  }
  return b.every(Boolean) ? { winner: "draw", line: [] } : null;
}

function cellClick(e) {
  const i = Number(e.currentTarget.dataset.i);
  if (board[i]) return;
  board[i] = current;
  const cell = e.currentTarget;
  cell.textContent = current;
  cell.classList.add(current.toLowerCase());
  Sfx.pop();

  const res = checkWin(board);
  if (res) {
    if (res.winner === "draw") {
      Sfx.draw();
      $("tttStatus").textContent = "It's a draw!";
      round += 1;
      flash("draw-flash");
    } else {
      score[res.winner] += 1;
      res.line.forEach((idx) => document.querySelector(`.cell[data-i="${idx}"]`).classList.add("win-cell"));
      $("tttStatus").textContent = `${nameOf(res.winner)} wins the round!`;
      Sfx.win();
      flash("win-flash");
      round += 1;
      if (score[res.winner] >= WINS_NEEDED) {
        $("tttStatus").textContent = `🏆 ${nameOf(res.winner)} takes the match!`;
        Sfx.win();
        renderScore();
        return; // match over
      }
    }
    renderScore();
    setTimeout(resetBoard, 1400);
    return;
  }
  current = current === "X" ? "O" : "X";
  turnUI();
}

function flash(cls) {
  const f = $("flash");
  f.className = "flash show " + cls;
  setTimeout(() => (f.className = "flash"), 900);
}

// setup
document.querySelectorAll("#roundSel .pill").forEach((b) => {
  b.addEventListener("click", () => {
    Sfx.click();
    document.querySelectorAll("#roundSel .pill").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    WINS_NEEDED = Number(b.dataset.n);
  });
});

$("tttStart").addEventListener("click", () => {
  Sfx.start();
  score = { X: 0, O: 0 };
  round = 1;
  $("tttSetup").classList.add("hidden");
  $("tttPlay").classList.remove("hidden");
  resetBoard();
  renderScore();
});

$("tttReset").addEventListener("click", () => {
  Sfx.click();
  $("tttPlay").classList.add("hidden");
  $("tttSetup").classList.remove("hidden");
});

document.querySelectorAll(".cell").forEach((c) => c.addEventListener("click", cellClick));
