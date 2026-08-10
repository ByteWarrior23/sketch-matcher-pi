/* Tiny Web-Audio sound engine — zero files, works everywhere */
"use strict";
const Sfx = (() => {
  let ctx = null;
  function ac() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) ctx = new AC();
    }
    if (ctx && ctx.state === "suspended") ctx.resume();
    return ctx;
  }
  function tone(freq, dur, type = "sine", vol = 0.2, when = 0, slide = 0) {
    const c = ac(); if (!c) return;
    const o = c.createOscillator();
    const g = c.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, c.currentTime + when);
    if (slide) o.frequency.exponentialRampToValueAtTime(slide, c.currentTime + when + dur);
    g.gain.setValueAtTime(vol, c.currentTime + when);
    g.gain.exponentialRampToValueAtTime(0.001, c.currentTime + when + dur);
    o.connect(g); g.connect(c.destination);
    o.start(c.currentTime + when);
    o.stop(c.currentTime + when + dur + 0.02);
  }
  return {
    unlock() { ac(); },
    click() { tone(520, 0.08, "triangle", 0.12); },
    count() { tone(440, 0.12, "triangle", 0.16); },
    go() { tone(880, 0.18, "triangle", 0.2); },
    thinking() { tone(300, 0.35, "sine", 0.12, 0, 220); },
    reveal() { tone(660, 0.12, "square", 0.1); tone(990, 0.18, "square", 0.1, 0.1); },
    win() { [523, 659, 784, 1047].forEach((f, i) => tone(f, 0.16, "triangle", 0.18, i * 0.09)); },
    lose() { [392, 330, 262].forEach((f, i) => tone(f, 0.18, "sawtooth", 0.1, i * 0.12)); },
    draw() { tone(500, 0.15, "sine", 0.14); tone(500, 0.15, "sine", 0.14, 0.16); },
    pop() { tone(700, 0.06, "square", 0.08, 0, 1400); },
    start() { tone(400, 0.1, "triangle", 0.15, 0, 800); },
  };
})();
document.addEventListener("click", () => Sfx.unlock(), { once: true });
