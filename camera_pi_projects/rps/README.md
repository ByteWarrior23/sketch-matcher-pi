# Neon Arcade

A **gesture-based arcade**. Your webcam tracks your **body pose** or **hand
gestures** and turns them into game input — no keyboard, mouse, or touch needed.
It drives real games running in BlueStacks (Subway Surfers, Temple Run) via ADB,
plus in-browser hand games (Rock Paper Scissors, Finger Math, Simon Says) and a
keyboard-bridged browser game (Level Devil).

Everything runs locally — camera frames never leave your machine.

## Quick start

1. **Install BlueStacks** (one time): double-click `scripts/install_bluestacks.bat`
   (or let the arcade install it automatically on first run).
2. In the BlueStacks Play Store, install **Subway Surfers** and/or **Temple Run 2**.
3. BlueStacks → **Settings → Advanced → Android Debug Bridge → Enable**.
4. Run the arcade: double-click `scripts/start_arcade.bat`.
5. Open **http://127.0.0.1:8123** and pick a game.

The **real game runs in BlueStacks**; the browser page is your camera + gesture
controller. First run auto-installs BlueStacks, connects ADB, and sideloads game
APKs from APKPure — keep the page open while it sets up.

## How to play

From the hub (**http://127.0.0.1:8123**), click a game card. Every game follows
one of three flows:

### 1. Real games in BlueStacks (Subway Surfers, Temple Run, RPS / Simon / Finger Math)

1. On the game page click **Start hand-swipe control** (Subway/Temple Run) or
   **Real game · BlueStacks** (RPS / Simon / Finger Math).
2. Allow camera access when prompted.
3. First time: the arcade auto-installs BlueStacks, enables + connects ADB,
   sideloads the game APK, and launches the game. **Keep this page open** — first
   run takes a few minutes and the status line shows progress.
4. When it says **"Real game launched"**, alt-tab to the BlueStacks window — the
   game is running there. The browser page stays open as your controller.
5. Swipe your hand to play (see Controls).

### 2. Browser games (RPS, Finger Math, Simon Says)

1. Click **Browser duel** (RPS) or the start button (Finger Math / Simon).
2. Allow camera access.
3. **RPS** — throw rock / paper / scissors by holding the gesture steady; first
   to 3 wins the match.
4. **Finger Math** — solve the equation by raising that many fingers.
5. **Simon Says** — watch the machine's sequence, then copy it gesture by
   gesture. 3 lives.

### 3. Level Devil (keyboard bridge)

1. Click **Start**. The game loads in a panel and in its own window.
2. **Click inside the game panel once** so it has keyboard focus — your hand
   swipes then send arrow keys / Space to the game.
3. Swipe left / right to move, swipe up to jump.

**Swipe tip:** make quick, deliberate hand swipes in the direction you want
(like swiping on a phone screen), and bring your hand back to center between
swipes. Swipes are edge-triggered — each one fires a single command.

## Games

All games are played with **gestures** — your body pose or hand movements become
the controller, so you never touch a keyboard, mouse, or screen.

| Game | Control | Target |
|---|---|---|
| Subway Surfers | Hand swipes | Real game in BlueStacks |
| Temple Run | Hand swipes | Real game in BlueStacks |
| Rock Paper Scissors | Hand gestures | Browser or real game in BlueStacks |
| Finger Math | Finger count (0–5) | Browser or real game in BlueStacks |
| Simon Says | Hand gestures | Browser or real game in BlueStacks |
| Level Devil | Hand swipes | Browser game via keyboard bridge |

### Hand-swipe controls (Subway Surfers, Temple Run, Level Devil)

Control games by moving your hand:

| Gesture | Subway Surfers | Temple Run | Level Devil |
|---|---|---|---|
| Swipe hand left | Swipe left | Turn left | Move left (←) |
| Swipe hand right | Swipe right | Turn right | Move right (→) |
| Swipe hand up | Jump | Jump | Jump (Space) |
| Swipe hand down | Roll | Slide | — (ignored) |

How it works:

1. **Swipe, don't hold** — make a quick, deliberate motion in the direction you
   want (like swiping on a phone screen). Each swipe fires one command.
2. **Reset between swipes** — bring your hand back toward center before the next
   swipe so movements don't merge.
3. For Subway/Temple Run the command becomes an ADB swipe on the BlueStacks
   screen (90 ms swipe); for Level Devil it becomes a keyboard press — click
   inside the game panel once so it has focus.
4. Swipe detection is edge-triggered with a ~0.6 s cooldown — a sustained held
   hand will never repeat a command.

### Hand-gesture controls (RPS, Simon, Finger Math)

| Game | Gesture | Action |
|---|---|---|
| RPS | Rock / Paper / Scissors held steady | Tap that move (real game) or register it (browser) |
| Simon Says | Rock → Green · Paper → Red · Scissors → Yellow | Tap the matching button |
| Finger Math | Show fingers 0–5 | Tap that number, then auto-confirm |

How it works:

1. **Hold steady to register** — keep the gesture still for ~4 camera frames
   (~0.6 s) at ≥ 70% confidence. Holding an already-registered gesture doesn't
   re-fire; relax or switch gestures first.
2. **RPS (browser)** — first to 3 wins the match; the machine predicts your moves
   from your history. **RPS (real)** — your hand taps rock/paper/scissors on the
   RPS app in BlueStacks via ADB.
3. **Simon Says** — watch the sequence, then copy it one gesture at a time. 3
   lives; a steady wrong gesture costs one. Between steps, relax your hand (or
   switch gesture) to break the hold.
4. **Finger Math** — solve `a + b` or `a − b` (answers 1–5) by raising that many
   fingers. In the real-game mode the number is tapped, then the confirm button
   is pressed automatically.

Real BlueStacks hand games (RPS, Simon, Finger Math) are played in the
BlueStacks window — your camera hand drives taps into it; the browser page is
just your controller.

## Architecture

```
rps/
├── src/            Python backend — web server, vision, input bridge, setup
├── web/            Frontend — hub, shared JS, per-game pages (web/games/), assets
├── models/         Runtime models (MediaPipe hand landmarker, TFLite fallback)
├── scripts/        Setup / start scripts
├── tools/          ADB platform-tools, cached game APKs, BlueStacks installer
├── tests/          Route + bridge smoke tests
├── docs/           Session notes
└── logs/           Runtime logs
```

Key components:

- **`src/server.py`** — stdlib HTTP server (port 8123). Serves static frontend,
  `POST /classify` + `POST /hand` (MediaPipe hand recognition), and the input
  bridge API (`/api/status`, `/api/connect`, `/api/launch`, `/api/action`).
- **`src/hand_rps.py`** — MediaPipe HandLandmarker: geometric RPS + finger
  counting + hand-swipe direction tracking.
- **`src/input_bridge.py`** — maps gestures to ADB taps/swipes (BlueStacks) and
  keyboard keys (Level Devil), turning each recognized gesture into game input.
- **`src/arcade_setup.py`** — one-click BlueStacks install, ADB enable/connect,
  and APK sideloading.
- **`web/js/`** — shared frontend: `arcade.js` (camera/UI/API),
  `hand-runner-bridge.js` (hand swipe → runner games + Level Devil),
  `hand-bridge.js` (hand game taps), `pose.js` / `runner-bridge.js` (legacy body
  pose path).
- **`web/js/games/`** — per-game gesture logic; **`web/games/`** — per-game pages.

## Manual commands

```powershell
# start server
& "E:\SoftComputing\sketch-matcher-pi\sketch_matcher_env\Scripts\python.exe" src/server.py --port 8123

# install all 5 game APKs (BlueStacks + ADB connected)
& "<venv-python>" scripts/install_apks.py

# verify routes against a running server
& "<venv-python>" scripts/verify_routes.py
```

## Requirements

- Windows + BlueStacks 5 (for real games)
- Python 3.11+ with `mediapipe`, `opencv-python`, `numpy`, `tensorflow`
  (see `requirements.txt`); `pydirectinput` for the Level Devil keyboard bridge

## Known notes

- Real-world APK mirrors can be flaky; APK download retries across package
  alternatives (e.g. RPS, Simon, Finger Math) are built into `src/arcade_setup.py`.
- Session history and tuning notes: `docs/session-notes.md`.
