# Neon Arcade

A **gesture-based arcade**. Your webcam tracks your **hand position** and turns
it into game input — no keyboard, mouse, or touch needed. It drives two browser
games: **Subway Surfers** (Poki, index-finger joystick) and **Level Devil**
(hand-position zones), both via a local keyboard bridge.

Everything runs locally — camera frames never leave your machine.

## Quick start

1. Install Python requirements: `pip install -r requirements.txt`.
2. Run the arcade: `python src/server.py --port 8080`.
3. Open **http://localhost:8080** and pick a game.

Games launch in their own Chrome/Edge window; the page becomes your gesture
controller.

## How to play

1. On a game page click **Start game** and allow camera access.
2. Click **Open game** to launch the game in its own window.
3. **Click inside the game once** so keyboard controls reach it, then steer.
4. Play with your hand (see Controls below).

### Index-finger joystick (Subway Surfers)

Keep your hand still and steer with the index fingertip. Its resting spot is
tracked as the center (white crosshair); moving the fingertip clear of the
center fires that direction's arrow key through the keyboard bridge, and
returning to center re-arms it.

| Index move | Command |
|---|---|
| Index left | Lane left (←) |
| Index right | Lane right (→) |
| Index up | Jump (↑) |
| Index down | Roll (↓) |
| Index to center | Re-arm between moves |

### Hand-zone controls (Level Devil)

Control the game by holding your hand in a zone. Zones use hysteresis, so the
hand can rest near a boundary without flickering commands.

| Zone | Level Devil |
|---|---|
| Hand left side | Move left (←) |
| Hand right side | Move right (→) |
| Hand high in center | Jump (Space) |
| Hand high on a side | Run + keep jumping |
| Hand low in center | Stop |

## Architecture

```
rps/
├── src/            Python backend — web server + keyboard input bridge
├── web/            Frontend — hub, shared JS, per-game pages (web/games/)
├── models/         MediaPipe hand landmarker model
├── scripts/        Start scripts
└── tests/          Route + bridge smoke tests
```

Key components:

- **`src/server.py`** — stdlib HTTP server (port 8080). Serves static frontend,
  `/api/status`, `/api/prepare` (launch game window) and `/api/key` (keyboard
  press/hold).
- **`src/input_bridge.py`** — maps gestures to keyboard keys for the focused
  game window; launches the games in Chrome/Edge.
- **`web/js/`** — `arcade.js` (camera/UI/API), `hand-runner-bridge.js`
  (in-browser MediaPipe HandLandmarker → hand position → game keys).

Hand detection runs **directly in the browser** via MediaPipe HandLandmarker
(WASM/GPU); only the final key action is POSTed to the server.

## Requirements

- Windows (for `pydirectinput` keyboard bridge) + Chrome or Edge
- Python 3.11+ with `pydirectinput` (see `requirements.txt`)

## Known notes

- The game window must have keyboard focus — click inside it once after launch.
- Session history and tuning notes: `docs/session-notes.md`.
