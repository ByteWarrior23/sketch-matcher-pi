# RPS Project — Session Notes

Project: Rock-Paper-Scissors AI game (camera gesture recognition)
Owner repo: `E:\SoftComputing\camera_pi_projects\rps\`
Laptop env: `E:\SoftComputing\sketch-matcher-pi\sketch_matcher_env` (TF 2.21.0, cv2, numpy 2.5.1)

## Status: LIVE AND WORKING
- Web game runs at **http://localhost:8123** — open in browser, allow camera, play.
- Uses the **local float32 stopgap model** (`models/local/rps_local.tflite`, 128px, **85.5%** test acc, ~6ms/img) — the BEST measured model.
- HPC int8 model was **REJECTED**: 75.3% acc + 664ms/img (XNNPACK incompatible) → degrades performance.
- HPC float32 conversion: 77% acc, 8ms/img — fast but lower acc on test set; kept as backup.

---

## Final model benchmark (Google rps-test-set, 372 images)
| Model | Acc | Latency | Verdict |
|---|---|---|---|
| local stopgap float32 (`models/local/rps_local.tflite`) | **85.5%** (318/372) | ~6ms | ✅ ACTIVE for web |
| HPC float32 (`models/hpc/rps_float32.tflite`, 12.4MB) | 77% (286/372) | 8ms | backup (fast, lower acc) |
| HPC int8 (`pi_deploy/model_data/rps_model.tflite`, 3.6MB) | 75.3% (280/372) | 664ms | ❌ rejected (slow + XNNPACK crash) |

Note: HPC models trained on a different dataset (kagglehub drgfreeman, real photos) than the Google test set → their lower test score is partly cross-dataset. For real-camera robustness they may still be fine; stopgap won on measured accuracy.

---

## Session log (2026-08-10)

### Done
- **HPC training completed**: 3-stage MobileNetV3Large transfer learning, early-stopped epoch 13/40, val_accuracy 1.0000, val_loss 0.0097. Saved `rps_model.keras` (38MB).
  - Bugs fixed before/during: backbone name (`MobileNetV3Large`), stage-3 freeze bug (was `None` → now `0` = full fine-tune), data leak (was train=val=test 2188 each → now stratified 1752/218/218).
- **int8 TFLite export** (`hpc/rps_export.py`): fixed rep dataset (was uint8, needs float32 [0,255] list), produced `rps_model.tflite` (3.6MB) + `labels.json`. Export script's own verify step crashed on XNNPACK/int8-MobileNetV3 incompat — model file itself valid, verify only.
- **TFJS conversion abandoned**: tensorflowjs 3.18 breaks on numpy2, 4.22 needs jax/decision-forests native build, fails on Windows. → Pivoted to **local Python server** running the TFLite classifier server-side.
- **Web frontend built** (glassmorphism): `web/index.html`, `web/style.css`, `web/app.js`.
  - Camera → 3-2-1 countdown → live hold-detection (5 consecutive frames ≥70% conf) → reveal you-vs-machine side-by-side → scoreboard.
  - Machine plays counter-strategy on your last 4 moves, 25% random.
- **`src/classifier.py` fixed**: added missing `labels_path` attribute; added XNNPACK workaround (`BUILTIN_WITHOUT_DEFAULT_DELEGATES`) needed for int8 model to load.
- **Server live**: `web/server.py` (stdlib only), port 8123, `pick_model()` prefers `pi_deploy/model_data` → falls back to `models/local`. /health, /, /classify all verified 200. Browser opened.
- int8 files moved to `*.int8.bak` / `labels.json.bak` so server picks the stopgap (restore anytime).

### Known gaps / next steps
- Browser end-to-end test with real hand (camera) — user to try at localhost:8123.
- If live acc is poor: try HPC float32 (`--model-dir models/hpc` with float32 tflite) or restore int8.
- **Raspberry Pi deployment** (`pi_deploy/main_rps.py`) — NOT started. int8 model is Pi-sized; verify delegate on Pi (XNNPACK works there), else use float32 (12.4MB).
- HPC SSH is flaky — retry short commands; account expires 2026-08-21.

## Commands
```powershell
# start server
& "E:\SoftComputing\sketch-matcher-pi\sketch_matcher_env\Scripts\python.exe" web/server.py --port 8123
# override model
& "...\python.exe" web/server.py --port 8123 --model-dir models/local
# quick classify test
Invoke-RestMethod -Method Post -Uri http://localhost:8123/classify -ContentType image/jpeg -Body ([IO.File]::ReadAllBytes("...testrock01-00.png"))
```
