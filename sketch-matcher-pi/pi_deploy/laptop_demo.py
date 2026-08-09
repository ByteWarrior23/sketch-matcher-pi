"""
laptop_demo.py — Live webcam demo of the sketch matcher on a laptop/PC.

Reuses the Pi deployment modules (CameraCapture + SketchMatcher + Display)
but opens the laptop webcam (OpenCV) instead of the Pi camera, and drops
all GPIO logic.

Controls:
  SPACE  -> capture current frame, run matcher, show top-K result
  ESC    -> quit

Usage:
  python pi_deploy/laptop_demo.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2

from config import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    from camera import CameraCapture
    from matcher import SketchMatcher
    from display import Display

    model_dir = Path(__file__).parent / "model_data"

    cam = CameraCapture(camera_id=0)
    cam.initialize()

    matcher = SketchMatcher(model_dir)
    display = Display()

    log.info("=" * 60)
    log.info("LIVE WEBcam SKETCH MATCHER (laptop)")
    log.info("Hold a sketch (black ink on white paper) filling the frame,")
    log.info("press SPACE to match. ESC to quit.")
    log.info("=" * 60)

    match_count = 0
    try:
        while True:
            frame = cam.capture_image()
            if frame is None:
                log.warning("No camera frame (camera busy?). Retrying...")
                time.sleep(0.1)
                continue

            preview = cv2.resize(frame, (960, 540))
            cv2.imshow("Live Preview", preview)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            if key != 32:  # SPACE
                continue

            match_count += 1
            log.info(f"\n--- Capture #{match_count} ---")

            t0 = time.time()
            processed = cam.preprocess(frame)
            t1 = time.time()
            log.info(f"Preprocessed: {processed.shape} ({t1-t0:.2f}s)")

            # --- debug: save raw frame + preprocessed input for inspection ---
            dbg_dir = Path(__file__).parent / "debug_captures"
            dbg_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(dbg_dir / f"raw_{match_count}.jpg"), frame)
            cv2.imwrite(str(dbg_dir / f"pre_{match_count}.jpg"),
                        (processed[0] * 255).astype("uint8"))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ink_frac = float((gray < 250).mean())
            log.info(f"DEBUG: ink_frac(frame)={ink_frac:.3f} "
                     f"pre_min={processed.min():.3f} pre_max={processed.max():.3f} "
                     f"pre_mean={processed.mean():.3f}")

            results, inf_ms, search_ms, rejected = matcher.match(processed)
            total_ms = inf_ms + search_ms
            log.info(f"Result: {results} ({total_ms:.0f}ms)"
                     f"{' [REJECTED -> NOT FOUND]' if rejected else ''}")

            display.show_result(results, inf_ms, search_ms, rejected)
            cv2.waitKey(3000)

    except KeyboardInterrupt:
        log.info("\nInterrupted by user.")
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
    finally:
        cam.release()
        cv2.destroyAllWindows()
        log.info("Goodbye!")


if __name__ == "__main__":
    main()
