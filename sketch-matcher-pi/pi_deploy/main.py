"""
main.py — Main Entry Point for Raspberry Pi Sketch Matcher

This script runs ON the Raspberry Pi 5 at boot time.
It orchestrates:

  1. Camera initialization
  2. Model + embeddings loading
  3. Display initialization (HDMI + GPIO LEDs)
  4. Main loop:
     a. Show "Ready" screen
     b. Wait for button press (GPIO) or keyboard (SPACE)
     c. Capture image from camera
     d. Preprocess image (crop, binarize, resize)
     e. Compute embedding via TFLite model
     f. Nearest neighbor search against photo database
     g. Display top-K results on HDMI + light LEDs
  5. Loop forever

To run on Pi:
  cd ~/sketch_matcher
  python pi_deploy/main.py

To auto-start on boot:
  Add to /etc/rc.local or create a systemd service.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PI_GPIO_BUTTON_PIN, LOG_LEVEL

import logging

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Try importing GPIO
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    log.warning("RPi.GPIO not available. Using keyboard (SPACE) for capture.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def setup_gpio_button():
    """Configure GPIO button with pull-up resistor and debounce."""
    if not GPIO_AVAILABLE:
        return None

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PI_GPIO_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Event for falling edge (button press)
    # We'll poll instead of interrupt for simplicity
    log.info(f"GPIO button configured on pin {PI_GPIO_BUTTON_PIN}")
    return PI_GPIO_BUTTON_PIN


def wait_for_capture(button_pin, timeout_seconds=None):
    """
    Wait for either GPIO button press or SPACE key.

    Returns:
      True if capture triggered, False on timeout
    """
    t_start = time.time()

    while True:
        # Check timeout
        if timeout_seconds and (time.time() - t_start) > timeout_seconds:
            return False

        # Check GPIO button (if available)
        if GPIO_AVAILABLE and button_pin:
            if GPIO.input(button_pin) == GPIO.LOW:
                # Debounce: wait 50ms and check again
                time.sleep(0.05)
                if GPIO.input(button_pin) == GPIO.LOW:
                    # Wait for release
                    while GPIO.input(button_pin) == GPIO.LOW:
                        time.sleep(0.01)
                    return True

        # Check keyboard (SPACE key)
        if CV2_AVAILABLE:
            key = cv2.waitKey(10) & 0xFF
            if key == 32:  # SPACE
                return True
            if key == 27:  # ESC
                log.info("ESC pressed. Exiting.")
                return None

        time.sleep(0.01)


def main():
    log.info("=" * 60)
    log.info("SKETCH MATCHER - Raspberry Pi 5")
    log.info("=" * 60)

    # Step 1: Initialize camera
    log.info("\n[1/5] Initializing camera...")
    from camera import CameraCapture
    cam = CameraCapture()
    cam.initialize()

    # Step 2: Load model + embeddings
    log.info("\n[2/5] Loading model and embeddings...")
    model_dir = Path(__file__).parent / "model_data"
    if not model_dir.exists():
        log.error(f"Model directory not found: {model_dir}")
        log.error("Copy model files first:")
        log.error("  scp -r pi_deploy/model_data/ pi@raspberrypi:~/sketch_matcher/pi_deploy/")
        cam.release()
        return

    from matcher import SketchMatcher
    matcher = SketchMatcher(model_dir)

    # Step 3: Initialize display
    log.info("\n[3/5] Initializing display...")
    from display import Display
    display = Display()
    display.show_ready()
    log.info("Display ready. Place sketch and press button/SPACE.")

    # Step 4: Setup GPIO button
    log.info("\n[4/5] Setting up capture button...")
    button_pin = setup_gpio_button()

    # Step 5: Main loop
    log.info("\n[5/5] Entering main loop...")
    log.info("Press SPACE or GPIO button to capture.")
    log.info("Press ESC to quit.")

    capture_count = 0

    try:
        while True:
            # Wait for capture trigger
            result = wait_for_capture(button_pin)
            if result is None:
                # ESC pressed — exit
                break
            if not result:
                # Timeout — continue waiting
                continue

            capture_count += 1
            log.info(f"\n--- Capture #{capture_count} ---")

            # Capture image
            display.show_processing()
            raw_img = cam.capture_image()
            log.info(f"Image captured: {raw_img.shape}")

            # Preprocess
            t0 = time.time()
            processed = cam.preprocess(raw_img)
            t1 = time.time()
            log.info(f"Preprocessed: {processed.shape} ({t1-t0:.2f}s)")

            # Match
            results, inf_ms, search_ms, rejected = matcher.match(processed)
            total_ms = inf_ms + search_ms
            log.info(f"Result: {results} ({total_ms:.0f}ms)"
                     f"{' [REJECTED -> NOT FOUND]' if rejected else ''}")

            # Display result
            display.show_result(results, inf_ms, search_ms, rejected)
            log.info("=" * 60)

            # Wait 2 seconds before showing "Ready" again
            cv2.waitKey(2000) if CV2_AVAILABLE else time.sleep(2)
            display.show_ready()

    except KeyboardInterrupt:
        log.info("\nUser interrupted.")
    except Exception as e:
        log.error(f"Error: {e}")
        display.show_error(str(e))
        time.sleep(3)
    finally:
        log.info("\nCleaning up...")
        cam.release()
        display.cleanup()
        log.info("Goodbye!")


if __name__ == "__main__":
    main()
