"""
display.py — Output Display for Raspberry Pi

Handles:
  1. Showing results on HDMI monitor / projector (OpenCV window)
  2. Controlling GPIO LEDs (green = genuine/match, red = forged/no match)
  3. Optional: speaking result via speaker (espeak)

Designed for stage demo: big clear text visible from across the room.
"""

from config import (
    PI_CONFIDENCE_THRESHOLD, PI_REJECT_THRESHOLD,
    PI_GPIO_LED_GREEN, PI_GPIO_LED_RED,
)
import logging
import numpy as np

log = logging.getLogger(__name__)

# Try importing GPIO (will fail on non-Pi systems = fine for testing)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    log.warning("RPi.GPIO not available (not running on Pi or not installed)")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class Display:
    """
    Manages visual + LED output for the sketch matcher.

    On HDMI: shows a big window with results
    On GPIO: lights green/red LEDs
    """

    def __init__(self, use_gpio=GPIO_AVAILABLE):
        self.use_gpio = use_gpio
        self.window_name = "Sketch Matcher"
        self.font = cv2.FONT_HERSHEY_SIMPLEX if CV2_AVAILABLE else None

        if use_gpio:
            self._init_gpio()

    def _init_gpio(self):
        """Initialize GPIO pins for LEDs."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PI_GPIO_LED_GREEN, GPIO.OUT)
        GPIO.setup(PI_GPIO_LED_RED, GPIO.OUT)
        self._led_off(PI_GPIO_LED_GREEN)
        self._led_off(PI_GPIO_LED_RED)
        log.info("GPIO LEDs initialized")

    def _led_on(self, pin):
        if self.use_gpio:
            GPIO.output(pin, GPIO.HIGH)

    def _led_off(self, pin):
        if self.use_gpio:
            GPIO.output(pin, GPIO.LOW)

    def show_result(self, results, inf_ms, search_ms, rejected=False):
        """
        Display matching results on screen + LEDs.

        Args:
          results: list of (category_name, confidence) tuples
          inf_ms: inference time in ms
          search_ms: nearest neighbor search time in ms
          rejected: True when open-set rejection says "NOT FOUND"
        """
        if not CV2_AVAILABLE:
            # Text-only output
            log.info("\n=== MATCH RESULT ===")
            if rejected:
                log.info("  >>> NOT FOUND (below reject threshold) <<<")
            for i, (cat, conf) in enumerate(results):
                marker = "✅" if conf >= PI_CONFIDENCE_THRESHOLD * 100 else "❌"
                log.info(f"  {marker} Top-{i+1}: {cat} ({conf:.1f}%)")
            log.info(f"  Inference: {inf_ms:.0f}ms | Search: {search_ms:.0f}ms")
            log.info("===================\n")
            return

        # Create a large clean display image
        display_img = self._create_result_display(results, inf_ms, search_ms, rejected)
        cv2.imshow(self.window_name, display_img)

        # Control LEDs
        if rejected or not results:
            self._led_on(PI_GPIO_LED_RED)
            self._led_off(PI_GPIO_LED_GREEN)
            return

        cat, conf = results[0]
        if conf >= PI_CONFIDENCE_THRESHOLD * 100:
            self._led_on(PI_GPIO_LED_GREEN)
            self._led_off(PI_GPIO_LED_RED)
        else:
            self._led_on(PI_GPIO_LED_RED)
            self._led_off(PI_GPIO_LED_GREEN)

    def _create_result_display(self, results, inf_ms, search_ms, rejected=False):
        """
        Create a large OpenCV image showing results.
        Designed for projector/HDMI visibility at distance.
        """
        width, height = 1280, 720
        img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        h, w = img.shape[:2]

        # Title
        cv2.putText(img, "SKETCH MATCHER RESULT", (w//2 - 300, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)

        # Top-3 results (big, bold, colored)
        colors = [(0, 180, 0), (100, 100, 100), (150, 150, 150)]  # Green, gray, light gray
        for i, (cat, conf) in enumerate(results):
            y = 150 + i * 140
            color = colors[i] if i < len(colors) else (0, 0, 0)

            # Category name (large)
            font_scale = 2.5 - i * 0.5  # Top-1 biggest
            text = f"#{i+1}: {cat.upper()}"
            cv2.putText(img, text, (100, y), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, color, 3)

            # Confidence (subtitle)
            conf_text = f"Confidence: {conf:.1f}%"
            cv2.putText(img, conf_text, (150, y + 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2)

            # Bar visualization
            bar_width = int((conf / 100) * 600)
            cv2.rectangle(img, (150, y + 70), (150 + bar_width, y + 85), color, -1)
            cv2.rectangle(img, (150, y + 70), (750, y + 85), (200, 200, 200), 1)

        # Timing info
        time_text = f"Inference: {inf_ms:.0f}ms  |  Search: {search_ms:.0f}ms  |  Total: {inf_ms + search_ms:.0f}ms"
        cv2.putText(img, time_text, (w//2 - 350, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        # Status line
        if rejected:
            status = "NOT FOUND  |  Below confidence threshold — try a clearer sketch"
            status_color = (0, 0, 180)
        elif results and results[0][1] >= PI_CONFIDENCE_THRESHOLD * 100:
            status = "MATCH FOUND  |  Confidence: HIGH"
            status_color = (0, 180, 0)
        else:
            status = "NO STRONG MATCH  |  Try a clearer sketch"
            status_color = (0, 0, 180)

        cv2.putText(img, status, (w//2 - 250, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        return img

    def show_ready(self):
        """Display 'Ready' screen — waiting for sketch."""
        if not CV2_AVAILABLE:
            return

        img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        cv2.putText(img, "SKETCH MATCHER", (340, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 0), 3)
        cv2.putText(img, "Place your sketch under the camera", (230, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
        cv2.putText(img, "Press BUTTON to capture", (380, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
        cv2.imshow(self.window_name, img)

    def show_processing(self):
        """Display 'Processing...' screen."""
        if not CV2_AVAILABLE:
            return

        img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        cv2.putText(img, "PROCESSING...", (380, 360),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 180), 3)
        cv2.imshow(self.window_name, img)

    def show_error(self, message):
        """Display error message."""
        if not CV2_AVAILABLE:
            log.error(message)
            return

        img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        cv2.putText(img, "ERROR", (540, 300),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 180), 3)
        cv2.putText(img, message, (440, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 0, 0), 2)
        cv2.imshow(self.window_name, img)

    def cleanup(self):
        """Clean up GPIO and close windows."""
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        if self.use_gpio:
            self._led_off(PI_GPIO_LED_GREEN)
            self._led_off(PI_GPIO_LED_RED)
            GPIO.cleanup()
        log.info("Display cleanup complete")
