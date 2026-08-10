"""Camera sources: laptop webcam or Android phone via IP Webcam app.

Usage:
  source = CameraSource.auto()                    # try webcam, then common IP cams
  source = CameraSource(port="0")                 # built-in webcam
  source = CameraSource(port="http://192.168.x.x:8080/video")  # IP Webcam URL
"""
import time

import cv2


class CameraSource:
    def __init__(self, port="0", width=640, height=480, retries=5):
        self.port = str(port)
        self.width = width
        self.height = height
        self.retries = retries
        self.cap = None
        self._open()

    def _open(self):
        port = int(self.port) if self.port.isdigit() else self.port
        self.cap = cv2.VideoCapture(port)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.is_ip():
            # IP Webcam streams need a moment to warm up
            time.sleep(1.0)

    def is_ip(self):
        return self.port.startswith("http")

    def read(self):
        """Return a BGR frame or None. Reconnects on IP streams if needed."""
        for _ in range(self.retries):
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame
            if self.is_ip():
                self.cap.release()
                time.sleep(1.0)
                self._open()
                time.sleep(0.5)
        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()

    @staticmethod
    def auto():
        """Try built-in webcam; else fall back to nothing (caller provides URL)."""
        try:
            c = CameraSource("0")
            frame = c.read()
            if frame is not None:
                return c
            c.release()
        except Exception:
            pass
        return None
