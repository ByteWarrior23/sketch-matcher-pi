"""
camera.py — Raspberry Pi Camera Capture + Preprocessing

Runs ON the Raspberry Pi 5 (not on training machine).

This module handles:
  1. Capturing an image from Pi Camera Module
  2. Preprocessing it identically to training data:
     - Convert to grayscale
     - Crop to bounding box (remove whitespace)
     - Binarize (Otsu threshold)
     - Resize to 224x224 with padding
     - Convert to 3-channel
     - Normalize pixel values

Preprocessing must match EXACTLY what was done during training,
otherwise the model will produce garbage embeddings.

Dependencies (install on Pi):
  pip install opencv-python numpy picamera2 tflite-runtime
"""

import cv2
import numpy as np

from config import (
    IMG_SIZE, PADDING, BINARY_THRESHOLD,
    PI_CAMERA_RESOLUTION, PI_CAMERA_FRAMERATE,
)

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Warning: picamera2 not installed. Using mock camera for testing.")


class CameraCapture:
    """
    Handles image capture and preprocessing on Raspberry Pi.
    """

    def __init__(self, camera_id=0):
        self.resolution = PI_CAMERA_RESOLUTION
        self.framerate = PI_CAMERA_FRAMERATE
        self.camera = None
        self.camera_id = camera_id

    def initialize(self):
        """Initialize the camera hardware."""
        if PICAMERA_AVAILABLE:
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(
                main={"size": self.resolution},
                lores={"size": (640, 480)},
                display="lores"
            )
            self.camera.configure(config)
            self.camera.start()
            print(f"Camera initialized: {self.resolution}")
        else:
            # Fall back to OpenCV USB camera or mock
            self.camera = cv2.VideoCapture(self.camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            print(f"USB camera initialized (ID: {self.camera_id})")

    def capture_image(self):
        """
        Capture a still image from the camera.

        Returns:
          img: numpy array (H, W, 3) in BGR format
        """
        if PICAMERA_AVAILABLE:
            # Capture using picamera2
            raw = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV compatibility
            img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        else:
            # Capture using OpenCV
            ret, img = self.camera.read()
            if not ret:
                # Return blank image if capture fails
                h, w = self.resolution[1], self.resolution[0]
                img = np.ones((h, w, 3), dtype=np.uint8) * 255
        return img

    def preprocess(self, img):
        """
        Preprocess captured image to match training data format EXACTLY.

        Steps (mirror of src/preprocess.py process_sketch):
          1. Convert BGR to grayscale
          2. Crop whitespace (bounding box around dark content)
          3. Binarize using Otsu threshold (ink = black, paper = white)
          4. Resize to 224x224 with aspect ratio, padded on WHITE canvas
          5. Convert single channel -> 3-channel (for MobileNetV2)
          6. Normalize pixel values to [0, 1]

        NOTE: no inversion! Training data is black ink on white paper;
        inverting here would shift the input distribution and break matching.
        """
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 2: Crop whitespace (dark pixels = content)
        mask = gray < 250
        coords = np.argwhere(mask)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0) + 1
            y_min = max(0, y_min - PADDING)
            x_min = max(0, x_min - PADDING)
            y_max = min(gray.shape[0], y_max + PADDING)
            x_max = min(gray.shape[1], x_max + PADDING)
            gray = gray[y_min:y_max, x_min:x_max]

        # Step 3: Binarize (Otsu) -> black ink on white paper
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Resize with WHITE padding (matches training)
        h, w = binary.shape
        scale = min(IMG_SIZE / h, IMG_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
        y_offset = (IMG_SIZE - new_h) // 2
        x_offset = (IMG_SIZE - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Step 5: 3-channel
        rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

        # Step 6: Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Add batch dimension -> (1, 224, 224, 3)
        return np.expand_dims(normalized, axis=0)

    def release(self):
        """Release camera resources."""
        if self.camera and not PICAMERA_AVAILABLE:
            self.camera.release()
        elif self.camera and PICAMERA_AVAILABLE:
            self.camera.stop()
        print("Camera released.")

    def __del__(self):
        self.release()


def test_capture():
    """Quick test: capture + preprocess, show dimensions."""
    cam = CameraCapture()
    cam.initialize()
    print("Press SPACE to capture, ESC to quit.")
    while True:
        img = cam.capture_image()
        processed = cam.preprocess(img)
        print(f"Preprocessed shape: {processed.shape}, min={processed.min():.3f}, max={processed.max():.3f}")

        # Show preview
        preview = cv2.resize(img, (640, 480))
        cv2.imshow("Camera Preview", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            print("Captured!")
            cv2.imwrite("captured_sketch.jpg", img)
            print("Saved: captured_sketch.jpg")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_capture()
