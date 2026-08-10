"""TFLite int8 classifier for Rock-Paper-Scissors. Input: uint8 [0,255] RGB."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import EXPORT_DIR, IMG_SIZE

import cv2  # noqa: E402

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    try:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
        _tf = tf
    except ImportError:
        raise ImportError("need tensorflow or tflite-runtime")

from config import EXPORT_DIR  # noqa: E402


class RPSClassifier:
    def __init__(self, model_dir=None):
        model_dir = Path(model_dir) if model_dir else EXPORT_DIR
        tflite = model_dir / "rps_model.tflite"
        if not tflite.exists():
            cands = sorted(model_dir.glob("*.tflite"))
            if not cands:
                raise FileNotFoundError(f"no .tflite in {model_dir}")
            tflite = cands[0]
        self.tflite_path = tflite
        self.labels_path = model_dir / "labels.json"
        kwargs = {}
        if "_tf" in globals() and hasattr(_tf.lite.experimental, "OpResolverType"):
            # int8 MobileNetV3 can break XNNPACK; fall back to builtin kernels
            kwargs["experimental_op_resolver_type"] = _tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        self.interp = TFLiteInterpreter(model_path=str(self.tflite_path), **kwargs)
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()[0]
        self.output_details = self.interp.get_output_details()[0]
        self.s_in, self.z_in = self.input_details["quantization"]
        self.s_out, self.z_out = self.output_details["quantization"]
        with open(self.labels_path) as f:
            self.labels = json.load(f)
        self.input_size = int(self.input_details["shape"][1])
        self._verify_input_shape()

    def _verify_input_shape(self):
        shape = self.input_details["shape"]
        assert shape[1] == shape[2] == 128 or shape[1] == shape[2] == 224, \
            f"unexpected input {shape}"

    def preprocess(self, bgr_frame):
        img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        if self.s_in == 0:  # float32 model: expect [0,255] float input
            return img[None, ...].astype(np.float32)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        q = (img.astype(np.float32) / self.s_in + self.z_in).round().astype(np.uint8)
        return q[None, ...]

    def predict_proba(self, bgr_frame):
        x = self.preprocess(bgr_frame)
        self.interp.set_tensor(self.input_details["index"], x)
        self.interp.invoke()
        yq = self.interp.get_tensor(self.output_details["index"]).astype(np.float32)
        if self.s_out != 0:
            yq = (yq - self.z_out) * self.s_out
        p = np.maximum(yq, 0)
        s = p.sum()
        if s > 0:
            p = p / s
        return p[0]

    def classify(self, bgr_frame):
        p = self.predict_proba(bgr_frame)
        idx = int(p.argmax())
        return self.labels[idx], float(p[idx]), p
