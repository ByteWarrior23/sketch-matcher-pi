"""Export the trained RPS model to TFLite int8 + labels + verify int8 accuracy.

Produces pi_deploy/model_data/rps_model.tflite and labels.json. The int8 model
takes uint8 [0,255] input (matches training input space).
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_DIR, EXPORT_DIR, MODELS_DIR, NUM_CLASSES

print("TF:", tf.__version__, "GPU:", tf.config.list_physical_devices("GPU"), flush=True)

model = tf.keras.models.load_model(MODELS_DIR / "rps_model.keras", safe_mode=False)

# int8 quantization with a representative set from training data
X_TR = np.load(DATA_DIR / "rps_train_x.npy")

def rep_data():
    n = X_TR.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=200, replace=False)
    for i in idx:
        yield X_TR[i:i + 1]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
(EXPORT_DIR / "rps_model.tflite").write_bytes(tflite_model)
with open(EXPORT_DIR / "labels.json", "w") as f:
    json.dump(["rock", "paper", "scissors"], f)
print(f"tflite bytes: {len(tflite_model)} -> {EXPORT_DIR / 'rps_model.tflite'}", flush=True)

# Verify int8 accuracy on the test set
X_TE = np.load(DATA_DIR / "rps_test_x.npy")
Y_TE = np.load(DATA_DIR / "rps_test_y.npy")

interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
s_in, z_in = inp["quantization"]
s_out, z_out = out["quantization"]

def to_input(imgs):
    x = np.clip(imgs.astype(np.float32), 0, 255)
    q = (x / s_in + z_in).round().astype(np.uint8)
    return q

correct = 0
for i in range(X_TE.shape[0]):
    if i % 100 == 0:
        print(f"  int8 eval {i}/{X_TE.shape[0]}", flush=True)
    interp.set_tensor(inp["index"], to_input(X_TE[i:i + 1]))
    interp.invoke()
    yq = interp.get_tensor(out["index"]).astype(np.float32)
    y = (yq - z_out) * s_out
    if int(y.argmax()) == int(Y_TE[i]):
        correct += 1
acc = correct / X_TE.shape[0]
print(f"int8 test accuracy: {acc:.4f} ({correct}/{X_TE.shape[0]})", flush=True)

with open(EXPORT_DIR / "int8_test_accuracy.json", "w") as f:
    json.dump({"int8_test_accuracy": float(acc), "n": int(X_TE.shape[0])}, f)
print("EXPORT OK", flush=True)
