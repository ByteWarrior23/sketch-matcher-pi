"""LOCAL stopgap: prepare RPS images, train a small CNN on CPU, export TFLite.

Purpose: test the full laptop-camera loop NOW while the real model trains on
the HPC. Uses 128x128 for speed. Outputs models/local/rps_local.tflite.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

from config import CLASSES, DATA_DIR, RANDOM_SEED  # noqa: E402

IMG = 128
RAW = ROOT / "data" / "raw"
OUT = ROOT / "models" / "local"
OUT.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_dir(directory, label_index):
    imgs, lbls = [], []
    for p in sorted(directory.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
        imgs.append(img)
        lbls.append(label_index)
    return imgs, lbls


def build(split_dir, class_names):
    x, y = [], []
    for idx, name in enumerate(class_names):
        d = split_dir / name
        imgs, lbls = load_dir(d, idx)
        print(f"  {name}: {len(imgs)}", flush=True)
        x.extend(imgs)
        y.extend(lbls)
    return np.stack(x).astype(np.uint8), np.asarray(y, dtype=np.int64)


def main():
    t0 = time.time()
    x_train, y_train = build(RAW / "rps" / "rps", CLASSES)
    x_test, y_test = build(RAW / "rps-test-set" / "rps-test-set", CLASSES)

    n = len(y_train)
    idx = np.random.permutation(n)
    n_val = int(0.2 * n)
    x_val, y_val = x_train[idx[:n_val]], y_train[idx[:n_val]]
    x_train, y_train = x_train[idx[n_val:]], y_train[idx[n_val:]]
    print(f"train {x_train.shape} val {x_val.shape} test {x_test.shape}", flush=True)

    model = tf.keras.Sequential([
        layers.Input(shape=(IMG, IMG, 3)),
        layers.Rescaling(1.0 / 127.5, offset=-1.0),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPool2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPool2D(2),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPool2D(2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              batch_size=32, epochs=6, verbose=2)

    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"stopgap val acc {acc:.4f}", flush=True)

    # export float32 TFLite (inputs float [0,255])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    (OUT / "rps_local.tflite").write_bytes(tflite_model)
    with open(OUT / "labels.json", "w") as f:
        json.dump(CLASSES, f)
    print(f"exported {OUT/'rps_local.tflite'} ({len(tflite_model)} bytes)", flush=True)

    # verify int via interpreter
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    correct = 0
    for i in range(len(x_test)):
        interp.set_tensor(inp["index"], x_test[i:i + 1].astype(np.float32))
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        if int(y.argmax()) == int(y_test[i]):
            correct += 1
    acc_test = correct / len(x_test)
    print(f"stopgap TEST acc {acc_test:.4f} ({correct}/{len(x_test)})  in {time.time()-t0:.0f}s", flush=True)
    with open(OUT / "test_accuracy.json", "w") as f:
        json.dump({"test_accuracy": float(acc_test)}, f)
    print("LOCAL OK", flush=True)


if __name__ == "__main__":
    main()
