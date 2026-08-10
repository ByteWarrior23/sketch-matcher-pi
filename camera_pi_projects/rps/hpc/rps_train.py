"""Train a Rock-Paper-Scissors classifier (MobileNetV3Large) on the HPC.

3 stages: frozen backbone (head only) -> unfreeze last 12 -> full fine-tune.
Saves models/rps_model.keras + best-checkpoint. Input space is [0,255] so the
keras-applications built-in rescale matches TFLite int8 uint8 deployment.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import (BACKBONE, BATCH_SIZE, DATA_DIR, EPOCHS_STAGE1,
                    EPOCHS_STAGE2, EPOCHS_STAGE3, LR_STAGE1, LR_STAGE2,
                    LR_STAGE3, MODELS_DIR, NUM_CLASSES, PATIENCE, RANDOM_SEED,
                    IMG_SIZE)

print("TF:", tf.__version__, "GPU:", tf.config.list_physical_devices("GPU"), flush=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X_TR = np.load(DATA_DIR / "rps_train_x.npy").astype(np.float32)
Y_TR = np.load(DATA_DIR / "rps_train_y.npy")
X_VA = np.load(DATA_DIR / "rps_val_x.npy").astype(np.float32)
Y_VA = np.load(DATA_DIR / "rps_val_y.npy")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = MODELS_DIR / "rps_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def build():
    backbone = getattr(keras.applications, BACKBONE)(
        weights="imagenet", include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg", classes=NUM_CLASSES)
    x = layers.Dropout(0.2)(backbone.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
    model = keras.Model(backbone.input, out)
    return model, backbone


def run_stage(model, backbone, trainable_last, epochs, lr, tag, patience):
    if trainable_last is None:
        backbone.trainable = False
        for layer in backbone.layers:
            layer.trainable = False
        print(f"[{tag}] backbone frozen", flush=True)
    else:
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = False
        for layer in backbone.layers[trainable_last:]:
            layer.trainable = True
        n_unfrozen = len(backbone.layers[trainable_last:])
        print(f"[{tag}] unfroze {n_unfrozen} layers (from index {trainable_last})", flush=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    ckpt = keras.callbacks.ModelCheckpoint(
        CKPT_DIR / "best.keras", monitor="val_accuracy", save_best_only=True,
        save_weights_only=False, verbose=1)
    early = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                          restore_best_weights=True)
    hist = model.fit(
        X_TR, Y_TR, validation_data=(X_VA, Y_VA),
        batch_size=BATCH_SIZE, epochs=epochs,
        callbacks=[ckpt, early], verbose=2)
    best = max(hist.history["val_accuracy"])
    print(f"[{tag}] best val acc {best:.4f}", flush=True)
    return best


def main():
    model, backbone = build()
    print(f"params: {model.count_params():,}", flush=True)

    run_stage(model, backbone, None, EPOCHS_STAGE1, LR_STAGE1, "stage1", PATIENCE)
    run_stage(model, backbone, -12, EPOCHS_STAGE2, LR_STAGE2, "stage2", PATIENCE)
    run_stage(model, backbone, 0, EPOCHS_STAGE3, LR_STAGE3, "stage3", PATIENCE)

    # final: reload best checkpoint and save as the canonical model
    model.load_weights(CKPT_DIR / "best.keras")
    model.save(MODELS_DIR / "rps_model.keras")
    print("saved models/rps_model.keras", flush=True)

    loss, acc = model.evaluate(X_VA, Y_VA, verbose=0)
    print(f"final val loss={loss:.4f} acc={acc:.4f}", flush=True)
    print("TRAIN OK", flush=True)


if __name__ == "__main__":
    main()
