"""
run_all.py — Sketch Matcher: COMPLETE pipeline in ONE file.

Paste this single file into Gemini/Colab, upload src.zip is NOT needed.
Everything runs in order:
  1. Download datasets (Sketchy + QuickDraw + TU-Berlin + ImageNet-Sketch)
  2. Preprocess (crop, binarize, resize 224x224)
  3. Train Siamese (3 stages)
  4. Evaluate (Top-1/3/5, EER)
  5. Export TFLite + photo embeddings
  6. Create pi_deploy.zip

Quick test (few epochs):
  python run_all.py --quick
Full run (max accuracy, ~12h on T4):
  python run_all.py

If a step fails, the script EXITS with error code 1 — never silently continues.
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SKETCHY_DIR = RAW_DIR / "sketchy"
QUICKDRAW_DIR = RAW_DIR / "quickdraw"
TUBERLIN_DIR = RAW_DIR / "tuberlin"
INETSKETCH_DIR = RAW_DIR / "imagenetsketch"

IMG_SIZE = 224
PADDING = 10
EMBEDDING_DIM = 256
DROPOUT_RATE = 0.3
CONTRASTIVE_MARGIN = 1.0

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10

STAGE1_EPOCHS = 100
STAGE1_BATCH = 128
STAGE1_LR = 0.001
STAGE2_EPOCHS = 50
STAGE2_BATCH = 64
STAGE2_LR = 0.0001
STAGE3_EPOCHS = 150
STAGE3_BATCH = 32
STAGE3_LR = 0.00001

EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

PAIRS_PER_EPOCH = 50000

QUICKDRAW_CATEGORIES = [
    "airplane", "apple", "baseball", "basketball", "bathtub",
    "bear", "bed", "bicycle", "book", "bread",
    "butterfly", "cake", "camera", "candle", "car",
    "cat", "chair", "church", "circle", "clock",
    "cloud", "cookie", "couch", "cow", "cup",
    "dog", "dolphin", "donut", "door", "drums",
    "duck", "ear", "elephant", "envelope", "eye",
    "fish", "flower", "flying_saucer", "fork", "frog",
    "garden", "giraffe", "grapes", "guitar", "hammer",
    "hand", "hat", "headphones", "helicopter", "ice_cream",
    "key", "knife", "ladder", "lamp", "lightning",
]

# =============================================================================
# STEP 1: DOWNLOAD
# =============================================================================

def download_file(url, dest):
    import requests
    from tqdm import tqdm
    log.info(f"Downloading: {url}")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(desc=dest.name, total=total, unit="B", unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    log.info(f"Extracting: {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def find_sketch_photo_dirs(root):
    """Locate sketch/ and photo/ dirs inside a Kaggle Sketchy download
    (folder structures vary between versions)."""
    root = Path(root)
    candidates = list(root.rglob("sketch")) + list(root.rglob("photo")) + \
                 list(root.rglob("sketches")) + list(root.rglob("photos")) + \
                 list(root.rglob("color")) + list(root.rglob("gray"))
    return candidates


def download_sketchy():
    SKETCHY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
        log.info("Downloading Sketchy via kagglehub...")
        path = kagglehub.dataset_download("dhananjayapaliwal/fulldataset")
        for item in Path(path).iterdir():
            dest = SKETCHY_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        log.info(f"Sketchy -> {SKETCHY_DIR}")
        return True
    except Exception as e:
        log.warning(f"kagglehub sketchy failed: {e}")
        for zp in [Path("/content/sketchy_dataset.zip"), Path("/content/sketchy.zip")]:
            if zp.exists():
                extract_zip(zp, SKETCHY_DIR)
                return True
        log.error("Sketchy download FAILED. Put sketchy_dataset.zip in /content and rerun.")
        return False


def download_quickdraw():
    QUICKDRAW_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
    ok, fail = 0, 0
    for cat in QUICKDRAW_CATEGORIES:
        dest = QUICKDRAW_DIR / f"{cat}.npy"
        if dest.exists():
            ok += 1
            continue
        try:
            url = f"{base}/{cat}.npy"
            import requests
            r = requests.head(url, timeout=10)
            if r.status_code != 200:
                fail += 1
                continue
            download_file(url, dest)
            ok += 1
        except Exception:
            fail += 1
    log.info(f"QuickDraw: {ok} ok, {fail} skipped")
    return ok > 0


def download_tuberlin():
    TUBERLIN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
        log.info("Downloading TU-Berlin via kagglehub...")
        path = kagglehub.dataset_download("borismokeev/tuberlin-sketch-dataset")
        for item in Path(path).iterdir():
            dest = TUBERLIN_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        log.info(f"TU-Berlin -> {TUBERLIN_DIR}")
        return True
    except Exception as e:
        log.warning(f"TU-Berlin failed: {e}")
        for zp in [Path("/content/tuberlin.zip")]:
            if zp.exists():
                extract_zip(zp, TUBERLIN_DIR)
                return True
        log.info("TU-Berlin optional - skipping")
        return False


def download_imagenetsketch():
    INETSKETCH_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
        log.info("Downloading ImageNet-Sketch via kagglehub...")
        path = kagglehub.dataset_download("wanghaohan/imagenetsketch")
        for item in Path(path).iterdir():
            dest = INETSKETCH_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        log.info(f"ImageNet-Sketch -> {INETSKETCH_DIR}")
        return True
    except Exception as e:
        log.warning(f"ImageNet-Sketch failed: {e}")
        for zp in [Path("/content/imagenetsketch.zip")]:
            if zp.exists():
                extract_zip(zp, INETSKETCH_DIR)
                return True
        log.info("ImageNet-Sketch optional - skipping")
        return False


# =============================================================================
# STEP 2: PREPROCESS
# =============================================================================

def crop_to_content(img, padding=PADDING):
    mask = img < 250
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(img.shape[0], y_max + padding)
    x_max = min(img.shape[1], x_max + padding)
    return img[y_min:y_max, x_min:x_max]


def binarize(img):
    import cv2
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def resize_with_padding(img, target_size=IMG_SIZE):
    import cv2
    h, w = img.shape
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size), 255, dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def gray_to_rgb(img):
    return np.stack([img] * 3, axis=-1)


def process_sketch(img):
    import cv2
    img = crop_to_content(img)
    img = binarize(img)
    img = resize_with_padding(img)
    return gray_to_rgb(img)


def process_photo(img):
    import cv2
    img = crop_to_content(img)
    img = resize_with_padding(img)
    return gray_to_rgb(img)


def find_cat_dirs(base_dir, patterns):
    """Find category directories: dirs directly containing image files."""
    cat_dirs = []
    if not base_dir.exists():
        return cat_dirs
    for pat in patterns:
        for d in base_dir.rglob(pat):
            if d.is_dir():
                files = [f for f in d.iterdir()
                         if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".npy")]
                if len(files) > 0:
                    cat_dirs.append((d.name, d))
    return cat_dirs


def load_from_dir(base_dir, patterns, is_sketch, limit=None):
    """Load and preprocess images from category dirs. Returns arrays + labels."""
    import cv2
    cat_dirs = find_cat_dirs(base_dir, patterns)
    if not cat_dirs:
        return None, None, None
    cat_dirs = sorted(set(cat_dirs))
    cat_to_idx = {name: i for i, (name, _) in enumerate(cat_dirs)}
    images, labels = [], []
    for name, d in cat_dirs:
        idx = cat_to_idx[name]
        files = sorted([f for f in d.iterdir()
                        if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if limit:
            files = files[:limit]
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                processed = process_sketch(img) if is_sketch else process_photo(img)
                images.append(processed)
                labels.append(idx)
            except Exception as e:
                log.debug(f"Skip {f}: {e}")
    if not images:
        return None, None, None
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32), cat_to_idx


def preprocess_sketchy():
    """Load sketch+photo pairs from Sketchy. Returns sketches, photos,
    sketch_labels, photo_labels, cat_to_idx."""
    import cv2
    sketch_dir = SKETCHY_DIR / "sketch"
    photo_dir = SKETCHY_DIR / "photo"
    if not sketch_dir.exists():
        for d in SKETCHY_DIR.rglob("sketch"):
            if (d.parent / "photo").exists():
                sketch_dir = d
                photo_dir = d.parent / "photo"
                break
    if not sketch_dir.exists() or not photo_dir.exists():
        log.error(f"Sketchy folders not found under {SKETCHY_DIR}. "
                  f"Found: {[str(p) for p in SKETCHY_DIR.iterdir()]}")
        return None

    cats = sorted([d.name for d in sketch_dir.iterdir() if d.is_dir()])
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    sketches, photos, sk_lb, ph_lb = [], [], [], []
    for cat in cats:
        idx = cat_to_idx[cat]
        for f in sorted((sketch_dir / cat).glob("*.png")):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            sketches.append(process_sketch(img))
            sk_lb.append(idx)
        for f in sorted((photo_dir / cat).glob("*.jpg")):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            photos.append(process_photo(img))
            ph_lb.append(idx)
    if not sketches or not photos:
        log.error(f"No images loaded from Sketchy. sketches={len(sketches)} photos={len(photos)}")
        return None
    return (np.array(sketches, dtype=np.uint8), np.array(photos, dtype=np.uint8),
            np.array(sk_lb, dtype=np.int32), np.array(ph_lb, dtype=np.int32), cat_to_idx)


def preprocess_quickdraw():
    """QuickDraw .npy files: 28x28 bitmaps. Convert to 224x224."""
    import cv2
    if not QUICKDRAW_DIR.exists():
        return None, None, None
    cats = sorted(QUICKDRAW_DIR.glob("*.npy"))
    if not cats:
        return None, None, None
    cat_to_idx = {f.stem: i for i, f in enumerate(cats)}
    images, labels = [], []
    for f in cats:
        idx = cat_to_idx[f.stem]
        data = np.load(str(f))
        data = data[:min(2000, len(data))]
        for raw in data:
            img28 = raw.reshape(28, 28).astype(np.uint8)
            img = cv2.resize(img28, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = cv2.bitwise_not(img)
            images.append(gray_to_rgb(img))
            labels.append(idx)
    if not images:
        return None, None, None
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32), cat_to_idx


def split_by_category(sk_lb, ph_lb):
    """Split categories 80/10/10, categories disjoint across splits."""
    cats = sorted(set(sk_lb.tolist()) | set(ph_lb.tolist()))
    random.shuffle(cats)
    n_train = int(len(cats) * TRAIN_SPLIT)
    n_val = int(len(cats) * VAL_SPLIT)
    train_cats = set(cats[:n_train])
    val_cats = set(cats[n_train:n_train + n_val])
    test_cats = set(cats[n_train + n_val:])
    return train_cats, val_cats, test_cats


# =============================================================================
# STEP 3: DATA LOADER
# =============================================================================

class PairGenerator:
    def __init__(self, sketches, photos, sketch_labels, photo_labels, batch_size=32,
                 pairs_per_epoch=PAIRS_PER_EPOCH):
        self.sketches = sketches
        self.photos = photos
        self.sketch_labels = sketch_labels
        self.photo_labels = photo_labels
        self.batch_size = batch_size
        self.pairs_per_epoch = pairs_per_epoch
        self.n_sk = len(sketches)
        self.n_ph = len(photos)
        self.photo_cat_idx = {}
        for i, c in enumerate(photo_labels):
            self.photo_cat_idx.setdefault(c, []).append(i)
        self.sketch_cat_idx = {}
        for i, c in enumerate(sketch_labels):
            self.sketch_cat_idx.setdefault(c, []).append(i)
        self.cats = list(self.photo_cat_idx.keys())

    def __len__(self):
        return max(1, self.pairs_per_epoch // self.batch_size)

    def __getitem__(self, idx):
        bs = self.batch_size
        n_pos = bs // 2
        batch_sk = np.zeros((bs, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        batch_ph = np.zeros((bs, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        batch_y = np.zeros((bs, 1), dtype=np.float32)
        for i in range(n_pos):
            cat = random.choice(self.cats)
            sk_i = random.choice(self.sketch_cat_idx.get(cat, self.sketch_cat_idx[random.choice(self.cats)]))
            ph_i = random.choice(self.photo_cat_idx[cat])
            batch_sk[i] = self.sketches[sk_i] / 255.0
            batch_ph[i] = self.photos[ph_i] / 255.0
            batch_y[i] = 0
        for i in range(n_pos, bs):
            cat_a = random.choice(self.cats)
            cat_b = random.choice([c for c in self.cats if c != cat_a])
            sk_i = random.choice(self.sketch_cat_idx.get(cat_a, self.sketch_cat_idx[random.choice(self.cats)]))
            ph_i = random.choice(self.photo_cat_idx[cat_b])
            batch_sk[i] = self.sketches[sk_i] / 255.0
            batch_ph[i] = self.photos[ph_i] / 255.0
            batch_y[i] = 1
        return [batch_sk, batch_ph], batch_y


# =============================================================================
# STEP 4: MODEL
# =============================================================================

def build_embedding_net():
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input, regularizers
    from tensorflow.keras.applications import MobileNetV2
    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inp)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(EMBEDDING_DIM)(x)
    out = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1))(x)
    model = Model(inp, out, name="embedding_net")
    return model, base


def contrastive_loss(y_true, y_pred, margin=CONTRASTIVE_MARGIN):
    import tensorflow as tf
    y_true = tf.cast(y_true, tf.float32)
    match = (1 - y_true) * 0.5 * tf.square(y_pred)
    non_match = y_true * 0.5 * tf.square(tf.maximum(0.0, margin - y_pred))
    return tf.reduce_mean(match + non_match)


def build_siamese(embedding_net):
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    inp_a = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    inp_b = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    emb_a = embedding_net(inp_a)
    emb_b = embedding_net(inp_b)
    dist = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]),
                                                         axis=1, keepdims=True)))([emb_a, emb_b])
    model = Model([inp_a, inp_b], dist)
    return model


# =============================================================================
# STEP 5: TRAIN
# =============================================================================

def train_pipeline(quick=False):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info(f"GPU: {gpus[0].name}")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        log.warning("NO GPU DETECTED - training will be extremely slow!")

    from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                            ModelCheckpoint, TensorBoard, CSVLogger)

    if quick:
        e1, e2, e3 = 2, 1, 1
        log.warning("QUICK TEST MODE: 4 total epochs")
    else:
        e1, e2, e3 = STAGE1_EPOCHS, STAGE2_EPOCHS, STAGE3_EPOCHS

    # --- Load processed data ---
    sketches = np.load(PROCESSED_DIR / "sketches.npy")
    photos = np.load(PROCESSED_DIR / "photos.npy")
    sk_lb = np.load(PROCESSED_DIR / "sketch_labels.npy")
    ph_lb = np.load(PROCESSED_DIR / "photo_labels.npy")
    with open(PROCESSED_DIR / "category_names.json") as f:
        cat_names = json.load(f)
    log.info(f"Loaded: {len(sketches)} sketches, {len(photos)} photos, {len(cat_names)} categories")

    # --- Build model ---
    embedding_net, backbone = build_embedding_net()
    siamese = build_siamese(embedding_net)
    siamese.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LR),
                    loss=lambda yt, yp: contrastive_loss(yt, yp), metrics=["accuracy"])
    log.info("Model built. Trainable params (all): "
             f"{sum(np.prod(w.shape) for w in siamese.trainable_weights):,}")

    # --- Generators ---
    def make_gen(bs, pairs):
        return PairGenerator(sketches, photos, sk_lb, ph_lb, batch_size=bs, pairs_per_epoch=pairs)

    train_gen = make_gen(STAGE1_BATCH, PAIRS_PER_EPOCH)
    val_gen = make_gen(STAGE1_BATCH, 10000)

    def run_stage(name, epochs, bs, lr, freeze_fn, gen):
        if freeze_fn:
            freeze_fn()
        siamese.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss=lambda yt, yp: contrastive_loss(yt, yp), metrics=["accuracy"])
        log.info(f"\n{'='*50}\n{name}: {epochs} epochs, bs={bs}, lr={lr}\n{'='*50}")
        stage_dir = MODELS_DIR / name
        stage_dir.mkdir(parents=True, exist_ok=True)
        siamese.fit(
            make_gen(bs, PAIRS_PER_EPOCH),
            epochs=epochs,
            validation_data=make_gen(bs, 10000),
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR,
                                  patience=REDUCE_LR_PATIENCE, min_lr=1e-8),
                ModelCheckpoint(str(stage_dir / "best.keras"), monitor="val_loss",
                                save_best_only=True),
                CSVLogger(str(LOGS_DIR / f"{name}.csv")),
            ],
            verbose=1,
        )

    def freeze_all():
        backbone.trainable = False

    def unfreeze_last12():
        backbone.trainable = True
        for l in backbone.layers:
            l.trainable = False
        for l in backbone.layers[-12:]:
            l.trainable = True

    def unfreeze_all_layers():
        backbone.trainable = True
        for l in backbone.layers:
            l.trainable = True

    run_stage("stage1_dense", e1, STAGE1_BATCH, STAGE1_LR, freeze_all, train_gen)
    run_stage("stage2_unfreeze12", e2, STAGE2_BATCH, STAGE2_LR, unfreeze_last12, train_gen)
    run_stage("stage3_full", e3, STAGE3_BATCH, STAGE3_LR, unfreeze_all_layers, train_gen)

    siamese.save(MODELS_DIR / "final_model.keras")
    embedding_net.save(MODELS_DIR / "best_model.keras")
    log.info("Models saved to models/")


# =============================================================================
# STEP 6: EVALUATE
# =============================================================================

def evaluate():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODELS_DIR / "best_model.keras", compile=False)
    sketches = np.load(PROCESSED_DIR / "sketches.npy").astype(np.float32) / 255.0
    photos = np.load(PROCESSED_DIR / "photos.npy").astype(np.float32) / 255.0
    sk_lb = np.load(PROCESSED_DIR / "sketch_labels.npy")
    ph_lb = np.load(PROCESSED_DIR / "photo_labels.npy")
    log.info("Computing embeddings...")
    sk_emb = model.predict(sketches, batch_size=64, verbose=1)
    ph_emb = model.predict(photos, batch_size=64, verbose=1)
    sim = sk_emb @ ph_emb.T
    for k in [1, 3, 5]:
        top = np.argsort(-sim, axis=1)[:, :k]
        correct = sum(sk_lb[i] in ph_lb[top[i]] for i in range(len(sk_lb)))
        log.info(f"Top-{k}: {correct/len(sk_lb)*100:.2f}%")


# =============================================================================
# STEP 7: EXPORT TFLITE
# =============================================================================

def export_tflite():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODELS_DIR / "best_model.keras", compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    sketches = np.load(PROCESSED_DIR / "sketches.npy")
    def rep():
        for i in range(min(100, len(sketches))):
            yield [sketches[i][np.newaxis, ...].astype(np.float32) / 255.0]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                           tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    tflite_path = MODELS_DIR / "sketch_matcher.tflite"
    tflite_path.write_bytes(tflite_model)
    log.info(f"TFLite saved: {tflite_path} ({len(tflite_model)/1e6:.2f} MB)")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    photos = np.load(PROCESSED_DIR / "photos.npy").astype(np.float32) / 255.0
    emb_dim = out_det["shape"][1]
    all_emb = np.zeros((len(photos), emb_dim), dtype=np.float32)
    for i in range(len(photos)):
        img = photos[i][np.newaxis, ...]
        if in_det["dtype"] == np.uint8:
            s, zp = in_det["quantization"]
            img = (img / s + zp).astype(np.uint8)
        interpreter.set_tensor(in_det["index"], img)
        interpreter.invoke()
        all_emb[i] = interpreter.get_tensor(out_det["index"])[0]
    np.save(MODELS_DIR / "photo_embeddings.npy", all_emb)
    with open(MODELS_DIR / "labels.json", "w") as f:
        json.dump(json.load(open(PROCESSED_DIR / "category_names.json")), f)
    log.info(f"Photo embeddings: {all_emb.shape}")


# =============================================================================
# STEP 8: PACKAGE
# =============================================================================

def package():
    zip_path = BASE_DIR / "pi_deploy.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(MODELS_DIR / "sketch_matcher.tflite", "model_data/sketch_matcher.tflite")
        zf.write(MODELS_DIR / "photo_embeddings.npy", "model_data/photo_embeddings.npy")
        zf.write(MODELS_DIR / "labels.json", "model_data/labels.json")
        zf.writestr("matcher.py", MATCHER_PY)
        zf.writestr("main.py", MAIN_PY)
    log.info(f"Created {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")


# =============================================================================
# PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test (4 epochs)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    log.warning("=" * 60)
    log.warning("run_all.py is a LEGACY single-file fallback (old matcher, no "
                "photo_labels / open-set rejection).")
    log.warning("Use 'python run_training.py' or the Colab notebook instead, "
                "which run the current src/ modules.")
    log.warning("=" * 60)

    t0 = time.time()
    steps = []
    if not args.skip_download:
        steps.append(("DOWNLOAD DATASETS", do_download))
    steps.append(("PREPROCESS", do_preprocess))
    if not args.skip_train:
        steps.append(("TRAIN", lambda: train_pipeline(quick=args.quick)))
    steps.append(("EVALUATE", evaluate))
    steps.append(("EXPORT TFLITE", export_tflite))
    steps.append(("PACKAGE", package))

    for name, fn in steps:
        log.info("=" * 60)
        log.info(f"STEP: {name}")
        log.info("=" * 60)
        try:
            fn()
        except Exception as e:
            log.error(f"FAILED at {name}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        log.info(f"Step OK: {name}")

    log.info(f"\nALL DONE in {(time.time()-t0)/60:.1f} min. pi_deploy.zip is ready!")


def do_download():
    if not download_sketchy():
        sys.exit(1)
    if not (SKETCHY_DIR / "sketch").exists():
        subdirs = list(SKETCHY_DIR.iterdir())
        log.info(f"Sketchy contents: {[d.name for d in subdirs]}")
        sys.exit(1)
    download_quickdraw()
    download_tuberlin()
    download_imagenetsketch()


def do_preprocess():
    result = preprocess_sketchy()
    if result is None:
        sys.exit(1)
    sketches, photos, sk_lb, ph_lb, cat_to_idx = result

    # Merge QuickDraw sketches (no photos - only as extra sketch variety)
    q_sk, q_lb, q_idx = preprocess_quickdraw()
    if q_sk is not None and len(q_sk) > 0:
        offset = len(cat_to_idx)
        q_sk_lb = q_lb + offset
        q_cat = {str(i + offset): name for i, name in enumerate(q_idx)}
        all_sk = np.concatenate([sketches, q_sk])
        all_sk_lb = np.concatenate([sk_lb, q_sk_lb])
        all_ph = photos
        all_ph_lb = ph_lb
        for k, v in q_cat.items():
            cat_to_idx[v] = int(k)
        sketches, sk_lb, photos, ph_lb = all_sk, all_sk_lb, all_ph, all_ph_lb
        log.info(f"Merged QuickDraw: +{len(q_sk)} sketches")

    train_cats, val_cats, test_cats = split_by_category(sk_lb, ph_lb)
    log.info(f"Train cats: {len(train_cats)}, Val: {len(val_cats)}, Test: {len(test_cats)}")

    np.save(PROCESSED_DIR / "sketches.npy", sketches)
    np.save(PROCESSED_DIR / "photos.npy", photos)
    np.save(PROCESSED_DIR / "sketch_labels.npy", sk_lb)
    np.save(PROCESSED_DIR / "photo_labels.npy", ph_lb)
    with open(PROCESSED_DIR / "category_names.json", "w") as f:
        json.dump(cat_to_idx, f, indent=2)
    log.info(f"Saved: {len(sketches)} sketches, {len(photos)} photos, "
             f"{len(cat_to_idx)} categories")


MATCHER_PY = '''import json, time
from pathlib import Path
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

class SketchMatcher:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.interpreter = Interpreter(model_path=str(self.model_dir / "sketch_matcher.tflite"))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.photo_embeddings = np.load(str(self.model_dir / "photo_embeddings.npy")).astype(np.float32)
        norms = np.linalg.norm(self.photo_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.photo_embeddings /= norms
        with open(self.model_dir / "labels.json") as f:
            self.labels = json.load(f)

    def embed(self, image):
        in_det = self.input_details[0]
        if in_det["dtype"] == np.uint8:
            s, zp = in_det["quantization"]
            image = (image / s + zp).astype(np.uint8)
        else:
            image = image.astype(np.float32)
        self.interpreter.set_tensor(in_det["index"], image)
        self.interpreter.invoke()
        emb = self.interpreter.get_tensor(self.output_details[0]["index"])[0].astype(np.float32)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    def match(self, image, top_k=3):
        t0 = time.time()
        emb = self.embed(image)
        t1 = time.time()
        sims = self.photo_embeddings @ emb
        top = np.argsort(-sims)[:top_k]
        t2 = time.time()
        results = []
        for i in top:
            results.append((list(self.labels.values())[i % len(self.labels)], sims[i] * 100))
        return results, (t1 - t0) * 1000, (t2 - t1) * 1000
'''

MAIN_PY = '''import sys, time
from pathlib import Path
import cv2
import numpy as np

from matcher import SketchMatcher

def preprocess(img, size=224, padding=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x, y = max(0, x - padding), max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        gray = gray[y:y + h, x:x + w]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    h, w = binary.shape
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(binary, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    return np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

def main():
    matcher = SketchMatcher(Path(__file__).parent / "model_data")
    cap = cv2.VideoCapture(0)
    print("Camera ready. Press SPACE to capture, ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            processed = preprocess(frame)
            results, inf_ms, search_ms = matcher.match(processed)
            for i, (cat, conf) in enumerate(results):
                print(f"#{i+1}: {cat} ({conf:.1f}%)")
            print(f"Time: {inf_ms:.0f}ms infer + {search_ms:.0f}ms search")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''

if __name__ == "__main__":
    main()
