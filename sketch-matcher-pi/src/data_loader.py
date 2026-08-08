"""
data_loader.py

Generates Siamese training pairs on-the-fly (+ optional distillation targets).

Pair types:
  - Positive (label=0): sketch + photo of SAME category (pull together)
  - Negative (label=1): sketch + photo of DIFFERENT category (push apart)

Improvements over the previous generator:
  1. FIXED: photo indices now come from a separate photo-label->index map
     (the old version indexed photos with sketch-array indices -> bug).
  2. Hard-negative sampling: a fraction of negatives are drawn from
     CONFUSABLE_PAIRS categories so the model learns fine discriminations.
  3. On-the-fly augmentation (sketch rotation/stroke thickness, photo
     brightness/contrast) for regularization.
  4. Optional teacher embeddings -> distillation generator yields extra
     y-targets [distance_label, teacher_sketch_emb, teacher_photo_emb].

Note: when distillation is on, augmentation should stay off so teacher
targets (computed on the raw images) match student inputs (train.py handles
this; it simply doesn't enable augment).
"""

import json
import random

import numpy as np
from tensorflow.keras.utils import Sequence

try:
    from config import (PROCESSED_DIR, HARD_NEGATIVE_RATIO, CONFUSABLE_PAIRS,
                        IMG_SIZE, USE_ARC_FACE)
except ModuleNotFoundError:  # Colab: imported as src.data_loader
    from src.config import (PROCESSED_DIR, HARD_NEGATIVE_RATIO, CONFUSABLE_PAIRS,
                            IMG_SIZE, USE_ARC_FACE)

import cv2

# The node watchdog SIGKILLs jobs that oversubscribe the ncpus PBS granted
# (cv2's internal thread pool otherwise scales to all 128 physical cores).
cv2.setNumThreads(1)


# =============================================================================
# AUGMENTATION  (operates on float32 [0,1] images)
# =============================================================================
def _to_uint8(x):
    if x.dtype == np.uint8:
        return x
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _to_float(u):
    return u.astype(np.float32) / 255.0


def augment_sketch(img):
    """Sketch augmentation: small rotation + stroke thickness jitter."""
    u = _to_uint8(img)

    angle = random.uniform(-5.0, 5.0)
    if abs(angle) > 0.5:
        h, w = u.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        u = cv2.warpAffine(u, M, (w, h), borderValue=255)

    kernel_size = random.choice([None, None, None, 3, 5])
    if kernel_size is not None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() < 0.5:
            u = cv2.erode(u, kernel, iterations=1)   # thinner strokes
        else:
            u = cv2.dilate(u, kernel, iterations=1)  # thicker strokes

    return _to_float(u)


def augment_photo(img):
    """Mild photo augmentation: brightness/contrast jitter + tiny rotation."""
    u = _to_uint8(img)

    if random.random() < 0.5:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-20, 20)
        u = cv2.convertScaleAbs(u, alpha=alpha, beta=beta)

    angle = random.uniform(-4.0, 4.0)
    if abs(angle) > 0.5:
        h, w = u.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        u = cv2.warpAffine(u, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    return _to_float(u)


# =============================================================================
# GENERATOR
# =============================================================================
class SketchPhotoPairGenerator(Sequence):
    """
    Yields (inputs, outputs).

      inputs :
        standard:            [sketch_batch (B,224,224,3), photo_batch (B,224,224,3)]
        arcface mode:        [..., sk_cat_batch (B,), ph_cat_batch (B,)]
      outputs:
        standard:                label_batch (B,1)              (0=match, 1=no match)
        distillation mode:       [label_batch, teacher_sk, teacher_ph]
        arcface mode adds:       [..., sk_cat_batch (B,), ph_cat_batch (B,)]
          (category ids serve as BOTH the ArcFace head's forward-pass inputs
          and the sparse-CE targets, per the standard Keras ArcFace pattern)

    sketch_labels / photo_labels: category index (int) per image.
    """

    def __init__(self, sketches, photos, sketch_labels, photo_labels,
                 category_names, batch_size=32, pairs_per_epoch=10000,
                 shuffle=True, augment=False, teacher_sketch_embs=None,
                 teacher_photo_embs=None, hard_negative_ratio=0.0,
                 confusable_pairs=None, sk_global=None, ph_global=None,
                 arcface=False):
        super().__init__()
        # Full arrays are passed as MEMORY-MAPPED reads (mmap_mode='r') so no
        # 100+ GB slice is ever materialized in RAM (workq mem cap is 32 GB).
        # sk_global/ph_global map LOCAL (sliced) row index -> GLOBAL row index.
        self.sketches = sketches
        self.photos = photos
        self.sk_global = (np.arange(len(sketch_labels)) if sk_global is None
                          else np.asarray(sk_global, dtype=np.int64))
        self.ph_global = (np.arange(len(photo_labels)) if ph_global is None
                          else np.asarray(ph_global, dtype=np.int64))
        self.sk_labels = np.asarray(sketch_labels, dtype=np.int64)
        self.ph_labels = np.asarray(photo_labels, dtype=np.int64)
        self.category_names = category_names  # {idx: name}

        self.batch_size = batch_size
        self.pairs_per_epoch = pairs_per_epoch
        self.shuffle = shuffle
        self.augment = augment

        self.teacher_sketch_embs = teacher_sketch_embs
        self.teacher_photo_embs = teacher_photo_embs
        self.distill = (teacher_sketch_embs is not None
                        and teacher_photo_embs is not None)

        self.arcface = arcface
        self.hard_negative_ratio = hard_negative_ratio

        self.sk_cat_map = self._build_index(self.sk_labels)
        self.ph_cat_map = self._build_index(self.ph_labels)

        # Only categories with BOTH sketch and photo can form pairs
        self.valid_cats = sorted(set(self.sk_cat_map) & set(self.ph_cat_map))
        if not self.valid_cats:
            raise ValueError("No overlap between sketch and photo categories!")

        self.confusable_pairs = None
        if confusable_pairs:
            self.confusable_pairs = self._resolve_confusable(confusable_pairs)

    def _build_index(self, labels):
        idx_map = {}
        for i, cat in enumerate(labels):
            idx_map.setdefault(int(cat), []).append(i)
        return idx_map

    def _resolve_confusable(self, confusable_pairs):
        """Map confusable-pair NAMES -> category indices present in this set."""
        known = {}
        for cat_idx in self.valid_cats:
            name = self.category_names.get(int(cat_idx))
            if name:
                known.setdefault(name, int(cat_idx))

        resolved = {}
        for name, others in confusable_pairs.items():
            idx = known.get(name)
            if idx is None:
                continue
            mapped = [known[o] for o in others if o in known]
            if mapped:
                resolved[idx] = mapped
        return resolved

    def __len__(self):
        return int(np.ceil(self.pairs_per_epoch / self.batch_size))

    def _pick_negative_cat(self, cat_a):
        if (self.hard_negative_ratio > 0 and self.confusable_pairs
                and cat_a in self.confusable_pairs
                and random.random() < self.hard_negative_ratio):
            return random.choice(self.confusable_pairs[cat_a])
        others = [c for c in self.valid_cats if c != cat_a]
        return random.choice(others) if others else None

    def _norm(self, img):
        """uint8 data -> float32 [0,1]; float32 passes through unchanged."""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img

    def _aug_sketch(self, img):
        if not self.augment:
            return self._norm(img)
        return augment_sketch(img)

    def _aug_photo(self, img):
        if not self.augment:
            return self._norm(img)
        return augment_photo(img)

    def __getitem__(self, idx):
        n = self.batch_size
        n_pos = int(n * 0.5)
        n_neg = n - n_pos

        batch_sk = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        batch_ph = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        labels = np.zeros((n, 1), dtype=np.float32)
        t_sk = t_ph = None
        if self.distill:
            emb_dim = self.teacher_sketch_embs.shape[1]
            t_sk = np.empty((n, emb_dim), dtype=np.float32)
            t_ph = np.empty((n, emb_dim), dtype=np.float32)
        sk_cats = np.zeros((n,), dtype=np.int64)
        ph_cats = np.zeros((n,), dtype=np.int64)

        k = 0

        # Positive pairs (label 0)
        for _ in range(n_pos):
            cat = random.choice(self.valid_cats)
            si = random.choice(self.sk_cat_map[cat])
            pi = random.choice(self.ph_cat_map[cat])
            batch_sk[k] = self._aug_sketch(self.sketches[self.sk_global[si]])
            batch_ph[k] = self._aug_photo(self.photos[self.ph_global[pi]])
            labels[k] = 0.0
            sk_cats[k] = cat
            ph_cats[k] = cat
            if self.distill:
                t_sk[k] = self.teacher_sketch_embs[si]
                t_ph[k] = self.teacher_photo_embs[pi]
            k += 1

        # Negative pairs (label 1)
        for _ in range(n_neg):
            cat_a = random.choice(self.valid_cats)
            cat_b = self._pick_negative_cat(cat_a)
            if cat_b is None:
                continue
            si = random.choice(self.sk_cat_map[cat_a])
            pi = random.choice(self.ph_cat_map[cat_b])
            batch_sk[k] = self._aug_sketch(self.sketches[self.sk_global[si]])
            batch_ph[k] = self._aug_photo(self.photos[self.ph_global[pi]])
            labels[k] = 1.0
            sk_cats[k] = cat_a
            ph_cats[k] = cat_b
            if self.distill:
                t_sk[k] = self.teacher_sketch_embs[si]
                t_ph[k] = self.teacher_photo_embs[pi]
            k += 1

        if k < n:
            batch_sk = batch_sk[:k]
            batch_ph = batch_ph[:k]
            labels = labels[:k]
            sk_cats = sk_cats[:k]
            ph_cats = ph_cats[:k]
            if self.distill:
                t_sk = t_sk[:k]
                t_ph = t_ph[:k]

        inputs = (batch_sk, batch_ph)
        if self.arcface:
            inputs = (batch_sk, batch_ph, sk_cats, ph_cats)

        if self.distill:
            outputs = (labels, t_sk, t_ph)
            if self.arcface:
                outputs = (labels, t_sk, t_ph, sk_cats, ph_cats)
            return inputs, outputs
        if self.arcface:
            return inputs, (labels, sk_cats, ph_cats)
        return inputs, labels

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.valid_cats)


# =============================================================================
# LOADING + SPLIT MASKS
# =============================================================================
def _norm_batch(batch):
    """uint8 -> float32 [0,1]; float32 passes through unchanged."""
    if batch.dtype == np.uint8:
        return batch.astype(np.float32) / 255.0
    return batch


def predict_normalized(model, images, batch_size=128, verbose=1):
    """
    Predict embeddings over a (possibly uint8, memory-mapped) image array,
    normalizing each chunk to float32 [0,1] before feeding the model.
    """
    n = len(images)

    def gen():
        for i in range(0, n, batch_size):
            yield (_norm_batch(images[i:i + batch_size]),)

    steps = -(-n // batch_size)
    return model.predict(gen(), steps=steps, verbose=verbose)


def load_processed_data():
    # Prefer the uint8 copies (sketches_u8/photos_u8, ~28 GB total) created by
    # hpc/convert_to_u8.py; fall back to the original float32 arrays (~110 GB).
    # All reads are memory-mapped (mmap_mode="r"); pages are cached by the OS
    # and read per-row in the generator. The generator normalizes uint8 ->
    # float32 [0,1] internally.
    sk_path = PROCESSED_DIR / "sketches_u8.npy"
    ph_path = PROCESSED_DIR / "photos_u8.npy"
    if not (sk_path.exists() and ph_path.exists()):
        sk_path = PROCESSED_DIR / "sketches.npy"
        ph_path = PROCESSED_DIR / "photos.npy"
    sketches = np.load(sk_path, mmap_mode="r")
    photos = np.load(ph_path, mmap_mode="r")

    skl_path = PROCESSED_DIR / "sketch_labels_u8.npy"
    phl_path = PROCESSED_DIR / "photo_labels_u8.npy"
    if not (skl_path.exists() and phl_path.exists()):
        skl_path = PROCESSED_DIR / "sketch_labels.npy"
        phl_path = PROCESSED_DIR / "photo_labels.npy"
    sketch_labels = np.load(skl_path)
    photo_labels = np.load(phl_path)

    with open(PROCESSED_DIR / "category_names.json") as f:
        category_names = {int(k): v for k, v in json.load(f).items()}

    with open(PROCESSED_DIR / "metadata.json") as f:
        metadata = json.load(f)

    return sketches, photos, sketch_labels, photo_labels, category_names, metadata


def get_split_masks(sketch_labels, photo_labels, metadata, category_names=None):
    """Return masks for train/val/test categories (from metadata)."""
    train_cats = metadata.get("train_categories", [])
    val_cats = metadata.get("val_categories", [])
    test_cats = metadata.get("test_categories", [])

    if not train_cats:
        cats = list(category_names.keys()) if category_names else \
            sorted(set(sketch_labels) | set(photo_labels))
        train_cats = val_cats = test_cats = cats

    def _int_list(x):
        return [int(c) for c in x]

    train_cats = _int_list(train_cats)
    val_cats = _int_list(val_cats)
    test_cats = _int_list(test_cats)

    return {
        "tr_sk": np.isin(sketch_labels, train_cats),
        "tr_ph": np.isin(photo_labels, train_cats),
        "va_sk": np.isin(sketch_labels, val_cats),
        "va_ph": np.isin(photo_labels, val_cats),
        "te_sk": np.isin(sketch_labels, test_cats),
        "te_ph": np.isin(photo_labels, test_cats),
        "train_cats": train_cats,
        "val_cats": val_cats,
        "test_cats": test_cats,
    }


def create_data_generators(batch_size=32, pairs_per_epoch=50000, augment=False,
                           teacher_sketch_embs=None, teacher_photo_embs=None,
                           hard_negative_ratio=HARD_NEGATIVE_RATIO,
                           confusable_pairs=CONFUSABLE_PAIRS,
                           arcface=USE_ARC_FACE):
    """
    Build train/val/test pair generators.

    Args:
      batch_size, pairs_per_epoch: training batch / epoch size
      augment: on-the-fly augmentation (keep False when distilling)
      teacher_sketch_embs/teacher_photo_embs: (N, EMB) arrays aligned with the
        FULL processed arrays; rows are sub-sliced by the split masks.
      hard_negative_ratio, confusable_pairs: hard-negative mining config
      arcface: emit category labels (extra model inputs + CE targets)
    """
    (sketches, photos, sketch_labels, photo_labels,
     category_names, metadata) = load_processed_data()

    m = get_split_masks(sketch_labels, photo_labels, metadata, category_names)

    def _slice(embs, mask):
        return None if embs is None else embs[mask]

    def _global(mask):
        # LOCAL (masked) row index -> GLOBAL row index into the full memmapped
        # array, so the generator never materializes the masked slice.
        return np.where(mask)[0]

    def _make(sk_mask, ph_mask, ppb, aug, ratio, conf):
        return SketchPhotoPairGenerator(
            sketches, photos,
            sketch_labels[sk_mask], photo_labels[ph_mask],
            category_names,
            batch_size=batch_size, pairs_per_epoch=ppb,
            shuffle=True, augment=aug,
            teacher_sketch_embs=_slice(teacher_sketch_embs, sk_mask),
            teacher_photo_embs=_slice(teacher_photo_embs, ph_mask),
            hard_negative_ratio=ratio, confusable_pairs=conf,
            sk_global=_global(sk_mask), ph_global=_global(ph_mask),
            arcface=arcface,
        )

    train_gen = _make(m["tr_sk"], m["tr_ph"], pairs_per_epoch, augment,
                      hard_negative_ratio, confusable_pairs)
    val_gen = _make(m["va_sk"], m["va_ph"], max(64, pairs_per_epoch // 5),
                    False, 0.0, None)
    test_gen = _make(m["te_sk"], m["te_ph"], max(64, pairs_per_epoch // 5),
                     False, 0.0, None)

    return train_gen, val_gen, test_gen, category_names
