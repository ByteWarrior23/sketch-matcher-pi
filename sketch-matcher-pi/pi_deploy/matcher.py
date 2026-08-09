"""
matcher.py — Nearest Neighbor Search on Raspberry Pi

Loads the TFLite model + pre-computed photo embeddings + per-photo labels.
Compares a new sketch embedding against all photo embeddings and returns
the top-K closest CATEGORIES.

FIXES vs previous version:
  - Photo embedding rows are mapped to categories via photo_labels.npy
    (row-aligned with photo_embeddings.npy) — no more index modulo hack.
  - Results are deduplicated by category (max similarity per category), so
    a category with thousands of photos can't flood the top-3.
  - Open-set rejection: if the top-1 similarity < PI_REJECT_THRESHOLD,
    the result is flagged "NOT FOUND".

This runs in ~5ms for tens of thousands of photos (numpy vectorized).
"""

import json
import time
from pathlib import Path

import numpy as np

from config import (
    PI_SHOW_TOP_K, PI_CONFIDENCE_THRESHOLD, PI_REJECT_THRESHOLD,
    TFLITE_FILENAME, PHOTO_EMBEDDINGS_FILENAME, PHOTO_LABELS_FILENAME,
    LABELS_FILENAME,
)

try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except ImportError:
    # Fall back to full TensorFlow on non-Pi systems
    import tensorflow as tf
    TFLITE_AVAILABLE = False


class SketchMatcher:
    """
    Matches a sketch against a database of photo embeddings.

    Flow:
      1. Load TFLite model + photo embeddings + labels on init
      2. embed(sketch_image) -> N-dim vector
      3. match(sketch_image)  -> top-K categories with confidence + reject flag
    """

    def __init__(self, model_dir: Path):
        """model_dir must contain: .tflite, photo_embeddings.npy,
        photo_labels.npy, labels.json."""
        self.model_dir = model_dir
        self.interpreter = None
        self.photo_embeddings = None
        self.photo_labels = None
        self.labels = None

        self._load_model()
        self._load_embeddings()
        self._load_photo_labels()
        self._load_labels()

    def _load_model(self):
        model_path = self.model_dir / TFLITE_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        t0 = time.time()
        if TFLITE_AVAILABLE:
            self.interpreter = Interpreter(model_path=str(model_path))
        else:
            # Full-TensorFlow fallback (e.g. laptop/PC): skip the XNNPACK
            # delegate, which fails on some CPU setups with "failed to
            # prepare" (same workaround as src/export_tflite.py).
            self.interpreter = tf.lite.Interpreter(
                model_path=str(model_path),
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        t1 = time.time()
        print(f"Model loaded: {model_path.name} ({t1-t0:.2f}s)")
        print(f"  Input shape:  {self.input_details[0]['shape']}")
        print(f"  Output shape: {self.output_details[0]['shape']}")
        print(f"  Input dtype:  {self.input_details[0]['dtype']}")

    def _load_embeddings(self):
        emb_path = self.model_dir / PHOTO_EMBEDDINGS_FILENAME
        if not emb_path.exists():
            raise FileNotFoundError(f"Photo embeddings not found: {emb_path}")

        self.photo_embeddings = np.load(str(emb_path))
        print(f"Photo embeddings loaded: {self.photo_embeddings.shape}")

        norms = np.linalg.norm(self.photo_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.photo_embeddings = self.photo_embeddings / norms

    def _load_photo_labels(self):
        """Per-photo category index, row-aligned with photo_embeddings."""
        path = self.model_dir / PHOTO_LABELS_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"photo_labels.npy not found at {path}. "
                f"Re-export with the updated src/export_tflite.py.")
        self.photo_labels = np.load(str(path))
        print(f"Photo labels loaded: {self.photo_labels.shape}")

    def _load_labels(self):
        labels_path = self.model_dir / LABELS_FILENAME
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        with open(labels_path) as f:
            self.labels_raw = json.load(f)
        # int index -> category name
        self.labels = {int(k): v for k, v in self.labels_raw.items()}
        print(f"Categories loaded: {len(self.labels)}")

    def embed(self, image):
        """
        Compute the n-dim embedding for one image.

        Args: image shape (1,224,224,3), normalized [0,1].
        Returns: embedding (n,) L2-normalized.
        """
        input_index = self.input_details[0]["index"]
        input_dtype = self.input_details[0]["dtype"]

        if input_dtype == np.uint8:
            input_scale, input_zero_point = self.input_details[0]["quantization"]
            if not input_scale:
                input_scale = 1.0
            image_uint8 = np.rint(image / input_scale + input_zero_point).astype(np.uint8)
            self.interpreter.set_tensor(input_index, image_uint8)
        else:
            self.interpreter.set_tensor(input_index, image.astype(np.float32))

        self.interpreter.invoke()

        output_index = self.output_details[0]["index"]
        embedding = self.interpreter.get_tensor(output_index)[0]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def match(self, image, top_k=PI_SHOW_TOP_K):
        """
        Match a sketch image against the photo database.

        Args:
          image: preprocessed sketch (1,224,224,3)
          top_k: number of categories to return

        Returns:
          results: list of (category_name, confidence_percent, category_idx)
          inference_ms, search_ms
          rejected: True if top-1 similarity < PI_REJECT_THRESHOLD (open-set)
        """
        t0 = time.time()
        sketch_emb = self.embed(image)
        t1 = time.time()
        inference_ms = (t1 - t0) * 1000

        similarities = np.dot(self.photo_embeddings, sketch_emb)
        t2 = time.time()
        search_ms = (t2 - t1) * 1000

        # Group by category: keep the MAX similarity per category (vectorized,
        # fast even for tens of thousands of photos on the Pi).
        n_cats = int(self.photo_labels.max()) + 1
        cat_max = np.full(n_cats, -np.inf, dtype=np.float32)
        np.maximum.at(cat_max, self.photo_labels, similarities)

        valid = np.isfinite(cat_max)
        cat_indices = np.flatnonzero(valid)
        top = cat_indices[np.argsort(cat_max[valid])[::-1][:top_k]]

        results = []
        for cat_idx in top:
            name = self.labels.get(int(cat_idx), f"cat_{int(cat_idx)}")
            results.append((name, float(cat_max[cat_idx]) * 100))

        top_conf = results[0][1] / 100.0 if results else 0.0
        rejected = top_conf < PI_REJECT_THRESHOLD

        return results, inference_ms, search_ms, rejected

    def match_with_threshold(self, image):
        """Legacy single-result API (kept for compatibility).

        NOTE: unlike match(), this does NOT deduplicate by category — the
        top result is the single highest-scoring PHOTO (row index), mapped
        to its category via photo_labels.npy.
        """
        by_index = self._raw_scores(image)
        top_row = int(np.argmax(by_index))
        top_score = float(by_index[top_row])
        cat_idx = int(self.photo_labels[top_row])
        cat_name = self.labels.get(cat_idx, f"cat_{cat_idx}")
        is_match = top_score >= PI_CONFIDENCE_THRESHOLD
        return is_match, cat_name, top_score * 100.0

    def _raw_scores(self, image):
        return np.dot(self.photo_embeddings, self.embed(image))


def test_matcher():
    """Test the matcher with a dummy image."""
    model_dir = Path(__file__).parent / "model_data"
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        print("Run src/export_tflite.py first, then copy files to pi_deploy/model_data/")
        return

    matcher = SketchMatcher(model_dir)

    dummy = np.ones((1, 224, 224, 3), dtype=np.float32) * 0.5

    print("\nTesting with dummy image...")
    results, inf_ms, search_ms, rejected = matcher.match(dummy)

    print(f"Inference: {inf_ms:.1f}ms")
    print(f"Search:    {search_ms:.1f}ms")
    print(f"Total:     {inf_ms + search_ms:.1f}ms")
    print(f"Rejected (NOT FOUND): {rejected}")
    print("\nTop matches:")
    for i, (cat, conf) in enumerate(results):
        print(f"  {i+1}. {cat}: {conf:.1f}%")


if __name__ == "__main__":
    test_matcher()