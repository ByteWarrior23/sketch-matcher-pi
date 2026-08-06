"""
evaluate.py

Evaluates the trained embedding model on held-out test categories.

Metrics computed:
  - Top-1 accuracy (exact match)
  - Top-3 accuracy (correct category in top 3)
  - Top-5 accuracy (correct category in top 5)
  - Per-category accuracy breakdown
  - Confusion matrix (for debugging which categories are confused)

For signature verification (if using CEDAR dataset):
  - False Acceptance Rate (FAR)
  - False Rejection Rate (FRR)
  - Equal Error Rate (EER)

Usage:
    python src/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    BEST_MODEL_PATH, PROCESSED_DIR, TOP_K, DISTANCE_METRIC,
    LOGS_DIR, LOG_LEVEL,
)
from data_loader import load_processed_data, predict_normalized

import logging

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def compute_embeddings(model, images, batch_size=64):
    """
    Compute 128-dim embeddings for a set of images.

    Args:
      model: tf.keras.Model (single embedding branch, not Siamese)
      images: np.array of shape (N, 224, 224, 3)
      batch_size: inference batch size

    Returns:
      embeddings: np.array of shape (N, 128)
    """
    embeddings = predict_normalized(model, images, batch_size=batch_size, verbose=1)
    return embeddings


def compute_similarity(emb_a, emb_b, metric=DISTANCE_METRIC):
    """
    Compute similarity/distance between two embedding vectors.

    Args:
      emb_a: shape (128,) or (N, 128)
      emb_b: shape (128,) or (N, 128)
      metric: "cosine" or "euclidean"

    Returns:
      scalar or array of similarity scores
    """
    if metric == "cosine":
        # Cosine similarity: 1.0 = identical, -1.0 = opposite
        emb_a = emb_a / (np.linalg.norm(emb_a, axis=-1, keepdims=True) + 1e-8)
        emb_b = emb_b / (np.linalg.norm(emb_b, axis=-1, keepdims=True) + 1e-8)
        return np.sum(emb_a * emb_b, axis=-1)
    elif metric == "euclidean":
        return -np.linalg.norm(emb_a - emb_b, axis=-1)  # Negative so higher = more similar
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_top_k(model, sketches, photos, sketch_labels, photo_labels, category_names, k_values=TOP_K):
    log.info("Computing sketch embeddings...")
    sketch_embs = compute_embeddings(model, sketches)

    log.info("Computing photo embeddings...")
    photo_embs = compute_embeddings(model, photos)

    log.info("Computing similarity matrix...")
    sim_matrix = compute_similarity(
        sketch_embs[:, np.newaxis, :],
        photo_embs[np.newaxis, :, :],
    )

    n_sketches = len(sketches)
    results = {}

    for k in k_values:
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]

        correct = 0
        for i in range(n_sketches):
            top_k_categories = photo_labels[top_k_indices[i]]
            if sketch_labels[i] in top_k_categories:
                correct += 1

        accuracy = correct / n_sketches * 100
        results[f"top_{k}"] = accuracy
        log.info(f"  Top-{k} accuracy: {accuracy:.2f}% ({correct}/{n_sketches})")

    return results


def evaluate_threshold(model, sketches, photos, sketch_labels, photo_labels, category_names):
    log.info("Computing embeddings for threshold tuning...")
    sketch_embs = compute_embeddings(model, sketches)
    photo_embs = compute_embeddings(model, photos)

    def _cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) + 1e-8) / (np.linalg.norm(b) + 1e-8))

    genuine_sims = []
    for cat_idx in set(sketch_labels):
        cat_sketch_indices = np.where(sketch_labels == cat_idx)[0]
        cat_photo_indices = np.where(photo_labels == cat_idx)[0]
        for si in cat_sketch_indices[:10]:
            for pi in cat_photo_indices[:10]:
                genuine_sims.append(_cosine(sketch_embs[si], photo_embs[pi]))

    forged_sims = []
    categories = list(set(sketch_labels))
    for _ in range(len(genuine_sims)):
        cat_a = np.random.choice(categories)
        cat_b = np.random.choice([c for c in categories if c != cat_a])
        si = np.random.choice(np.where(sketch_labels == cat_a)[0])
        pi = np.random.choice(np.where(photo_labels == cat_b)[0])
        forged_sims.append(_cosine(sketch_embs[si], photo_embs[pi]))

    genuine_sims = np.array(genuine_sims)
    forged_sims = np.array(forged_sims)

    # Sweep thresholds in COSINE-SIMILARITY space (same space the Pi matcher
    # uses via PI_CONFIDENCE_THRESHOLD / PI_REJECT_THRESHOLD).
    thresholds = np.linspace(
        min(genuine_sims.min(), forged_sims.min()),
        max(genuine_sims.max(), forged_sims.max()),
        200,
    )

    best_diff = float("inf")
    best_threshold = 0.0
    best_far = best_frr = 0.0

    log.info("\nThreshold sweep for EER (cosine similarity):")
    for threshold in thresholds:
        # FAR: forged falsely accepted (similarity above threshold)
        far = np.mean(forged_sims >= threshold) * 100
        # FRR: genuine falsely rejected (similarity below threshold)
        frr = np.mean(genuine_sims <= threshold) * 100
        if abs(far - frr) < best_diff:
            best_diff = abs(far - frr)
            best_threshold = threshold
            best_far, best_frr = far, frr

    eer = (best_far + best_frr) / 2

    log.info(f"  Best threshold: {best_threshold:.4f}")
    log.info(f"  FAR at threshold: {best_far:.2f}%")
    log.info(f"  FRR at threshold: {best_frr:.2f}%")
    log.info(f"  EER: {eer:.4f}")

    return best_threshold, eer


def main():
    log.info("=" * 60)
    log.info("Sketch Matcher - Evaluation")
    log.info("=" * 60)

    # Step 1: Load model
    log.info("\n[1/4] Loading embedding model...")
    if not BEST_MODEL_PATH.exists():
        log.error(f"Model not found at {BEST_MODEL_PATH}")
        log.error("Run training first: python src/train.py")
        sys.exit(1)

    model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
    model.summary()

    log.info("\n[2/4] Loading preprocessed data...")
    sketches, photos, sketch_labels, photo_labels, category_names, metadata = load_processed_data()

    test_cats = metadata.get("test_categories", [])
    if test_cats:
        test_cats = [int(c) for c in test_cats]
        test_sk_mask = np.isin(sketch_labels, test_cats)
        test_ph_mask = np.isin(photo_labels, test_cats)
        test_sketches = sketches[test_sk_mask]
        test_photos = photos[test_ph_mask]
        test_labels = np.concatenate([sketch_labels[test_sk_mask], photo_labels[test_ph_mask]])
    else:
        test_sketches = sketches
        test_photos = photos
        test_labels = np.concatenate([sketch_labels, photo_labels])

    log.info(f"  Test sketches: {len(test_sketches)}")
    log.info(f"  Test photos:   {len(test_photos)}")
    log.info(f"  Test categories: {len(set(test_labels))}")

    test_sk_labels = sketch_labels[test_sk_mask] if test_cats else sketch_labels
    test_ph_labels = photo_labels[test_ph_mask] if test_cats else photo_labels

    log.info("\n[3/4] Computing top-K accuracy...")
    results = evaluate_top_k(model, test_sketches, test_photos, test_sk_labels, test_ph_labels, category_names)

    log.info("\n[4/4] Computing threshold metrics...")
    best_threshold, eer = evaluate_threshold(model, test_sketches, test_photos, test_sk_labels, test_ph_labels, category_names)

    # Step 5: Save results
    results["best_threshold"] = float(best_threshold)
    results["eer"] = float(eer)

    results_path = LOGS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {results_path}")

    log.info("\n" + "=" * 60)
    log.info("Evaluation complete!")
    top1 = results.get("top_1", 0.0)
    top3 = results.get("top_3", 0.0)
    log.info(f"  Best Top-1: {top1:.2f}%")
    log.info(f"  Best Top-3: {top3:.2f}%")
    log.info(f"  Best EER:   {results.get('eer', 0.0):.4f}")
    log.info("Next step: python src/export_tflite.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
