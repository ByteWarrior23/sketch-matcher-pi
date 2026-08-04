"""
export_tflite.py

Converts the trained embedding model to TensorFlow Lite for Raspberry Pi deployment.

Conversion steps:
  1. Load the trained embedding model (single branch, not Siamese)
  2. Convert to TFLite with float32 precision
  3. Apply post-training int8 quantization (smaller, faster on Pi)
  4. Pre-compute all photo embeddings for fast matching on Pi
  5. Save everything in pi_deploy/ folder

Usage:
    python src/export_tflite.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from config import (
        BEST_MODEL_PATH, MODELS_DIR, PROCESSED_DIR,
        TFLITE_FILENAME, TFLITE_QUANTIZE,
        PHOTO_EMBEDDINGS_FILENAME, PHOTO_LABELS_FILENAME,
        LABELS_FILENAME, LOG_LEVEL,
    )
except ModuleNotFoundError:  # Colab: imported as src.export_tflite
    from src.config import (
        BEST_MODEL_PATH, MODELS_DIR, PROCESSED_DIR,
        TFLITE_FILENAME, TFLITE_QUANTIZE,
        PHOTO_EMBEDDINGS_FILENAME, PHOTO_LABELS_FILENAME,
        LABELS_FILENAME, LOG_LEVEL,
    )
from data_loader import load_processed_data

import logging

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def convert_to_tflite(model_path, quantize=TFLITE_QUANTIZE):
    """
    Convert Keras embedding model to TFLite.

    Args:
      model_path: path to the trained .keras file
      quantize: if True, apply int8 quantization (smaller but slight accuracy drop)

    Returns:
      tflite_model: bytes of the converted model
    """
    log.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Check input shape (should be [None, 224, 224, 3])
    input_shape = model.input_shape
    log.info(f"Model input shape: {input_shape}")
    log.info(f"Model output shape: {model.output_shape}")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        # Apply int8 quantization
        # This reduces model size ~4x (7 MB → 1.8 MB)
        # Inference is ~2-3x faster on Pi
        # Accuracy drop: typically <1%
        log.info("Applying int8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Provide representative dataset for calibration
        # (needed for accurate int8 quantization)
        def representative_dataset():
            # Load a small subset of images for calibration
            sketches = np.load(PROCESSED_DIR / "sketches.npy")
            # Take first 100 sketches for calibration
            for i in range(min(100, len(sketches))):
                img = sketches[i].astype(np.float32)
                img = img[np.newaxis, ...]  # Add batch dimension
                yield [img]

        converter.representative_dataset = representative_dataset
        # Fall back to float16 for layers that don't support int8
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
        log.info("  Target: int8 quantized (input: uint8, output: float32)")
    else:
        # Float32 conversion (no quantization)
        log.info("Converting to float32 TFLite...")

    tflite_model = converter.convert()
    return tflite_model


def precompute_photo_embeddings(tflite_model_path, photos):
    """
    Pre-compute embeddings for all photos using the TFLite model.

    These embeddings are stored on the Pi and used at inference time
    to compare against the sketch embedding.

    Args:
      tflite_model_path: path to the .tflite file
      photos: np.array of shape (N, 224, 224, 3)

    Returns:
      photo_embeddings: np.array of shape (N, 128)
    """
    log.info("Pre-computing photo embeddings with TFLite model...")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check input type (may be uint8 if quantized)
    input_dtype = input_details[0]["dtype"]
    log.info(f"  TFLite input type: {input_dtype}")

    n_photos = len(photos)
    embedding_dim = output_details[0]["shape"][1]
    all_embeddings = np.zeros((n_photos, embedding_dim), dtype=np.float32)

    # Process in batches to avoid OOM
    batch_size = 64
    for start in range(0, n_photos, batch_size):
        end = min(start + batch_size, n_photos)
        batch = photos[start:end].astype(np.float32)

        # If quantized model expects uint8 input, scale
        if input_dtype == np.uint8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            if not input_scale:
                input_scale = 1.0
            batch = np.rint(batch / input_scale + input_zero_point).astype(np.uint8)
        else:
            batch = batch.astype(np.float32)

        for i in range(len(batch)):
            interpreter.set_tensor(input_details[0]["index"], batch[i:i+1])
            interpreter.invoke()
            all_embeddings[start + i] = interpreter.get_tensor(output_details[0]["index"])

        if (start // batch_size) % 10 == 0:
            log.info(f"  Processed {end}/{n_photos} photos...")

    log.info(f"  Computed {n_photos} embeddings, dim={embedding_dim}")
    return all_embeddings


def main():
    log.info("=" * 60)
    log.info("Sketch Matcher - TFLite Export")
    log.info("=" * 60)

    # Step 1: Check that trained model exists
    if not BEST_MODEL_PATH.exists():
        log.error(f"Trained model not found at {BEST_MODEL_PATH}")
        log.error("Run: python src/train.py")
        sys.exit(1)

    # Step 2: Convert to TFLite
    log.info("\n[1/4] Converting model to TFLite...")
    tflite_model = convert_to_tflite(BEST_MODEL_PATH)

    tflite_path = MODELS_DIR / TFLITE_FILENAME
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    log.info(f"  Saved to: {tflite_path}")
    log.info(f"  Size: {size_mb:.2f} MB")
    log.info(f"  Quantized: {TFLITE_QUANTIZE}")

    log.info("\n[2/4] Loading photos from processed data...")
    sketches, photos, sketch_labels, photo_labels, category_names, metadata = load_processed_data()
    log.info(f"  Total photos: {len(photos)}")

    log.info("\n[3/4] Pre-computing photo embeddings...")
    photo_embeddings = precompute_photo_embeddings(tflite_path, photos)

    embeddings_path = MODELS_DIR / PHOTO_EMBEDDINGS_FILENAME
    np.save(embeddings_path, photo_embeddings)
    log.info(f"  Saved to: {embeddings_path}")
    log.info(f"  Shape: {photo_embeddings.shape}")

    # Save category labels
    labels_path = MODELS_DIR / LABELS_FILENAME
    with open(labels_path, "w") as f:
        json.dump({int(k): v for k, v in category_names.items()}, f, indent=2)
    log.info(f"  Labels saved to: {labels_path}")

    # Save per-photo category labels (ALIGNED with photo_embeddings rows!)
    photo_labels_path = MODELS_DIR / PHOTO_LABELS_FILENAME
    np.save(photo_labels_path, photo_labels)
    log.info(f"  Photo labels saved to: {photo_labels_path} ({photo_labels.shape})")

    # Step 4: Copy files to pi_deploy/ for easy transfer
    log.info("\n[4/4] Copying files to pi_deploy/...")
    pi_deploy_dir = Path(__file__).resolve().parent.parent / "pi_deploy" / "model_data"
    pi_deploy_dir.mkdir(parents=True, exist_ok=True)

    # Copy TFLite model
    import shutil
    shutil.copy2(tflite_path, pi_deploy_dir / TFLITE_FILENAME)
    log.info(f"  Copy: {TFLITE_FILENAME}")

    # Copy photo embeddings
    shutil.copy2(embeddings_path, pi_deploy_dir / PHOTO_EMBEDDINGS_FILENAME)
    log.info(f"  Copy: {PHOTO_EMBEDDINGS_FILENAME}")

    # Copy per-photo labels (needed to map embedding rows -> categories on Pi)
    shutil.copy2(photo_labels_path, pi_deploy_dir / PHOTO_LABELS_FILENAME)
    log.info(f"  Copy: {PHOTO_LABELS_FILENAME}")

    # Copy labels
    shutil.copy2(labels_path, pi_deploy_dir / LABELS_FILENAME)
    log.info(f"  Copy: {LABELS_FILENAME}")

    log.info(f"\nFiles ready for Pi deployment in: {pi_deploy_dir}")
    log.info(f"  Total size: ~{size_mb + len(photo_embeddings.tobytes()) / (1024*1024):.1f} MB")
    log.info("\n" + "=" * 60)
    log.info("Export complete!")
    log.info("Next step: Copy pi_deploy/ folder to Raspberry Pi")
    log.info("  scp -r pi_deploy/ pi@raspberrypi:~/sketch_matcher/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
