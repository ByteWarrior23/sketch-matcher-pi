"""
train.py

3-stage Siamese training pipeline with optional teacher->student distillation.

Distillation (ENABLE_DISTILLATION=True):
  Phase A: train a TEACHER (e.g. ConvNeXtTiny) with the chosen metric loss
           -> teacher_model.keras + cached teacher embeddings.
  Phase B: train the STUDENT (MobileNetV2 -- the model that ships to Pi)
           with combined loss:
             (1-alpha)*pair_loss + alpha/2*MSE(emb_a, teacher_a)
                                 + alpha/2*MSE(emb_b, teacher_b)

Each phase runs 3 stages:
  Stage 1: freeze backbone (100 ep, LR 1e-3)
  Stage 2: unfreeze last N backbone layers (60 ep, LR 1e-4)
  Stage 3: full fine-tune (150 ep, LR 1e-5)

Usage:
    python src/train.py
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from config import (
        STAGE1_EPOCHS, STAGE1_BATCH_SIZE, STAGE1_LEARNING_RATE,
        STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_LEARNING_RATE,
        STAGE3_EPOCHS, STAGE3_BATCH_SIZE, STAGE3_LEARNING_RATE,
        EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
        BEST_MODEL_PATH, FINAL_MODEL_PATH, TEACHER_MODEL_PATH, CHECKPOINT_DIR,
        TENSORBOARD_LOG_DIR, LOGS_DIR, PROCESSED_DIR, LOG_LEVEL,
        PAIRS_PER_EPOCH, BACKBONE, TEACHER_BACKBONE, UNFREEZE_FROM_STAGE2,
        EMBEDDING_DIM, LOSS_TYPE, ENABLE_DISTILLATION,
    )
except ModuleNotFoundError:  # Colab
    from src.config import (
        STAGE1_EPOCHS, STAGE1_BATCH_SIZE, STAGE1_LEARNING_RATE,
        STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_LEARNING_RATE,
        STAGE3_EPOCHS, STAGE3_BATCH_SIZE, STAGE3_LEARNING_RATE,
        EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
        BEST_MODEL_PATH, FINAL_MODEL_PATH, TEACHER_MODEL_PATH, CHECKPOINT_DIR,
        TENSORBOARD_LOG_DIR, LOGS_DIR, PROCESSED_DIR, LOG_LEVEL,
        PAIRS_PER_EPOCH, BACKBONE, TEACHER_BACKBONE, UNFREEZE_FROM_STAGE2,
        EMBEDDING_DIM, LOSS_TYPE, ENABLE_DISTILLATION,
    )

from model import (
    create_model, freeze_backbone, unfreeze_last_n, unfreeze_all,
    count_trainable_params, recompile_with_lr,
)
from data_loader import load_processed_data, create_data_generators

import logging

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "training.log"),
    ],
)
log = logging.getLogger(__name__)


def get_callbacks(stage_name, stage_dir):
    stage_dir.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR,
                          patience=REDUCE_LR_PATIENCE, min_lr=1e-8, verbose=1),
        ModelCheckpoint(str(stage_dir / "best_epoch.keras"),
                        monitor="val_loss", save_best_only=True, verbose=1),
        TensorBoard(log_dir=str(TENSORBOARD_LOG_DIR / stage_name)),
        CSVLogger(str(LOGS_DIR / f"{stage_name}_history.csv")),
    ]


def train_stage(siamese, embedding_net, backbone, train_gen, val_gen,
                epochs, batch_size, learning_rate, stage_name, freeze_fn=None):
    if freeze_fn:
        freeze_fn(embedding_net, backbone)
        count_trainable_params(siamese)

    recompile_with_lr(siamese, learning_rate)

    log.info(f"\n{'=' * 60}")
    log.info(f"Stage: {stage_name}")
    log.info(f"  Epochs: {epochs}")
    log.info(f"  Batch size: {batch_size}")
    log.info(f"  Learning rate: {learning_rate}")
    log.info(f"  Pairs per epoch: {train_gen.pairs_per_epoch}")
    log.info(f"{'=' * 60}")

    stage_dir = CHECKPOINT_DIR / stage_name
    return siamese.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(stage_name, stage_dir),
        verbose=1,
    )


def make_generators(batch_size, augment=False, teacher_sketch_embs=None,
                    teacher_photo_embs=None):
    """Create train/val/test generators for one stage."""
    train_gen, val_gen, test_gen, names = create_data_generators(
        batch_size=batch_size, pairs_per_epoch=PAIRS_PER_EPOCH,
        augment=augment,
        teacher_sketch_embs=teacher_sketch_embs,
        teacher_photo_embs=teacher_photo_embs,
    )
    return train_gen, val_gen, test_gen, names


def run_three_stages(siamese, embedding_net, backbone, tag,
                     teacher_sketch_embs=None, teacher_photo_embs=None,
                     augment=False):
    """Run all 3 training stages for a phase (batch size changes per stage)."""
    log.info(f"\n=== {tag.upper()} phase ===")

    train_gen, val_gen, _, _ = make_generators(
        STAGE1_BATCH_SIZE, augment, teacher_sketch_embs, teacher_photo_embs)
    train_stage(siamese, embedding_net, backbone, train_gen, val_gen,
                epochs=STAGE1_EPOCHS, batch_size=STAGE1_BATCH_SIZE,
                learning_rate=STAGE1_LEARNING_RATE,
                stage_name=f"{tag}_stage1", freeze_fn=freeze_backbone)

    train_gen, val_gen, _, _ = make_generators(
        STAGE2_BATCH_SIZE, augment, teacher_sketch_embs, teacher_photo_embs)
    train_stage(siamese, embedding_net, backbone, train_gen, val_gen,
                epochs=STAGE2_EPOCHS, batch_size=STAGE2_BATCH_SIZE,
                learning_rate=STAGE2_LEARNING_RATE,
                stage_name=f"{tag}_stage2",
                freeze_fn=lambda net, bb: unfreeze_last_n(
                    net, bb, n=UNFREEZE_FROM_STAGE2))

    train_gen, val_gen, _, _ = make_generators(
        STAGE3_BATCH_SIZE, augment, teacher_sketch_embs, teacher_photo_embs)
    train_stage(siamese, embedding_net, backbone, train_gen, val_gen,
                epochs=STAGE3_EPOCHS, batch_size=STAGE3_BATCH_SIZE,
                learning_rate=STAGE3_LEARNING_RATE,
                stage_name=f"{tag}_stage3", freeze_fn=unfreeze_all)


def compute_teacher_embeddings(teacher_net, batch_size=128, force=False):
    """Teacher embeddings for ALL processed images (cached to disk)."""
    sk_path = PROCESSED_DIR / "teacher_sketch_embs.npy"
    ph_path = PROCESSED_DIR / "teacher_photo_embs.npy"
    if not force and sk_path.exists() and ph_path.exists():
        log.info("Loading cached teacher embeddings...")
        return np.load(sk_path), np.load(ph_path)

    sketches, photos, _, _, _, _ = load_processed_data()
    log.info("Computing teacher sketch embeddings...")
    t_sk = teacher_net.predict(sketches, batch_size=batch_size, verbose=1)
    log.info("Computing teacher photo embeddings...")
    t_ph = teacher_net.predict(photos, batch_size=batch_size, verbose=1)

    np.save(sk_path, t_sk)
    np.save(ph_path, t_ph)
    log.info(f"Teacher embeddings saved: {t_sk.shape}, {t_ph.shape}")
    return t_sk, t_ph


def main():
    log.info("=" * 60)
    log.info("Sketch Matcher - Training Pipeline")
    log.info("=" * 60)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info(f"GPU available: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        log.warning("No GPU found! Training will be very slow on CPU.")

    if not (PROCESSED_DIR / "sketches.npy").exists():
        log.error("Preprocessed data not found! Run: python src/preprocess.py")
        sys.exit(1)

    log.info(f"\nLoss: {LOSS_TYPE}  |  Distillation: {ENABLE_DISTILLATION}")

    if not ENABLE_DISTILLATION:
        # ---------------- Single-model training ----------------
        log.info("\n[1/3] Creating model...")
        siamese, embedding_net, backbone = create_model(
            backbone_name=BACKBONE, loss_type=LOSS_TYPE,
            include_embeddings=False, embedding_dim=EMBEDDING_DIM)
        count_trainable_params(siamese)

        run_three_stages(siamese, embedding_net, backbone, "single",
                         augment=True)

        log.info("\nSaving final models...")
        siamese.save(FINAL_MODEL_PATH)
        embedding_net.save(BEST_MODEL_PATH)
        log.info(f"  Siamese: {FINAL_MODEL_PATH}")
        log.info(f"  Embedding: {BEST_MODEL_PATH}")

    else:
        # ---------------- Phase A: teacher ----------------
        log.info("\n[1/4] Training TEACHER...")
        teacher_siamese, teacher_net, teacher_base = create_model(
            backbone_name=TEACHER_BACKBONE, loss_type=LOSS_TYPE,
            include_embeddings=False, embedding_dim=EMBEDDING_DIM)
        count_trainable_params(teacher_siamese)

        run_three_stages(teacher_siamese, teacher_net, teacher_base, "teacher",
                         augment=True)

        teacher_net.save(TEACHER_MODEL_PATH)
        log.info(f"  Teacher embedding saved: {TEACHER_MODEL_PATH}")

        # Compute + cache teacher embeddings for all processed images
        t_sk, t_ph = compute_teacher_embeddings(teacher_net)

        # ---------------- Phase B: student with distillation ----------------
        log.info(f"\n[2/4] Training STUDENT ({BACKBONE}) with distillation...")
        siamese, embedding_net, backbone = create_model(
            backbone_name=BACKBONE, loss_type=LOSS_TYPE,
            include_embeddings=True, embedding_dim=EMBEDDING_DIM)
        count_trainable_params(siamese)

        run_three_stages(siamese, embedding_net, backbone, "student",
                         teacher_sketch_embs=t_sk, teacher_photo_embs=t_ph,
                         augment=False)

        log.info("\n[3/4] Saving student model...")
        siamese.save(FINAL_MODEL_PATH)
        embedding_net.save(BEST_MODEL_PATH)
        log.info(f"  Siamese: {FINAL_MODEL_PATH}")
        log.info(f"  Embedding: {BEST_MODEL_PATH}")

    # ---------------- Sanity check ----------------
    log.info("\n[4/4] Quick validation check...")
    _, val_gen, _, _ = make_generators(64)
    siamese.evaluate(val_gen, steps=50, verbose=1)

    log.info("\n" + "=" * 60)
    log.info("Training complete!")
    log.info("  1. Evaluate: python src/evaluate.py")
    log.info("  2. Export TFLite: python src/export_tflite.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()