"""
model.py

Siamese CNN architecture for sketch-photo matching (configurable backbone).

Architecture:
  Input A (sketch) ──┐
                      ├── shared embedding net ── Dense ── L2Norm ── Embedding
  Input B (photo) ──┘

Supported backbones (all ImageNet-pretrained, all TFLite-convertible):
  - "mobilenetv2"     3.5M params  ~150ms on Pi 5 (older STUDENT option)
  - "mobilenetv3large" 5.4M params ~100-150ms on Pi 5 (STUDENT, ships to Pi;
                 better accuracy than V2 at similar/smaller FLOPs, has a
                 built-in [-1,1] rescale so [0,1] inputs are fed correctly)
  - "efficientnetv2s" 21M params  high accuracy    (TEACHER)
  - "convnexttiny"   28M params  top accuracy      (TEACHER)

Losses:
  - contrastive (distance-based)
  - circle      (Sun et al. 2020, SOTA metric loss; operates on cosine sim
                 derived from the L2-normalized embedding distance)
  - arcface head (Deng et al. 2019, additive angular margin classifier on the
                 L2-normed embedding). When USE_ARC_FACE is on, each branch
                 also produces category logits trained with sparse CE; the
                 embedding stays L2-normed for retrieval.

Distillation: build_distilled_siamese_model() returns a 3-output model
[distance, emb_a, emb_b]. With teacher embeddings as extra y-targets the
combined loss becomes:
    (1-alpha) * pair_loss(distance) + alpha/2 * MSE(emb_a, teacher_a)
                                    + alpha/2 * MSE(emb_b, teacher_b)

Training strategy (3 stages):
  Stage 1: freeze backbone, train dense layers
  Stage 2: unfreeze last N backbone layers
  Stage 3: full fine-tune, very low LR
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers

try:
    from config import (
        IMG_SIZE, IMG_CHANNELS, EMBEDDING_DIM, DROPOUT_RATE,
        CONTRASTIVE_MARGIN, CIRCLE_M, CIRCLE_GAMMA, USE_PRETRAINED,
        BACKBONE, TEACHER_BACKBONE, LOSS_TYPE, DISTILL_ALPHA,
        USE_ARC_FACE, ARC_FACE_MARGIN, ARC_FACE_SCALE, ARC_FACE_LAMBDA,
        NUM_CLASSES, LOG_LEVEL,
    )
except ModuleNotFoundError:  # Colab: imported as src.model
    from src.config import (
        IMG_SIZE, IMG_CHANNELS, EMBEDDING_DIM, DROPOUT_RATE,
        CONTRASTIVE_MARGIN, CIRCLE_M, CIRCLE_GAMMA, USE_PRETRAINED,
        BACKBONE, TEACHER_BACKBONE, LOSS_TYPE, DISTILL_ALPHA,
        USE_ARC_FACE, ARC_FACE_MARGIN, ARC_FACE_SCALE, ARC_FACE_LAMBDA,
        NUM_CLASSES, LOG_LEVEL,
    )

import logging

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BACKBONE_BUILDERS = {
    "mobilenetv2": tf.keras.applications.MobileNetV2,
    "mobilenetv3large": tf.keras.applications.MobileNetV3Large,
    "efficientnetv2s": tf.keras.applications.EfficientNetV2S,
    "convnexttiny": tf.keras.applications.ConvNeXtTiny,
}


# =============================================================================
# EMBEDDING NETWORK
# =============================================================================
def get_backbone(backbone_name, input_shape):
    if backbone_name not in BACKBONE_BUILDERS:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Choose from {list(BACKBONE_BUILDERS.keys())}"
        )
    return BACKBONE_BUILDERS[backbone_name](
        include_top=False,
        weights="imagenet" if USE_PRETRAINED else None,
        input_shape=input_shape,
        pooling="avg",
    )


def build_embedding_network(input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS),
                            backbone_name=BACKBONE, embedding_dim=EMBEDDING_DIM,
                            dropout_rate=DROPOUT_RATE):
    """Build the single-branch embedding network. Returns (model, backbone)."""
    base = get_backbone(backbone_name, input_shape)

    inputs = Input(shape=input_shape, name="embedding_input")
    x = base(inputs)

    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(embedding_dim, name="embedding")(x)
    outputs = L2Normalize(name="l2_normalize")(x)

    model = Model(inputs, outputs, name="embedding_network")
    return model, base


# =============================================================================
# LOSSES
# =============================================================================
def contrastive_loss(y_true, y_pred, margin=CONTRASTIVE_MARGIN):
    """Classic contrastive loss on Euclidean distance (0=match, 1=no match)."""
    y_true = tf.cast(y_true, tf.float32)
    match_loss = (1 - y_true) * 0.5 * tf.square(y_pred)
    non_match_loss = y_true * 0.5 * tf.square(
        tf.maximum(0.0, margin - y_pred))
    return tf.reduce_mean(match_loss + non_match_loss)


def circle_loss(y_true, y_pred, m=CIRCLE_M, gamma=CIRCLE_GAMMA):
    """
    Circle loss (Sun et al. 2020) on cosine similarity, per-pair masked.

    The siamese distance output d in [0,2] maps to cosine similarity
    s = 1 - d^2/2 for L2-normalized embeddings. Positive pairs (label 0)
    and negative pairs (label 1) are separated with adaptive margins:
        L_pos = log(1 + sum_pos exp(-gamma*alpha_p*(s_p - (1-m))))
        L_neg = log(1 + sum_neg exp(+gamma*alpha_n*(s_n - m)))
    where alpha_p = max(1 + m - s_p, 0), alpha_n = max(s_n + m, 0).

    Only same-class pairs contribute to each term (boolean-masked). The
    old version multiplied by (1-y_true)/y_true AFTER the exponential, so
    the opposite class still entered the sum as a constant exp(gamma*m)
    (~exp(20) with defaults) that dominated and hid the real learning
    signal.
    """
    y_true = tf.cast(y_true, tf.float32)
    sim = 1.0 - 0.5 * tf.square(y_pred)  # cosine similarity in [0, 1]

    pos_sim = tf.boolean_mask(sim, y_true < 0.5)
    neg_sim = tf.boolean_mask(sim, y_true >= 0.5)

    has_pos = tf.greater(tf.size(pos_sim), 0)
    has_neg = tf.greater(tf.size(neg_sim), 0)

    def pos_term():
        alpha_p = tf.maximum(1.0 + m - pos_sim, 0.0)
        l_pos = tf.reduce_sum(tf.exp(-gamma * alpha_p * (pos_sim - (1.0 - m))))
        return tf.math.log1p(l_pos)

    def neg_term():
        alpha_n = tf.maximum(neg_sim + m, 0.0)
        l_neg = tf.reduce_sum(tf.exp(gamma * alpha_n * (neg_sim - m)))
        return tf.math.log1p(l_neg)

    zero = tf.constant(0.0, dtype=tf.float32)
    batch = tf.cast(tf.shape(y_true)[0], tf.float32)
    return (tf.cond(has_pos, pos_term, lambda: zero)
            + tf.cond(has_neg, neg_term, lambda: zero)) / batch


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def get_pair_loss(loss_type=LOSS_TYPE):
    if loss_type == "contrastive":
        return contrastive_loss
    if loss_type == "circle":
        return circle_loss
    raise ValueError(f"Unknown LOSS_TYPE '{loss_type}'")


# =============================================================================
# SIAMESE MODELS
# =============================================================================
@tf.keras.utils.register_keras_serializable()
class L2Normalize(layers.Layer):
    """L2-normalizes the embedding along axis 1 (replaces a Lambda so the
    model survives save/load and TFLite conversion)."""

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


@tf.keras.utils.register_keras_serializable()
class DistanceLayer(layers.Layer):
    """Euclidean distance between two L2-normalized embeddings."""

    def call(self, inputs):
        embedding_a, embedding_b = inputs
        return tf.sqrt(tf.reduce_sum(
            tf.square(embedding_a - embedding_b), axis=1, keepdims=True))


@tf.keras.utils.register_keras_serializable()
class ArcFaceLayer(layers.Layer):
    """
    Additive Angular Margin (ArcFace) classifier on an L2-normalized embedding.

    call([embedding (B, D), labels (B,)]) -> cosine logits (B, C) with the
    additive margin m applied to the target logit and the whole thing scaled
    by s. Trained with SparseCategoricalCrossentropy(from_logits=True).

    The labels come in TWICE: once as this layer's input (needed during the
    forward pass to apply the margin) and again as the CE target from the
    generator (standard Keras ArcFace pattern).
    """

    def __init__(self, num_classes, margin=0.5, scale=64.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.margin = float(margin)
        self.scale = float(scale)

    def build(self, input_shapes):
        embedding_dim = input_shapes[0][-1]
        self.w = self.add_weight(
            name="arcface_w",
            shape=(embedding_dim, self.num_classes),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        embedding, labels = inputs
        w_norm = tf.math.l2_normalize(self.w, axis=0)
        cos_theta = tf.matmul(embedding, w_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

        theta = tf.math.acos(cos_theta)
        target_logits = tf.math.cos(theta + self.margin)

        one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
        logits = self.scale * (
            cos_theta * (1.0 - one_hot) + target_logits * one_hot)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "margin": self.margin,
            "scale": self.scale,
        })
        return config


def build_siamese_model(embedding_net, loss_type=LOSS_TYPE,
                        include_embeddings=False, alpha=DISTILL_ALPHA,
                        use_arcface=USE_ARC_FACE, num_classes=NUM_CLASSES,
                        arc_margin=ARC_FACE_MARGIN, arc_scale=ARC_FACE_SCALE,
                        arc_lambda=ARC_FACE_LAMBDA):
    """
    Build the Siamese model.

    Args:
      embedding_net: shared embedding network (built by build_embedding_network)
      loss_type: "contrastive" | "circle"
      include_embeddings: if True, model outputs [distance, emb_a, emb_b]
        (for distillation); else just [distance].
      alpha: distillation weight (used to set loss weights).
      use_arcface: if True, add an ArcFace classifier head per branch. The
        category labels become extra model INPUTS (for the margin during the
        forward pass) and are also returned by the generator as CE targets.
      num_classes, arc_margin, arc_scale, arc_lambda: ArcFace config.

    Returns: compiled siamese model.
    """
    input_a = Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS), name="input_sketch")
    input_b = Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS), name="input_photo")

    model_inputs = [input_a, input_b]
    targets_extra = []

    embedding_a = embedding_net(input_a)
    embedding_b = embedding_net(input_b)

    distance = DistanceLayer(name="distance")([embedding_a, embedding_b])

    arc_layer = None
    if use_arcface:
        label_a = Input(shape=(), dtype=tf.int64, name="label_sketch")
        label_b = Input(shape=(), dtype=tf.int64, name="label_photo")
        model_inputs += [label_a, label_b]
        # ONE shared ArcFaceLayer instance -> shared class weight matrix.
        arc_layer = ArcFaceLayer(
            num_classes=num_classes, margin=arc_margin, scale=arc_scale,
            name="arcface")
        logits_a = arc_layer([embedding_a, label_a])
        logits_b = arc_layer([embedding_b, label_b])

    if include_embeddings:
        outputs = [distance, embedding_a, embedding_b]
        if arc_layer is not None:
            outputs += [logits_a, logits_b]
        model = Model(inputs=model_inputs, outputs=outputs,
                      name="siamese_distill")
        # NOTE: cannot use dict losses keyed by output names here because the
        # embedding outputs both come from the same shared "l2_normalize" layer
        # (their auto-names would collide). Positional lists map 1:1 in order:
        #   output[0]=distance -> pair_loss
        #   output[1]=emb_a    -> MSE vs teacher sketch embedding
        #   output[2]=emb_b    -> MSE vs teacher photo embedding
        #   output[3/4]        -> ArcFace sparse CE (when use_arcface)
        pair_loss = get_pair_loss(loss_type)
        losses = [pair_loss, mse_loss, mse_loss]
        loss_weights = [1.0 - alpha, alpha / 2.0, alpha / 2.0]
        if arc_layer is not None:
            ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            losses += [ce, ce]
            loss_weights += [arc_lambda, arc_lambda]
        model.loss_plan = (losses, loss_weights)
    else:
        outputs = [distance]
        if arc_layer is not None:
            outputs += [logits_a, logits_b]
        model = Model(inputs=model_inputs, outputs=outputs, name="siamese")
        pair_loss = get_pair_loss(loss_type)
        losses = [pair_loss]
        loss_weights = [1.0]
        if arc_layer is not None:
            ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            losses += [ce, ce]
            loss_weights += [arc_lambda, arc_lambda]
        model.loss_plan = (losses, loss_weights)

    _compile_siamese(model, learning_rate=0.001)
    return model


def _compile_siamese(model, learning_rate):
    losses, weights = model.loss_plan
    # jit_compile=False: Keras 3 auto-enables XLA for fit() on GPU, but XLA's
    # fusion autotuner shells out to the system ptxas (CUDA 11.8) which cannot
    # target CC 9.0 (H100) -> "Autotuner could not compile any configs" crash.
    # Plain Grappler/cuDNN/cuBLAS execution avoids the ptxas dependency.
    if weights is None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=losses,
            jit_compile=False,
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=losses,
            loss_weights=weights,
            jit_compile=False,
        )

def recompile_with_lr(siamese, learning_rate):
    """Recompile a built siamese model with a new learning rate."""
    _compile_siamese(siamese, learning_rate)


# =============================================================================
# FREEZE / UNFREEZE SCHEDULE
# =============================================================================
def freeze_backbone(embedding_net, backbone):
    """Stage 1: only dense layers trainable."""
    backbone.trainable = False
    log.info("Frozen backbone: only dense layers are trainable")


def unfreeze_last_n(embedding_net, backbone, n=-12):
    """Stage 2: unfreeze the last |n| backbone layers."""
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[n:]:
        layer.trainable = True
    log.info(f"Unfroze last {abs(n)} layers of backbone")


def unfreeze_all(embedding_net, backbone):
    """Stage 3: full fine-tune."""
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = True
    log.info("Unfroze ALL layers (full fine-tune)")


def count_trainable_params(model):
    total = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    log.info(f"Trainable params: {total:,}")
    log.info(f"Non-trainable params: {non_trainable:,}")
    log.info(f"Total params: {total + non_trainable:,}")
    return total, non_trainable


# =============================================================================
# CONVENIENCE
# =============================================================================
def create_model(backbone_name=BACKBONE, loss_type=LOSS_TYPE,
                 include_embeddings=False, embedding_dim=EMBEDDING_DIM,
                 use_arcface=USE_ARC_FACE):
    """Build embedding net + compiled siamese. Returns (siamese, embedding_net, backbone)."""
    embedding_net, backbone = build_embedding_network(
        backbone_name=backbone_name, embedding_dim=embedding_dim)
    siamese = build_siamese_model(embedding_net, loss_type=loss_type,
                                  include_embeddings=include_embeddings,
                                  use_arcface=use_arcface)
    return siamese, embedding_net, backbone
