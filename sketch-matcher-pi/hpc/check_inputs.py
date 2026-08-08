import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "GPU-e1df975a-3c1b-ed1f-8803-08f5c22da8ae")
import tensorflow as tf
import numpy as np

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

for name, builder in [
    ("mobilenetv3large", tf.keras.applications.MobileNetV3Large),
    ("convnexttiny", tf.keras.applications.ConvNeXtTiny),
    ("mobilenetv2", tf.keras.applications.MobileNetV2),
    ("efficientnetv2s", tf.keras.applications.EfficientNetV2S),
]:
    m = builder(include_top=False, weights=None, input_shape=(224, 224, 3), pooling="avg")
    first = m.layers[1] if m.layers[0].__class__.__name__ == "InputLayer" else m.layers[0]
    print("=" * 70)
    print(name, "-> layers[0..2]:", [l.__class__.__name__ for l in m.layers[:3]])
    for l in m.layers[:2]:
        cfg = getattr(l, "get_config", lambda: {})()
        if "scale" in cfg or "offset" in cfg:
            print("   rescaling config:", cfg)

    # feed the SAME white image at [0,1] and at [0,255]
    white01 = np.full((1, 224, 224, 3), 1.0, dtype=np.float32)
    white255 = np.full((1, 224, 224, 3), 255.0, dtype=np.float32)
    black01 = np.zeros((1, 224, 224, 3), dtype=np.float32)
    f01 = m.predict(white01, verbose=0)
    f255 = m.predict(white255, verbose=0)
    fb = m.predict(black01, verbose=0)
    c01_255 = float(np.dot(f01.ravel(), f255.ravel()) / (np.linalg.norm(f01) * np.linalg.norm(f255) + 1e-8))
    c01_b = float(np.dot(f01.ravel(), fb.ravel()) / (np.linalg.norm(f01) * np.linalg.norm(fb) + 1e-8))
    print(f"   feature norm: [0,1]->{np.linalg.norm(f01):.4f}  [0,255]->{np.linalg.norm(f255):.4f}")
    print(f"   cos(feat(white01), feat(white255)) = {c01_255:.6f}")
    print(f"   cos(feat(white01), feat(black01))  = {c01_b:.6f}")
