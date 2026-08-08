import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras import layers

t = tf.keras.applications.ConvNeXtTiny(include_top=False, weights=None,
                                       input_shape=(224, 224, 3), pooling="avg")
s = tf.keras.applications.MobileNetV3Large(include_top=False, weights=None,
                                           input_shape=(224, 224, 3), pooling="avg")
print("teacher_backbone convnexttiny:", format(t.count_params(), ","))
print("student_backbone mobilenetv3large:", format(s.count_params(), ","))

def head_params(in_dim):
    b1 = layers.Dense(512)
    bn1 = layers.BatchNormalization()
    b2 = layers.Dense(256)
    bn2 = layers.BatchNormalization()
    b3 = layers.Dense(256)
    b1.build((None, in_dim))
    bn1.build((None, 512))
    b2.build((None, 512))
    bn2.build((None, 256))
    b3.build((None, 256))
    head1 = b1.count_params() + bn1.count_params()
    head2 = b2.count_params() + bn2.count_params()
    head3 = b3.count_params()
    return head1, head2, head3

for name, backbone, in_dim in [
    ("teacher convnexttiny", t, 768),
    ("student mobilenetv3large", s, 960),
]:
    h1, h2, h3 = head_params(in_dim)
    ht = h1 + h2 + h3
    print(f"{name}: backbone={backbone.count_params():,} | D512+BN={h1:,} | D256+BN={h2:,} | embed256={h3:,} | head={ht:,} | TOTAL={backbone.count_params() + ht:,}")
