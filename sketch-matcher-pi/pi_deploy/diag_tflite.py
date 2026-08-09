"""
diag_tflite.py — diagnose the deployed TFLite model's input/output behavior.

Checks:
  1. Input tensor dtype/quantization params (scale, zero_point) — is the
     matcher's uint8 conversion mapping BLACK correctly?
  2. Embedding norms + cosine spread for real sketches vs a white image.
  3. Whether the matcher path and a naive uint8 path give the same embedding.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from matcher import SketchMatcher, TFLITE_AVAILABLE
from camera import CameraCapture

MODEL_DIR = Path(__file__).parent / "model_data"
TEST_DIR = Path(__file__).parent.parent / "hpc" / "sketch_test_export"


def main():
    matcher = SketchMatcher(MODEL_DIR)
    interp = matcher.interpreter
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    print(f"\nInput:  dtype={inp['dtype']} shape={inp['shape']} "
          f"quant={inp['quantization']}")
    print(f"Output: dtype={out['dtype']} shape={out['shape']} "
          f"quant={out['quantization']}")
    scale, zp = inp["quantization"]
    print(f"  -> scale={scale}, zero_point={zp}")
    print(f"  BLACK(0.0)  maps to q={0.0/scale + zp} (dequant {scale*(0-zp):.3f})")
    print(f"  WHITE(1.0)  maps to q={1.0/scale + zp} (dequant {scale*(255-zp):.3f})")

    # ---- embed a plain WHITE image (should be the dataset "blank" reference)
    white = np.ones((1, 224, 224, 3), dtype=np.float32)
    e_white = matcher.embed(white)
    print(f"\nWhite image  -> norm={np.linalg.norm(e_white):.3f}")

    # ---- embed several real sketches + a black-filled square
    cam = CameraCapture()
    samples = []
    for cdir in sorted(TEST_DIR.iterdir())[:4]:
        pngs = sorted(cdir.glob("*.png"))[:2]
        for p in pngs:
            img = cv2.imread(str(p))
            samples.append((f"real:{cdir.name.split('_',1)[1]}/{p.stem}", img))

    black = np.zeros((400, 400, 3), dtype=np.uint8)
    samples.append(("filled black square", black))

    embs = {}
    for name, img in samples:
        pre = cam.preprocess(img)
        e = matcher.embed(pre)
        embs[name] = e
        print(f"  {name:28s} norm={np.linalg.norm(e):.3f}")

    # cosine between all pairs of these embeddings
    print("\nCosine matrix (real sketches should differ; black square far):")
    keys = list(embs)
    hdr = "".join(f"{k.split(':')[-1][:7]:>9}" for k in keys)
    print(f"{'':22s}{hdr}")
    for a in keys:
        row = ""
        for b in keys:
            c = float(np.dot(embs[a], embs[b]))
            row += f"{c:9.3f}"
        print(f"{a.split(':')[-1][:22]:22s}{row}")

    # ---- compare matcher path vs naive uint8 path on one sketch
    img = cv2.imread(str(sorted(TEST_DIR.iterdir())[0].glob('*.png')).replace("'", "") if False else
                     str(list((sorted(TEST_DIR.iterdir())[0]).glob('*.png'))[0]))
    pre = cam.preprocess(img)
    e_matcher = matcher.embed(pre)

    iidx = inp["index"]
    naive = np.rint(pre[0] * 255).astype(np.uint8)[np.newaxis, ...]
    interp.set_tensor(iidx, naive)
    interp.invoke()
    e_naive = interp.get_tensor(out["index"])[0]
    n = np.linalg.norm(e_naive)
    e_naive = e_naive / n if n > 0 else e_naive
    print(f"\nmatcher-uint8 vs naive-uint8 cosine: {float(np.dot(e_matcher, e_naive)):.4f}")

    # ---- top-1 for a white image through the DB (sanity)
    results, _, _, rejected = matcher.match(white)
    print(f"\nWhite image top-3: {results} rejected={rejected}")


if __name__ == "__main__":
    main()
