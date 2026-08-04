import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
STEPS = [
    ("Step 1/6: Download datasets", [
        sys.executable, "src/download_data.py", "--quickdraw", "--tuberlin", "--imagenetsketch"
    ]),
    ("Step 2/6: Preprocess images", [
        sys.executable, "src/preprocess.py"
    ]),
    ("Step 3/6: Train (3 stages x 2 phases: teacher + student)", [
        sys.executable, "src/train.py"
    ]),
    ("Step 4/6: Evaluate", [
        sys.executable, "src/evaluate.py"
    ]),
    ("Step 5/6: Export TFLite + embeddings", [
        sys.executable, "src/export_tflite.py"
    ]),
    ("Step 6/6: Package for Pi deploy", [
        sys.executable, "-c", """
import zipfile, sys
from pathlib import Path
ROOT = Path(__file__).parent
model_dir = ROOT / "models"
missing = [n for n in ["sketch_matcher.tflite", "photo_embeddings.npy",
                       "photo_labels.npy", "labels.json"]
           if not (model_dir / n).exists()]
if missing:
    print(f"ERROR: Model files missing: {missing}. Training did not complete!")
    sys.exit(1)
zip_path = ROOT / "pi_deploy.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    pi_dir = ROOT / "pi_deploy"
    for f in pi_dir.rglob("*"):
        zf.write(f, f.relative_to(ROOT))
    for name in ["sketch_matcher.tflite", "photo_embeddings.npy",
                 "photo_labels.npy", "labels.json"]:
        p = model_dir / name
        if p.exists():
            zf.write(p, f"pi_deploy/model_data/{name}")
print(f"Created {zip_path}")
"""]
    ),
]

def main():
    print("=" * 60)
    print("SKETCH MATCHER - Full Training Pipeline")
    print("GPU:", end=" ")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"{gpus[0].name}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("NONE DETECTED! This will be very slow on CPU.")
    except:
        print("UNKNOWN")
    print("=" * 60)

    for name, cmd in STEPS:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED at {name} (exit code {result.returncode})")
            sys.exit(result.returncode)
        print(f"Completed: {name}")

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("pi_deploy.zip is ready for your Raspberry Pi")
    print("=" * 60)

if __name__ == "__main__":
    main()
