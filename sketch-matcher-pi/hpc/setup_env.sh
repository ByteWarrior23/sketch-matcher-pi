#!/bin/bash
# One-time HPC environment setup for the Sketch Matcher project.
# Run from the HPC home directory (~):
#   bash setup_env.sh
# Per NITJ HPC User Manual v2.0. Idempotent-ish; safe to re-run.

set -e
LOG=~/setup_env.log
exec > >(tee -a "$LOG") 2>&1

echo "=== [1/5] Check conda + create env ==="
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx sketch_matcher; then
  conda create -n sketch_matcher python=3.10 -y
else
  echo "env 'sketch_matcher' already exists"
fi
conda activate sketch_matcher

echo "=== [2/5] Install packages (this takes several minutes) ==="
pip install --upgrade pip
pip install "tensorflow[and-cuda]"
pip install numpy opencv-python-headless tqdm requests scikit-learn kagglehub

echo "=== [3/5] Verify TF + package versions ==="
$CONDA_PREFIX/bin/python - <<'PY'
import tensorflow as tf
import cv2, sklearn, tqdm, requests, numpy
print("TF", tf.__version__, "| OpenCV", cv2.__version__, "| sklearn", sklearn.__version__, "| numpy", numpy.__version__)
print("TF-CUDA built with:", tf.sysconfig.get_build_info().get("cuda_version"))
PY

echo "=== [4/5] Kaggle token check ==="
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "kaggle.json present at ~/.kaggle/kaggle.json"
  chmod 600 "$HOME/.kaggle/kaggle.json"
else
  echo "WARNING: ~/.kaggle/kaggle.json MISSING. Upload it (WinSCP) before qsub job_preprocess.pbs."
fi

echo "=== [5/5] Create project dir in /Data ==="
mkdir -p /Data/"$USER"/sketch_matcher
echo "Done. Environment ready. Next: upload the project to /Data/$USER/sketch_matcher"
