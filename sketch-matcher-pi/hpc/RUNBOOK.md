# HPC Runbook - NITJ H100 cluster (workq/cpuq)

All commands run on the HPC unless marked [LAPTOP]. Replace `<username>` with
your NITJ HPC username.

## 0. Local validation (before any submission)
[LAPTOP] Project must compile locally first (already verified):

    cd sketch-matcher-pi
    python -m py_compile src/*.py

## 1. First login + environment setup (once)
    ssh <username>@10.10.11.201
    mkdir -p /Data/<username>/sketch_matcher
    cd ~
    conda create -n sketch_matcher python=3.10 -y
    conda activate sketch_matcher
    pip install "tensorflow[and-cuda]"
    pip install opencv-python-headless tqdm requests scikit-learn

Kaggle credentials (needed only if downloading datasets on the HPC):

    mkdir -p ~/.kaggle
    # upload kaggle.json to ~/.kaggle/kaggle.json via scp/WinSCP

## 2. Upload the project (once)
[LAPTOP]

    scp -r sketch-matcher-pi <username>@10.10.11.201:/Data/<username>/sketch_matcher/

Run from /Data/<username>/sketch_matcher (NOT home; home is purged after 15 days).

## 3. Submit jobs IN ORDER

    ssh <username>@10.10.11.201
    cd /Data/<username>/sketch_matcher

Step A - download + preprocess (cpuq, max 16 cores):
    qsub hpc/job_preprocess.pbs

Step B - wait, then train (workq, GPU; holds the H100 the whole run):
    qstat -u <username>      # Q = queued, R = running, E = finished
    qsub hpc/job_train.pbs

Step C - wait, then evaluate + export TFLite (cpuq, no GPU needed):
    qsub hpc/job_finalize.pbs

Debugging:
    tracejob <JobID>
    cat preprocess.log / train.log / finalize.log

## 4. OOM on a small MIG slice
    nvidia-smi -L              # see which MIG device you got
If the slice is small (e.g. 1g.10gb), uncomment in hpc/job_train.pbs:

    export SKETCH_BATCH1=32
    export SKETCH_BATCH2=32
    export SKETCH_BATCH3=16

## 5. Pull results back
[LAPTOP]
    scp -r <username>@10.10.11.201:/Data/<username>/sketch_matcher/models/ .
    scp -r <username>@10.10.11.201:/Data/<username>/sketch_matcher/pi_deploy/model_data/ .

## 6. Deploy to Pi
Copy tflite + photo_embeddings.npy + labels.json into pi_deploy/model_data/,
then run pi_deploy/main.py on the Pi.

## Rules that matter
- workq = GPU ONLY; CPU-only jobs get auto-killed (that's why finalize is in cpuq).
- 1 GPU per user at a time.
- Big data lives in /Data, never home.
- hpc@nitj.ac.in is for system issues only.

## Timing (rough, H100 + current config)
    Preprocess (cpuq):            2-4 h
    Teacher, 3 stages (workq):    ~24-48 h (early-stopped; cap 100/60/150 epochs)
    Teacher embeddings cache:     ~1 h
    Student, 3 stages:            ~8-24 h (MobileNetV2, ~3-5x faster)
    Eval + TFLite (cpuq):         ~1 h
    Total GPU time:               ~2-4 days -> fits the 15-day window with buffer.

CAUTION: job_train.pbs walltime is 120h (5 days) because teacher+student run in
ONE process. If the cluster rejects walltime > 24h, check the max allowed on
workq and, if needed, split train.py into teacher/student phases.
