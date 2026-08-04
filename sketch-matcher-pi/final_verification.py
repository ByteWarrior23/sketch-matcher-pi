#!/usr/bin/env python3
"""Final verification script - run before connecting to HPC."""
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_MODULES = [
    "tensorflow", "cv2", "numpy", "sklearn", "tqdm", "kagglehub", "requests",
]

REQUIRED_FILES = [
    "src\\config.py", "src\\model.py", "src\\train.py",
    "pi_deploy\\main.py", "run_training.py",
    "hpc\\job_preprocess.pbs", "hpc\\job_train.pbs", "hpc\\job_finalize.pbs",
]

PASS, FAIL = [], []


def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(("OK  " if ok else "MISS") + "  " + name)


def main():
    print("Python: " + sys.version.split()[0])

    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            check(module, True)
        except Exception as e:
            print("FAIL " + module + "  ->  " + str(e))
            FAIL.append(module)

    for path in REQUIRED_FILES:
        check(path, os.path.exists(os.path.join(BASE_DIR, path)))

    print()
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)
    print("All checks passed - ready for HPC.")


if __name__ == "__main__":
    main()
