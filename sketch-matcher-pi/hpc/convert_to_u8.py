"""Convert the float32 processed arrays to a small uint8 subsample.

The float32 sketches.npy/photos.npy are ~110 GB total. On the shared HPC node
the training process mmaps them and randomly faults pages (pair generation
touches the whole dataset every epoch); under node memory pressure the kernel
OOM-killer SIGKILLs the job even though RSS is small (the mmap page cache
counts against the 32 GB workq cgroup cap).

This writes per-category subsampled uint8 arrays:
  - sketches_u8.npy / photos_u8.npy       (~27.5 GB at full size -> 4x smaller)
  - sketch_labels_u8.npy / photo_labels_u8.npy (aligned with the u8 arrays)
The default cap is MAX_PER_CAT rows per category (tunable via env var), which
keeps total size comfortably inside the 32 GB cap (e.g. 400/cat ~= 100k images
~= 15 GB uint8) and speeds up every epoch. All 125 categories (incl. the 13
held-out test categories) are preserved.

Run via hpc/job_convert_u8.pbs.
"""

import os
import sys
from pathlib import Path

import numpy as np

PROC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Data4/ee_24126016/sketch_matcher/data/processed")

MAX_PER_CAT = int(os.environ.get("MAX_PER_CAT", "400"))


def subsample_indices(labels, max_per_cat):
    """Row indices to keep: at most `max_per_cat` per category (sorted)."""
    order = np.argsort(labels, kind="stable")
    sorted_lab = labels[order]
    keep = []
    n = len(sorted_lab)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_lab[end] == sorted_lab[start]:
            end += 1
        count = min(end - start, max_per_cat)
        keep.append(order[start:start + count])
        start = end
    return np.sort(np.concatenate(keep))


def convert(name_in, name_out, keep, desc):
    src = np.load(PROC / name_in, mmap_mode="r")
    m = len(keep)
    dst = np.lib.format.open_memmap(
        PROC / name_out, mode="w+", dtype=np.uint8,
        shape=(m,) + src.shape[1:])
    CHUNK = 128
    for i in range(0, m, CHUNK):
        j = min(i + CHUNK, m)
        dst[i:j] = np.clip(src[keep[i:j]] * 255.0, 0, 255).astype(np.uint8)
        if (i // CHUNK) % 64 == 0:
            print(f"{desc}: {j}/{m}", flush=True)
    dst.flush()
    del dst, src
    print(f"{desc}: done -> {PROC / name_out}", flush=True)


def main():
    print(f"MAX_PER_CAT = {MAX_PER_CAT}", flush=True)

    sk_lab = np.load(PROC / "sketch_labels.npy")
    ph_lab = np.load(PROC / "photo_labels.npy")
    print(f"sketches: {len(sk_lab)} labels | photos: {len(ph_lab)} labels",
          flush=True)

    sk_keep = subsample_indices(sk_lab, MAX_PER_CAT)
    ph_keep = subsample_indices(ph_lab, MAX_PER_CAT)
    print(f"after subsample: sketches {len(sk_keep)} | photos {len(ph_keep)}",
          flush=True)
    print(f"categories kept: {len(np.unique(sk_lab[sk_keep]))} sketch / "
          f"{len(np.unique(ph_lab[ph_keep]))} photo", flush=True)

    np.save(PROC / "sketch_labels_u8.npy", sk_lab[sk_keep])
    np.save(PROC / "photo_labels_u8.npy", ph_lab[ph_keep])
    print("labels saved", flush=True)

    convert("sketches.npy", "sketches_u8.npy", sk_keep, "sketches")
    convert("photos.npy", "photos_u8.npy", ph_keep, "photos")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
