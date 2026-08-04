import argparse
import sys
import zipfile
import shutil
import requests
from pathlib import Path
from tqdm import tqdm

try:
    from config import RAW_DIR, DATASETS, LOG_LEVEL
except ModuleNotFoundError:  # Colab: imported as src.download_data
    from src.config import RAW_DIR, DATASETS, LOG_LEVEL

import logging

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Alternative Kaggle dataset slugs (tried in order if one fails)
SKETCHY_SLUGS = ["balraj98/sketchydataset", "sharanyasundar/sketchy-dataset"]
TUBERLIN_SLUGS = ["borismokeev/tuberlin-sketch-dataset",
                  "zara2099/tu-berlin-hand-sketch-image-dataset"]
IMAGENETSKETCH_SLUGS = ["wanghaohan/imagenetsketch"]


def _copy_kagglehub(path, dest_dir):
    """Copy everything kagglehub downloaded into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in Path(path).iterdir():
        dest = dest_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def _normalize_sketchy_layout(dest_dir):
    """
    Kaggle mirrors wrap the data in varying folder structures.
    Find the 'sketch/' and 'photo/' directories wherever they live and
    hoist them to the top level if needed.
    """
    top_sketch = dest_dir / "sketch"
    top_photo = dest_dir / "photo"
    if top_sketch.is_dir() and top_photo.is_dir():
        return True

    found_sketch = None
    found_photo = None
    for d in dest_dir.rglob("*"):
        if d.is_dir() and d.name == "sketch" and found_sketch is None:
            found_sketch = d
        if d.is_dir() and d.name == "photo" and found_photo is None:
            found_photo = d

    moved = False
    if found_sketch is not None and not top_sketch.exists():
        shutil.move(str(found_sketch), str(top_sketch))
        moved = True
    if found_photo is not None and not top_photo.exists():
        shutil.move(str(found_photo), str(top_photo))
        moved = True
    if moved:
        log.info("Sketchy layout normalized (sketch/ + photo/ hoisted to top level).")
    return top_sketch.is_dir() and top_photo.is_dir()


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    log.info(f"Downloading: {url}")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name, total=total, unit="B", unit_scale=True
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
    log.info(f"Download complete: {dest}")


def extract_zip(zip_path: Path, extract_to: Path):
    log.info(f"Extracting: {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    log.info("Extraction complete.")


def download_sketchy():
    dest_dir = DATASETS["sketchy"]["raw_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try kagglehub (needs Kaggle API token at ~/.kaggle/kaggle.json)
    for slug in SKETCHY_SLUGS:
        try:
            import kagglehub
            log.info(f"Downloading Sketchy from Kaggle via kagglehub ({slug})...")
            log.info("NOTE: Requires Kaggle API token. Set up at:")
            log.info("  https://www.kaggle.com/settings -> API -> Create New Token")
            log.info("  Then upload kaggle.json to Colab: ~/.kaggle/kaggle.json")
            path = kagglehub.dataset_download(slug)
            _copy_kagglehub(path, dest_dir)
            if _normalize_sketchy_layout(dest_dir):
                log.info(f"Sketchy downloaded to {dest_dir}")
                return
            else:
                log.warning(f"'{slug}' did not contain sketch/ + photo/ layout.")
        except Exception as e:
            log.warning(f"kagglehub failed for '{slug}': {e}")

    # Check if manually uploaded
    manual_zip = Path("/content/sketchy_dataset.zip")
    if manual_zip.exists():
        log.info("Found manually uploaded sketchy_dataset.zip")
        extract_zip(manual_zip, dest_dir)
        _normalize_sketchy_layout(dest_dir)
        log.info(f"Sketchy extracted to {dest_dir}")
        return

    log.error("Sketchy dataset not available.")
    log.info("=" * 60)
    log.info("MANUAL DOWNLOAD REQUIRED:")
    log.info("  1. Go to: https://www.kaggle.com/datasets/balraj98/sketchydataset")
    log.info("  2. Click 'Download' (requires free Kaggle account)")
    log.info("  3. Rename the downloaded file to 'sketchy_dataset.zip'")
    log.info("  4. Upload it to Colab when prompted")
    log.info("=" * 60)


def manual_upload_prompt():
    """Prompt user to upload manually downloaded dataset zip."""
    from google.colab import files
    log.info("Please upload sketchy_dataset.zip (downloaded from Kaggle)...")
    uploaded = files.upload()
    for fn in uploaded.keys():
        if fn.endswith(".zip"):
            shutil.move(fn, "/content/sketchy_dataset.zip")
            log.info(f"Uploaded: {fn} -> /content/sketchy_dataset.zip")
            return True
    return False


# Verified QuickDraw categories (confirmed to exist in GCS bucket)
QUICKDRAW_CATEGORIES = [
    "airplane", "apple", "baseball", "basketball", "bathtub",
    "bear", "bed", "bicycle", "book", "bread",
    "butterfly", "cake", "camera", "candle", "car",
    "cat", "chair", "church", "circle", "clock",
    "cloud", "cookie", "couch", "cow", "cup",
    "dog", "dolphin", "donut", "door", "drums",
    "duck", "ear", "elephant", "envelope", "eye",
    "fish", "flower", "flying_saucer", "fork", "frog",
    "garden", "giraffe", "grapes", "guitar", "hammer",
    "hand", "hat", "headphones", "helicopter", "ice_cream",
    "key", "knife", "ladder", "lamp", "lightning",
]


def download_quickdraw():
    dest_dir = DATASETS["quickdraw"]["raw_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
    success = 0
    failed = 0

    for category in QUICKDRAW_CATEGORIES:
        filename = f"{category}.npy"
        url = f"{base_url}/{filename}"
        dest_path = dest_dir / filename
        if dest_path.exists():
            success += 1
            continue
        try:
            resp = requests.head(url, timeout=10)
            if resp.status_code != 200:
                failed += 1
                continue
            download_file(url, dest_path)
            success += 1
        except Exception:
            failed += 1

    log.info(f"QuickDraw: {success} downloaded, {failed} skipped.")
    log.info(f"Location: {dest_dir}")


def download_tuberlin():
    dest_dir = DATASETS["tuberlin"]["raw_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    for slug in TUBERLIN_SLUGS:
        try:
            import kagglehub
            log.info(f"Downloading TU-Berlin from Kaggle ({slug})...")
            path = kagglehub.dataset_download(slug)
            _copy_kagglehub(path, dest_dir)
            log.info(f"TU-Berlin downloaded to {dest_dir}")
            return
        except Exception as e:
            log.warning(f"TU-Berlin Kaggle download failed for '{slug}': {e}")

    manual_zip = Path("/content/tuberlin.zip")
    if manual_zip.exists():
        log.info("Found manually uploaded tuberlin.zip")
        extract_zip(manual_zip, dest_dir)
        return

    log.info("=" * 60)
    log.info("TU-Berlin MANUAL DOWNLOAD:")
    log.info("  1. Go to: https://www.kaggle.com/datasets/borismokeev/tuberlin-sketch-dataset")
    log.info("  2. Click Download")
    log.info("  3. Rename to 'tuberlin.zip' and upload when prompted")
    log.info("=" * 60)


def download_imagenetsketch():
    dest_dir = DATASETS["imagenetsketch"]["raw_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    for slug in IMAGENETSKETCH_SLUGS:
        try:
            import kagglehub
            log.info(f"Downloading ImageNet-Sketch from Kaggle ({slug})...")
            path = kagglehub.dataset_download(slug)
            _copy_kagglehub(path, dest_dir)
            log.info(f"ImageNet-Sketch downloaded to {dest_dir}")
            return
        except Exception as e:
            log.warning(f"ImageNet-Sketch download failed for '{slug}': {e}")

    manual_zip = Path("/content/imagenetsketch.zip")
    if manual_zip.exists():
        log.info("Found manually uploaded imagenet-sketch.zip")
        extract_zip(manual_zip, dest_dir)
        return

    log.info("=" * 60)
    log.info("ImageNet-Sketch MANUAL DOWNLOAD:")
    log.info("  1. Go to: https://www.kaggle.com/datasets/wanghaohan/imagenetsketch")
    log.info("  2. Click Download")
    log.info("  3. Rename to 'imagenetsketch.zip' and upload")
    log.info("=" * 60)


def verify_downloads():
    sketchy_dir = DATASETS["sketchy"]["raw_dir"]
    sketch_ok = (sketchy_dir / "sketch").exists()
    photo_ok = (sketchy_dir / "photo").exists()

    if sketch_ok and photo_ok:
        n_sketches = len(list((sketchy_dir / "sketch").rglob("*.png")))
        n_photos = len(list((sketchy_dir / "photo").rglob("*.jpg")))
        log.info(f"Sketchy: {n_sketches} sketches, {n_photos} photos found.")
    else:
        log.warning("Sketchy dataset incomplete.")

    qdraw_dir = DATASETS["quickdraw"]["raw_dir"]
    n_qdraw = len(list(qdraw_dir.glob("*.npy")))
    if n_qdraw > 0:
        log.info(f"QuickDraw: {n_qdraw} category files found.")
    else:
        log.info("QuickDraw: not downloaded (optional).")

    tuberlin_dir = DATASETS["tuberlin"]["raw_dir"]
    n_tuberlin = len(list(tuberlin_dir.rglob("*.png")))
    if n_tuberlin > 0:
        log.info(f"TU-Berlin: {n_tuberlin} sketches found.")
    else:
        log.info("TU-Berlin: not downloaded (optional).")

    inet_dir = DATASETS["imagenetsketch"]["raw_dir"]
    n_inet = len(list(inet_dir.rglob("*.png"))) + len(list(inet_dir.rglob("*.jpg")))
    if n_inet > 0:
        log.info(f"ImageNet-Sketch: {n_inet} images found.")
    else:
        log.info("ImageNet-Sketch: not downloaded (optional).")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Sketch Matcher")
    parser.add_argument("--quickdraw", action="store_true")
    parser.add_argument("--tuberlin", action="store_true")
    parser.add_argument("--imagenetsketch", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--manual-upload", action="store_true",
                        help="Upload manually downloaded dataset zip")
    args = parser.parse_args()

    if args.verify:
        verify_downloads()
        return

    if args.manual_upload:
        manual_upload_prompt()
        return

    log.info("=" * 60)
    log.info("Sketch Matcher - Dataset Downloader")
    log.info("=" * 60)

    log.info("\n[1/4] Downloading Sketchy Dataset...")
    download_sketchy()

    if not (DATASETS["sketchy"]["raw_dir"] / "sketch").exists():
        log.error("Sketchy dataset is MISSING after download. Cannot continue.")
        sys.exit(1)

    if args.quickdraw:
        log.info("\n[2/4] Downloading QuickDraw subset...")
        download_quickdraw()
    else:
        log.info("\n[2/4] Skipping QuickDraw (use --quickdraw to include)")

    if args.tuberlin:
        log.info("\n[3/4] Downloading TU-Berlin...")
        download_tuberlin()
    else:
        log.info("\n[3/4] Skipping TU-Berlin (use --tuberlin to include)")

    if args.imagenetsketch:
        log.info("\n[4/4] Downloading ImageNet-Sketch...")
        download_imagenetsketch()
    else:
        log.info("\n[4/4] Skipping ImageNet-Sketch (use --imagenetsketch to include)")

    log.info("\nVerifying downloads...")
    verify_downloads()

    log.info("\n" + "=" * 60)
    log.info("Download complete!")
    log.info("Next step: python src/preprocess.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
