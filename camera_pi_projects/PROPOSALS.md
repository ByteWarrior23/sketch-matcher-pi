# Camera + Raspberry Pi Project Candidates (shortlist)

Goal: pick ONE camera-based project that is (a) trainable on the HPC,
(b) fully verifiable offline on the Pi, (c) known to reach GOOD accuracy,
(d) makes a believable live demo. Ranked by confidence.

## 1. Traffic Sign Recognition (RECOMMENDED)
- Dataset: GTSRB (German Traffic Sign Benchmark), 43 classes, ~52k images, free download.
- Proven accuracy: fine-tuned MobileNetV3/EfficientNet **97-99%** (paper-standard, not marketing).
- Demo: point the Pi camera at a printed/phone-displayed traffic sign -> label + confidence.
- Pi: int8 TFLite, real-time, fully offline. HPC: 1-2 h training.
- Verification: held-out test split 97%+ is easy to show with a number.
- Risk: low. Lighting/skew handled by data augmentation (already in the dataset).

## 2. Fruit Recognition / Quality Grading
- Dataset: Fruits-360 (131 classes incl. varieties/ripeness, ~90k images).
- Proven accuracy: **99%+** with a small fine-tuned CNN.
- Demo: point camera at a fruit -> "Granny Smith / ripe banana 98%".
- Pi: trivially real-time. HPC: <1 h. Verification: instant, unambiguous numbers.
- Risk: very low. Best "guaranteed good accuracy" option.

## 3. Mask / No-Mask Detection
- Dataset: public mask-wearing datasets (~10k images, 2-3 classes).
- Proven accuracy: **98-99%**.
- Demo: person in front of camera -> MASK/NO MASK + box.
- Needs face detection on the Pi (MediaPipe works). Slightly more moving parts.
- Risk: low-medium (face detector quality dominates).

## 4. Plant Disease Classification
- Dataset: PlantVillage (38 classes of leaf diseases, ~54k images).
- Proven accuracy: **95-98%** (in-dataset).
- Demo: point camera at a leaf -> disease + treatment hint.
- Risk: MEDIUM — real leaves under real light drop accuracy a lot vs dataset;
  best demo uses printed/phone leaf images. Still verifiable but honest numbers lower.

## 5. Food Recognition
- Dataset: Food-101 (101 classes, ~100k images).
- Proven accuracy: **~82-85%** top-1 (fine-tuned MobileNetV3). Real-world lower.
- Risk: medium. Fun demo but accuracy is the weakest of the list.

## NOT recommended this round
- Sketch-to-photo matching (current project) — proven painful for us.
- Handwriting OCR / gesture / ASL — good but add extra parts (detection, segmentation).
- Object detection (COCO) — heavier to make "fully verified" on a Pi demo.

## Decision rule
Fruits-360 (99%+, simplest, bulletproof) OR GTSRB traffic signs (97-99%, most
impressive demo, still bulletproof). Both: one HPC training run, TFLite int8,
real-time on Pi, clear PASS/FAIL eval. I recommend GTSRB if you want the most
"wow" live demo; Fruits-360 if you want the least risk.

## Camera
I WILL need camera access for the final live-demo test (pointing the Pi/laptop
camera at real objects/signs). I will ask again right before opening it.

## DECISION (2026-08-10) — locked in with the user
- **Selected: Traffic Sign Recognition (GTSRB), 43 classes.** Reason: the user
  must PRESENT THIS IN CLASS. GTSRB gives the best combination of proven 97-99%
  accuracy, an instantly-understandable story ("offline mini self-driving vision
  on a Pi"), and a crowd-pleasing live demo (point camera at sign -> label +
  confidence). Fruits-360 kept as backup.
- Class-presentation requirement is a first-class constraint for everything we
  build: story, demo, and a clean accuracy number must all be presentable.
- Notes for the talk: GTSRB = German Traffic Sign Benchmark, 43 classes, ~52k
  images; fine-tuned MobileNetV3/EfficientNet reaches 97-99% top-1; runs real-time
  offline on Raspberry Pi 5 via TFLite int8 (same deploy path we already have).

