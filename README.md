# Scaffold Plank Counter

Counts the steel scaffold planks in a photo of a stack and marks each one with a numbered box. Built as a test task for the scaffold template of the **CountX** iOS app.

The user marks each column of the stack with four corners, the same as CountX's range-selection step. The counter finds every plank end inside those columns, draws a numbered box on it and returns the total.

## Results

Each image is counted by a model that **never saw it during training** (leave-one-image-out).

| Image | Correct | YOLO11n | Classical |
|---|---:|---:|---:|
| 1 | 54 | 55 (+1) | 54 (0) |
| 2 | 27 | 26 (−1) | 19 (−8) |
| 3 | 76 | 78 (+2) | 77 (+1) |
| 4 | 18 | 15 (−3) | 10 (−8) |
| 5 | 60 | 62 (+2) | 60 (0) |
| **Planks counted correctly** | | **96.2%** | 92.8% |
| **Mean error per image** | | 1.8 | 3.4 |
| **Time per image** (Apple M4, PyTorch) | | 29 ms | 12 ms |

The shipped model exports to **Core ML**: 5.4 MB (FP16, NMS included) and 2.5 ms per image on an Apple M4 (CPU + GPU + Neural Engine).

## Requirements

- macOS on Apple silicon (training uses the `mps` GPU backend)
- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/thepreakerebi/scaffold-plank-counter.git
cd scaffold-plank-counter
uv sync
```

The five test images are confidential and are **not** in the repository. Place them here:

```
data/raw/plank1.jpg
data/raw/plank2.jpg
data/raw/plank3.jpg
data/raw/plank4.jpg
data/raw/plank5.jpg   # the original is .avif; convert it to .jpg first, e.g. sips -s format jpeg in.avif --out plank5.jpg
```

## Usage

| Command | What it does | Output |
|---|---|---|
| `uv run plank-count classical` | Training-free counter on all five images | `outputs/classical/*.jpg`, `outputs/classical.md` |
| `uv run plank-count yolo` | Trains five held-out YOLO models (first run only) and counts each image with the model that never saw it | `runs/train/holdout_*`, `outputs/yolo/*.jpg`, `outputs/yolo.md` |
| `uv run plank-count hybrid` | Experimental: classical where its two checks agree, YOLO elsewhere | `outputs/hybrid/*.jpg`, `outputs/hybrid.md` |
| `uv run --with "numpy<2.3" plank-count coreml` | Trains the final model on all five images, exports it to Core ML and measures latency | `runs/train/all/weights/last.mlpackage`, `outputs/coreml.md` |

Trained weights are reused on later runs. Delete `runs/` to retrain from scratch. One YOLO model takes about 7–8 minutes on an M4.

## How it works

### 1. Range step

The user marks each stack column as a four-corner shape (`annotations/columns.json` for the test images). Everything outside these shapes is greyed out before counting, so background stacks and side views are never counted. Training uses the same crops as inference.

### 2. Classical counter (no training)

1. **Straighten** each column with a perspective warp to a fixed width, so the same thresholds work at any photo size.
2. **Hole pairs (the count).** Every plank end carries two punched holes. A row with two small dark blobs spread across the column is one plank.
3. **Seams (the check).** The dark lines between planks give an independent count. When the two disagree, the column is flagged for the user to review.

Plank boxes run from seam to seam, so each box covers the plank's top flange, its face and both holes.

### 3. YOLO11n detector

- Fine-tuned from COCO-pretrained `yolo11n.pt`.
- Every training photo is repeated so each epoch sees many augmented versions: mosaic, scale, small rotation and perspective, colour jitter and left-right flips. There are no vertical flips, because stacks are never upside down.
- Three post-processing rules clean up the output:
  1. Drop boxes that sit mostly inside a more confident box (half planks).
  2. Keep only boxes centred inside a marked column.
  3. Merge side-by-side halves of the same plank within a column.

## Labelling rules

The ground-truth counts (`annotations/ground_truth.json`) were made by hand at 3× zoom and checked by a second independent count. Every label follows these rules:

- One plank is one end face, **including its top flange**. The flange is not a separate plank.
- Every plank end has **two holes**.
- Planks shifted in or out of the stack count, even when their holes are hidden.
- Planks cut off by the frame count if the end face can still be identified.

Box labels (`annotations/boxes.json`) come from the classical counter for images 1, 3 and 5 and from hand-drawn row ranges (`annotations/manual_rows.json`) for the staggered images 2 and 4. Building them fails if any image's boxes don't match its hand count.

## Project structure

```
annotations/
  columns.json        column corners marked for each image
  ground_truth.json   hand counts and the counting rule
  manual_rows.json    hand labels for images 2 and 4
  boxes.json          per-plank training boxes for all images
src/plank_counter/
  cli.py              plank-count command
  dataset.py          image and annotation loading
  geometry.py         perspective warp and quad helpers
  holes.py            hole-pair detection and seam-snapped plank edges
  seams.py            seam profile (the classical cross-check)
  classical.py        training-free counter with review flag
  labels.py           builds and validates the training boxes
  roi.py              crops the marked stack, greys out the rest
  yolo.py             training, inference and post-processing
  export.py           Core ML export and latency measurement
  draw.py             numbered-box overlays
  report.py           count-vs-truth results table
tests/                unit tests on synthetic images (no photos needed)
```

## Testing

```bash
uv run pytest
uv run ruff check .
```

The tests build synthetic plank stacks, so they run without the confidential images.

## Core ML notes

- `coremltools` 9 fails on NumPy 2.3 or newer, while Ultralytics on macOS needs a newer NumPy for training. The export command therefore layers an older NumPy on top (`uv run --with "numpy<2.3" …`). `export.py` stops with this command if it is run otherwise.
- The Core ML model takes a fixed 640 × 640 input and has NMS built in. On the five images its counts match the PyTorch model within 1 plank on four images and within 3 on image 4. Match its confidence and IoU thresholds, and check on new photos, before release.

## Limitations

- **Five photos.** The remaining errors are on photo types the model rarely saw, especially the near top-down view in image 4. Collecting 50–100 real stack photos is the main lever.
- **Shifted planks** hide their holes, so the classical counter misses them. Its review flag catches this.
- **Run-to-run variation.** Training on the Apple GPU isn't fully deterministic. Two identical runs can differ by a plank or two per image.

## Author

Elvis Kerebi
