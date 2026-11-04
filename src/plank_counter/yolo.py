"""YOLO plank detector trained from very few photos.

Small-data choices:
- start from COCO-pretrained YOLO11s (edges, corners, metal texture already learned)
- every training photo is repeated so one epoch sees many random augmentations:
  mild zoom/shift, small rotation/perspective, colour jitter, left-right flips
- no mosaic and little zoom-in: those slice planks at the cut, and a sliced plank
  labelled "plank" teaches the model to fire on half planks
- the model only sees the marked stack (roi_crop), exactly as at inference
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from plank_counter import dataset, labels
from plank_counter.geometry import Quad, as_quad, bbox
from plank_counter.roi import roi_crop

RUNS = dataset.ROOT / "runs"
BASE_WEIGHTS = "yolo11s.pt"
IMGSZ = 640

AUGMENT = dict(
    mosaic=0.0,  # never stitch crops: it slices planks
    close_mosaic=0,
    scale=0.2,  # mostly whole planks stay whole
    translate=0.05,
    degrees=5.0,
    perspective=0.0005,
    fliplr=0.5,
    flipud=0.0,  # stacks are never upside down in the app
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
)


def yolo_line(quad: Quad, offset: tuple[int, int], size: tuple[int, int]) -> str:
    """One label line: class 0, centre and size normalised to the crop."""
    x0, y0, x1, y1 = bbox(quad - np.asarray(offset, np.float32))
    w, h = size
    x0, x1 = np.clip([x0, x1], 0, w)
    y0, y1 = np.clip([y0, y1], 0, h)
    return f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def write_fold(train: list[str], fold: str, per_epoch: int = 80) -> Path:
    """Write the cropped training images (repeated to ~per_epoch per epoch) with labels, and the dataset yaml."""
    copies = max(per_epoch // len(train), 1)
    root = RUNS / "data" / fold
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    boxes = labels.load()
    for name in train:
        crop, offset = roi_crop(dataset.load_image(name), dataset.columns(name))
        text = "\n".join(yolo_line(q, offset, (crop.shape[1], crop.shape[0])) for q in boxes[name])
        for i in range(copies):
            cv2.imwrite(str(root / "images" / f"{name}_{i:02d}.jpg"), crop)
            (root / "labels" / f"{name}_{i:02d}.txt").write_text(text + "\n")
    yaml = root / "data.yaml"
    # no separate validation photo exists, so validation reuses the training crops (only for logging)
    yaml.write_text(f"path: {root}\ntrain: images\nval: images\nnames:\n  0: plank\n")
    return yaml


def train(train_names: list[str], fold: str, epochs: int = 60) -> Path:
    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=str(write_fold(train_names, fold)),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=8,
        device="mps",
        project=str(RUNS / "train"),
        name=fold,
        exist_ok=True,
        val=False,
        plots=False,
        verbose=False,
        seed=0,
        **AUGMENT,
    )
    return RUNS / "train" / fold / "weights" / "last.pt"


def contained(inner: np.ndarray, outer: np.ndarray) -> float:
    """Share of the inner box's area that lies inside the outer box."""
    w = max(min(inner[2], outer[2]) - max(inner[0], outer[0]), 0)
    h = max(min(inner[3], outer[3]) - max(inner[1], outer[1]), 0)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return float(w * h / area) if area > 0 else 0.0


def drop_contained(boxes: np.ndarray, scores: np.ndarray, limit: float = 0.7) -> np.ndarray:
    """Remove half-plank boxes that sit inside a more confident box.

    Normal NMS misses them: a half box overlaps its parent only ~50% by IoU."""
    kept: list[int] = []
    for i in np.argsort(-scores):
        if all(contained(boxes[i], boxes[j]) < limit for j in kept):
            kept.append(int(i))
    return boxes[sorted(kept, key=lambda i: (boxes[i][1], boxes[i][0]))]


def column_of(box: np.ndarray, columns: list) -> int | None:
    """Index of the marked column holding the box centre; None when it is outside them all."""
    centre = (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)
    hits = [i for i, q in enumerate(columns) if cv2.pointPolygonTest(np.asarray(q, np.float32), centre, False) >= 0]
    return hits[0] if hits else None


def side_by_side(a: np.ndarray, b: np.ndarray, limit: float = 0.6) -> bool:
    """Two halves of one plank: mostly the same rows, but barely overlapping across.

    Thin planks stacked on top of each other also share rows at their edges, but they
    overlap almost fully across, so they are never merged."""
    rows = min(a[3], b[3]) - max(a[1], b[1])
    across = min(a[2], b[2]) - max(a[0], b[0])
    return rows > limit * min(a[3] - a[1], b[3] - b[1]) and across < 0.3 * min(a[2] - a[0], b[2] - b[0])


def merge_same_row(boxes: np.ndarray, columns: list) -> np.ndarray:
    """Join side-by-side halves: in one column, a plank spans the whole width, so two boxes
    on the same rows are one plank split in two."""
    merged: list[tuple[int, np.ndarray]] = []
    for box in boxes:
        col = column_of(box, columns)
        match = next((k for k, (c, m) in enumerate(merged) if c == col and side_by_side(box, m)), None)
        if match is None:
            merged.append((col, box))
        else:
            m = merged[match][1]
            merged[match] = (
                col,
                np.array([min(m[0], box[0]), min(m[1], box[1]), max(m[2], box[2]), max(m[3], box[3])]),
            )
    return np.array([m for _, m in merged]).reshape(-1, 4)


def predict(model: YOLO, image: np.ndarray, columns: list, conf: float = 0.25) -> list[Quad]:
    """Plank boxes in full-image coordinates, top to bottom.

    Clean-up after the model: drop half boxes inside fuller ones, keep only boxes centred in a
    marked column, then join side-by-side halves within a column."""
    crop, offset = roi_crop(image, columns)
    result = model.predict(crop, imgsz=IMGSZ, conf=conf, iou=0.5, max_det=500, verbose=False)[0]
    boxes = drop_contained(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
    boxes = boxes + np.tile(np.asarray(offset, np.float32), 2)
    boxes = merge_same_row(np.array([b for b in boxes if column_of(b, columns) is not None]).reshape(-1, 4), columns)
    return [as_quad([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) for x0, y0, x1, y1 in boxes]
