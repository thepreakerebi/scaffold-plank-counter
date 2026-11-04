"""Core ML export for the iOS app, plus an on-device-style latency check on this Mac.

The exported model takes the cropped stack (after the user's range step) at 640x640
and returns boxes with NMS built in, so Swift only needs Vision's VNCoreMLRequest,
then the same "drop contained boxes" step and a count.
"""

import statistics
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image
from ultralytics import YOLO

from plank_counter import dataset, yolo
from plank_counter.roi import roi_crop

FOLD = "all"  # shipped model: trained on every labelled photo


def final_weights() -> Path:
    weights = yolo.RUNS / "train" / FOLD / "weights" / "last.pt"
    return weights if weights.exists() else yolo.train(dataset.image_names(), FOLD)


def export(half: bool = True) -> Path:
    """Write the .mlpackage (FP16 by default: half the size, same accuracy on the ANE)."""
    path = YOLO(str(final_weights())).export(format="coreml", imgsz=yolo.IMGSZ, nms=True, half=half)
    return Path(path)


def coreml_latency(package: Path, repeats: int = 30) -> dict[str, float]:
    """Median prediction time per image in ms, on all compute units (CPU + GPU + Neural Engine)."""
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
    input_name = model.get_spec().description.input[0].name
    times = []
    for name in dataset.image_names():
        crop, _ = roi_crop(dataset.load_image(name), dataset.columns(name))
        image = Image.fromarray(crop[..., ::-1]).resize((yolo.IMGSZ, yolo.IMGSZ))
        model.predict({input_name: image})  # warm-up
        for _ in range(repeats):
            start = time.perf_counter()
            model.predict({input_name: image})
            times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(times), "p90_ms": float(np.percentile(times, 90))}


def size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
