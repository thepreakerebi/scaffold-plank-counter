"""Command line: run a counter on all test images, save annotated images and a results table."""

import argparse
import statistics
import time
from collections.abc import Callable

import cv2
import numpy as np

from plank_counter import classical, dataset
from plank_counter.draw import draw
from plank_counter.geometry import Quad
from plank_counter.report import Row, markdown

Detector = Callable[[np.ndarray, list], tuple[list[Quad], bool]]  # image, columns -> planks, review flag


def timed(fn: Callable, repeats: int = 10) -> tuple[object, float]:
    """Result of fn() and its median wall time in milliseconds (first call is a warm-up)."""
    result = fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1000)
    return result, statistics.median(times)


def evaluate(name: str, method: str, detector: Detector) -> Row:
    image, columns = dataset.load_image(name), dataset.columns(name)
    (planks, flagged), ms = timed(lambda: detector(image, columns))
    out = dataset.OUTPUTS / method
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{name}.jpg"), draw(image, planks, f"{name} ({method})"))
    return Row(name, len(planks), dataset.true_count(name), ms, flagged)


def classical_detector(image: np.ndarray, columns: list) -> tuple[list[Quad], bool]:
    results = classical.detect(image, columns)
    return classical.all_planks(results), not all(r.agrees for r in results)


def training_set(name: str) -> tuple[str, list[str]]:
    """Fold name and training images for a test image: never train on the image being scored."""
    return f"holdout_{name}", [n for n in dataset.image_names() if n != name]


def yolo_detector(name: str) -> Detector:
    """Model for this test image; trains it on first use (delete runs/ to retrain)."""
    from ultralytics import YOLO  # heavy import, only for this method

    from plank_counter import yolo

    fold, train_names = training_set(name)
    weights = yolo.RUNS / "train" / fold / "weights" / "last.pt"
    model = YOLO(str(weights if weights.exists() else yolo.train(train_names, fold)))
    return lambda image, columns: (yolo.predict(model, image, columns), False)


def hybrid_detector(name: str) -> Detector:
    """Per column: the classical count when its two signals agree (fast, exact on clean stacks),
    otherwise the YOLO boxes in that column (handles staggered and shifted planks)."""
    from plank_counter.geometry import bbox
    from plank_counter.yolo import column_of

    learned = yolo_detector(name)

    def detect(image: np.ndarray, columns: list) -> tuple[list[Quad], bool]:
        boxes, _ = learned(image, columns)
        planks = []
        for i, result in enumerate(classical.detect(image, columns)):
            planks += (
                result.planks if result.agrees else [b for b in boxes if column_of(np.array(bbox(b)), columns) == i]
            )
        return planks, False

    return detect


def run_coreml() -> None:
    from plank_counter import export

    package = export.export()
    latency = export.coreml_latency(package)
    text = (
        f"### Core ML export\n\n{package.name}: {export.size_mb(package):.1f} MB (FP16, NMS included), "
        f"median {latency['median_ms']:.1f} ms, p90 {latency['p90_ms']:.1f} ms per image on this Mac (CPU+GPU+ANE)"
    )
    (dataset.OUTPUTS / "coreml.md").write_text(text + "\n")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count scaffold steel planks in the test images.")
    parser.add_argument("method", choices=["classical", "yolo", "hybrid", "coreml"])
    args = parser.parse_args()
    if args.method == "coreml":
        return run_coreml()
    detectors = {
        "classical": lambda name: classical_detector,
        "yolo": yolo_detector,
        "hybrid": hybrid_detector,
    }
    rows = [evaluate(name, args.method, detectors[args.method](name)) for name in dataset.image_names()]
    table = markdown(rows, f"{args.method} counter")
    (dataset.OUTPUTS / f"{args.method}.md").write_text(table + "\n")
    print(table)
