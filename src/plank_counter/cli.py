"""Command line: run a counter on all test images, save annotated images and a results table."""

import argparse
import statistics
import time
from collections.abc import Callable

import cv2

from plank_counter import classical, dataset
from plank_counter.draw import draw
from plank_counter.report import Row, markdown


def timed(fn: Callable, repeats: int = 10) -> tuple[object, float]:
    """Result of fn() and its median wall time in milliseconds."""
    times, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1000)
    return result, statistics.median(times)


def run_classical(name: str) -> Row:
    image = dataset.load_image(name)
    results, ms = timed(lambda: classical.detect(image, dataset.columns(name)))
    planks = classical.all_planks(results)
    out = dataset.OUTPUTS / "classical"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{name}.jpg"), draw(image, planks, f"{name} (classical)"))
    flagged = not all(r.agrees for r in results)
    return Row(name, len(planks), dataset.true_count(name), ms, flagged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count scaffold steel planks in the test images.")
    parser.add_argument("method", choices=["classical"])
    args = parser.parse_args()
    rows = [run_classical(name) for name in dataset.image_names()]
    table = markdown(rows, f"{args.method} counter")
    (dataset.OUTPUTS / f"{args.method}.md").write_text(table + "\n")
    print(table)
