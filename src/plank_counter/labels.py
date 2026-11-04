"""Per-plank box labels for YOLO training.

Images 1, 3, 5: seeded from the classical counter, corrected by hand (FIXES).
Images 2, 4 (staggered, shifted planks): drawn by hand as row ranges per column in
annotations/manual_rows.json, reviewed plank by plank against the human count.
"""

import json

import numpy as np

from plank_counter import classical, dataset
from plank_counter.geometry import Quad, as_quad, row_band

# (image, column index, first plank index, last plank index) -> merge into one plank (0-based)
FIXES = [("plank3", 0, 15, 16)]  # one end face split in two by a stray hole row


def merge(planks: list[Quad], first: int, last: int) -> list[Quad]:
    top, bottom = planks[first], planks[last]
    joined = as_quad([top[0], top[1], bottom[2], bottom[3]])
    return [*planks[:first], joined, *planks[last + 1 :]]


def column_planks(name: str) -> list[list[Quad]]:
    image = dataset.load_image(name)
    per_column = [r.planks for r in classical.detect(image, dataset.columns(name))]
    for fix_name, col, first, last in FIXES:
        if fix_name == name:
            per_column[col] = merge(per_column[col], first, last)
    return per_column


def manual_planks(name: str, rows: list) -> list[Quad]:
    return [
        row_band(as_quad(col), y0, y1)
        for col, ranges in zip(dataset.columns(name), rows, strict=True)
        for y0, y1 in ranges
    ]


def planks_for(name: str) -> list[Quad]:
    manual = dataset.read_json("manual_rows.json")
    if name in manual:
        return manual_planks(name, manual[name])
    return [p for col in column_planks(name) for p in col]


def build() -> dict[str, list]:
    """Quads per image; fails loudly if they don't match the human count."""
    boxes = {}
    for name in dataset.image_names():
        planks = planks_for(name)
        if len(planks) != dataset.true_count(name):
            raise ValueError(f"{name}: {len(planks)} boxes vs true count {dataset.true_count(name)}")
        boxes[name] = [np.round(p, 1).tolist() for p in planks]
    return boxes


def write() -> None:
    (dataset.ANNOTATIONS / "boxes.json").write_text(json.dumps(build()) + "\n")


def load() -> dict[str, list[Quad]]:
    raw = dataset.read_json("boxes.json")
    return {name: [as_quad(q) for q in quads] for name, quads in raw.items()}
