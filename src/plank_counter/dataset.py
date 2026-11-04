"""Load the test images and their annotations. Images are confidential and never committed."""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ANNOTATIONS = ROOT / "annotations"
OUTPUTS = ROOT / "outputs"


def read_json(name: str) -> dict:
    return json.loads((ANNOTATIONS / name).read_text())


def image_names() -> list[str]:
    return sorted(read_json("columns.json"))


def load_image(name: str) -> np.ndarray:
    image = cv2.imread(str(RAW / f"{name}.jpg"))
    if image is None:
        raise FileNotFoundError(f"{RAW / f'{name}.jpg'} missing — see README for how to fetch the images")
    return image


def columns(name: str) -> list:
    return read_json("columns.json")[name]


def true_count(name: str) -> int:
    return read_json("ground_truth.json")[name]["count"]
