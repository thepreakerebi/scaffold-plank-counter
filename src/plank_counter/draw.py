"""Overlay detections on an image: outline + running number, like the CountX result screen."""

import cv2
import numpy as np

from plank_counter.geometry import Quad

GREEN, WHITE, RED = (60, 200, 60), (255, 255, 255), (40, 40, 220)


def draw(image: np.ndarray, planks: list[Quad], label: str = "", scale: float = 2.0) -> np.ndarray:
    canvas = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for n, quad in enumerate(planks, start=1):
        pts = (quad * scale).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=GREEN, thickness=2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.circle(canvas, (cx, cy), 11, RED, -1)
        text = str(n)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    banner = f"{label}  count: {len(planks)}".strip()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (30, 30, 30), -1)
    cv2.putText(canvas, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    return canvas
