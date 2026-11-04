"""Quad geometry: warp a stack column to a straight rectangle and map cells back."""

import cv2
import numpy as np

Quad = np.ndarray  # shape (4, 2), float32, order: top-left, top-right, bottom-right, bottom-left


def as_quad(points) -> Quad:
    return np.asarray(points, dtype=np.float32).reshape(4, 2)


PATCH_WIDTH = 240  # every column is resampled to this width so thresholds work at any photo size


def rect_size(quad: Quad, width: int = PATCH_WIDTH) -> tuple[int, int]:
    """Size of the straightened rectangle: fixed width, height keeps the column's aspect ratio."""
    tl, tr, br, bl = quad
    src_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    src_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return width, max(int(round(src_h * width / max(src_w, 1))), 1)


def homography(quad: Quad) -> tuple[np.ndarray, tuple[int, int]]:
    w, h = rect_size(quad)
    target = as_quad([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(quad, target), (w, h)


def warp(image: np.ndarray, quad: Quad) -> tuple[np.ndarray, np.ndarray]:
    """Return the straightened patch and the homography that produced it."""
    matrix, size = homography(quad)
    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_CUBIC), matrix


def unwarp_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(matrix)).reshape(-1, 2)


def band_quads(ys: np.ndarray, width: int, matrix: np.ndarray) -> list[Quad]:
    """Horizontal bands between consecutive y cuts in the patch, mapped back to the image."""
    return [
        as_quad(unwarp_points([[0, y0], [width, y0], [width, y1], [0, y1]], matrix))
        for y0, y1 in zip(ys[:-1], ys[1:], strict=True)
    ]


def bbox(quad: Quad) -> tuple[float, float, float, float]:
    """Axis-aligned x0, y0, x1, y1."""
    (x0, y0), (x1, y1) = quad.min(axis=0), quad.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)
