"""Keep only the stack the user marked: grey out everything else, crop to it.

Training and inference share this step, mirroring CountX's range-selection screen,
so background stacks and side views never reach the model.
"""

import cv2
import numpy as np

GREY = (114, 114, 114)  # YOLO's own letterbox colour, so the masked area reads as "nothing"


def crop_box(columns: list, shape: tuple[int, ...], margin: float = 0.03) -> tuple[int, int, int, int]:
    pts = np.concatenate([np.asarray(q, np.float32) for q in columns])
    h, w = shape[:2]
    pad = margin * max(h, w)
    x0, y0 = np.maximum(pts.min(axis=0) - pad, 0).astype(int)
    x1, y1 = np.minimum(pts.max(axis=0) + pad, [w, h]).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


def roi_crop(image: np.ndarray, columns: list) -> tuple[np.ndarray, tuple[int, int]]:
    """The marked stack on a grey background, and the (x, y) offset of the crop."""
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(q, np.int32) for q in columns], 255)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    masked = np.where(mask[..., None] > 0, image, np.array(GREY, np.uint8))
    x0, y0, x1, y1 = crop_box(columns, image.shape)
    return masked[y0:y1, x0:x1], (x0, y0)
