"""Find plank ends by their punched holes.

Every steel-plank end face carries two holes near its left and right edges.
In a straightened column patch, one plank = one row holding at least two holes
that sit well apart horizontally. Lone dark blobs (dirt, marker pen) never pair up.
"""

import cv2
import numpy as np


def find_holes(patch: np.ndarray) -> tuple[np.ndarray, float]:
    """(x, y) centres of small, roughly round dark blobs, and their median diameter."""
    width = patch.shape[1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    k = max(int(0.1 * width), 5) | 1
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    lo, hi = (0.015 * width) ** 2, (0.12 * width) ** 2
    aspect = stats[:, 2] / np.maximum(stats[:, 3], 1)
    keep = (stats[:, 4] > lo) & (stats[:, 4] < hi) & (aspect > 0.4) & (aspect < 2.5)
    keep[0] = False  # background component
    diameter = float(np.median(np.sqrt(stats[keep, 4]))) if keep.any() else 0.0
    return centres[keep], diameter


def group_rows(centres: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Split centres into rows: a new row starts when y jumps by more than the tolerance."""
    if not len(centres):
        return []
    ordered = centres[np.argsort(centres[:, 1])]
    breaks = np.where(np.diff(ordered[:, 1]) > tolerance)[0] + 1
    return np.split(ordered, breaks)


def is_pair(row: np.ndarray, width: int) -> bool:
    """At least two holes, spread across 15%+ of the column (one hole near each plank edge)."""
    return len(row) >= 2 and np.ptp(row[:, 0]) >= 0.15 * width


def away_from_edges(ys: np.ndarray, height: int) -> np.ndarray:
    """Drop rows hugging the top or bottom edge of the marked range.

    The user marks the range from the first plank's end face, so holes right on the
    edge belong to the top surface of the stack, not to an end face."""
    if len(ys) < 2:
        return ys
    margin = 0.2 * float(np.median(np.diff(ys)))
    return ys[(ys > margin) & (ys < height - margin)]


def plank_rows(patch: np.ndarray) -> np.ndarray:
    """Mean y of every hole-pair row. Rows split where y jumps by more than one hole diameter."""
    height, width = patch.shape[:2]
    centres, diameter = find_holes(patch)
    rows = group_rows(centres, tolerance=max(2.0 * diameter, 3.0))
    return away_from_edges(np.asarray([r[:, 1].mean() for r in rows if is_pair(r, width)]), height)


def seam_edges(rows: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Plank edges on the real gaps between planks.

    Between two neighbouring hole rows, the edge is the strongest seam (dark line) in the
    seam profile. Above the first and below the last row, search up to one pitch away.
    So each box runs seam to seam: top flange, face and both holes inside, like a person draws it.
    """
    if len(rows) == 0:
        return np.asarray([])
    height = len(profile)
    pitch = float(np.median(np.diff(rows))) if len(rows) > 1 else float(height)
    bounds = np.concatenate([[max(rows[0] - pitch, 0)], rows, [min(rows[-1] + pitch, height - 1)]]).astype(int)
    return np.asarray([a + int(np.argmax(profile[a : b + 1])) for a, b in zip(bounds[:-1], bounds[1:], strict=True)])
