"""Find the dark seams between stacked plank ends in a straightened column patch.

Each plank end is a bright band; the gap to the next plank is a thin dark line.
A row-wise *median* across the column ignores the punched holes (small dark dots),
so the profile peaks only where a seam runs across the whole column.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def seam_profile(patch: np.ndarray) -> np.ndarray:
    """Seam strength per row: black-hat (thin dark lines) then row median."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    margin = blackhat.shape[1] // 10  # skip column edges, where neighbours bleed in
    core = blackhat[:, margin : blackhat.shape[1] - margin] if margin else blackhat
    return gaussian_filter1d(np.median(core, axis=1), sigma=1.0)


def dominant_period(profile: np.ndarray, min_lag: int = 6) -> float:
    """Plank pitch in pixels: the first strong autocorrelation peak.

    Taking the highest peak instead locks onto multiples of the pitch (2x, 3x)."""
    detrended = profile - gaussian_filter1d(profile, sigma=25)  # remove slow lighting drift
    ac = np.correlate(detrended, detrended, mode="full")[len(detrended) - 1 :][min_lag : len(detrended) // 3]
    peaks, _ = find_peaks(ac)
    if not len(peaks) or ac[peaks].max() <= 0:  # no repeating pattern
        return float(min_lag * 3)
    strong = peaks[ac[peaks] >= 0.5 * ac[peaks].max()]
    return float(strong[0] + min_lag)


def seam_rows(profile: np.ndarray, period: float, prominence: float = 0.25) -> np.ndarray:
    """Rows of seams, at least ~0.6 plank pitch apart and clearly above the background."""
    spread = np.percentile(profile, 90) - np.percentile(profile, 10)
    peaks, _ = find_peaks(profile, distance=max(int(0.6 * period), 1), prominence=prominence * spread)
    return peaks


def cut_rows(height: int, seams: np.ndarray, period: float) -> np.ndarray:
    """Band edges: column top, the seams, column bottom; drop slivers thinner than half a pitch."""
    min_gap = 0.5 * period
    keep = [0]
    for y in np.sort(seams):
        if y - keep[-1] >= min_gap and height - y >= min_gap:
            keep.append(int(y))
    return np.asarray([*keep, height])
