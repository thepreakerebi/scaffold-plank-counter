"""Training-free counter: the user marks each stack column as a quad (like CountX's range step).

Two independent signals per column:
- hole pairs (primary): one row of two holes = one plank end
- seams (check): dark lines between plank ends
When they disagree, the column is flagged so the app can ask the user to review it.
"""

from dataclasses import dataclass

import numpy as np

from plank_counter.geometry import Quad, as_quad, band_quads, warp
from plank_counter.holes import plank_rows, seam_edges
from plank_counter.seams import cut_rows, dominant_period, seam_profile, seam_rows


@dataclass(frozen=True)
class ColumnResult:
    planks: list[Quad]
    seam_count: int

    @property
    def count(self) -> int:
        return len(self.planks)

    @property
    def agrees(self) -> bool:
        return abs(self.count - self.seam_count) <= 1


def seam_count(profile: np.ndarray) -> int:
    period = dominant_period(profile)
    return len(cut_rows(len(profile), seam_rows(profile, period), period)) - 1


def detect_column(image: np.ndarray, quad) -> ColumnResult:
    patch, matrix = warp(image, as_quad(quad))
    profile = seam_profile(patch)
    edges = seam_edges(plank_rows(patch), profile)
    planks = band_quads(edges, patch.shape[1], matrix) if len(edges) else []
    return ColumnResult(planks=planks, seam_count=seam_count(profile))


def detect(image: np.ndarray, columns: list) -> list[ColumnResult]:
    return [detect_column(image, quad) for quad in columns]


def all_planks(results: list[ColumnResult]) -> list[Quad]:
    return [plank for result in results for plank in result.planks]
