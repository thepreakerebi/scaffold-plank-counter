"""Tests run on synthetic stacks, so they never need the confidential photos."""

import numpy as np
import pytest

from plank_counter import classical
from plank_counter.holes import band_edges, group_rows, is_pair
from plank_counter.report import Row, summary
from plank_counter.seams import cut_rows


def synthetic_stack(planks: int, pitch: int = 30, width: int = 160, holes: bool = True) -> np.ndarray:
    """Light plank ends separated by thin dark seams, each with a hole near either end."""
    image = np.full((planks * pitch, width, 3), 200, np.uint8)
    for i in range(planks):
        top = i * pitch
        image[top : top + 2] = 70  # seam
        if holes:
            cy = top + pitch // 2
            for cx in (width // 5, 4 * width // 5):
                image[cy - 2 : cy + 3, cx - 2 : cx + 3] = 20
    return image


def full_frame(image: np.ndarray) -> list:
    h, w = image.shape[:2]
    return [[[0, 0], [w, 0], [w, h], [0, h]]]


@pytest.mark.parametrize("planks", [5, 12, 20])
def test_counts_synthetic_stack(planks):
    image = synthetic_stack(planks)
    [result] = classical.detect(image, full_frame(image))
    assert result.count == planks
    assert result.agrees


def test_no_holes_means_no_planks():
    image = synthetic_stack(8, holes=False)
    [result] = classical.detect(image, full_frame(image))
    assert result.count == 0
    assert not result.agrees  # seams still see planks, so the column is flagged for review


def test_two_columns_add_up():
    left, right = synthetic_stack(6), synthetic_stack(9)
    image = np.zeros((max(len(left), len(right)), 320, 3), np.uint8)
    image[: len(left), :160], image[: len(right), 160:] = left, right
    cols = [
        [[0, 0], [160, 0], [160, len(left)], [0, len(left)]],
        [[160, 0], [320, 0], [320, len(right)], [160, len(right)]],
    ]
    assert len(classical.all_planks(classical.detect(image, cols))) == 15


def test_group_rows_splits_on_y_jumps():
    centres = np.array([[10, 5], [90, 7], [12, 40], [88, 41]], float)
    assert [len(r) for r in group_rows(centres, tolerance=5)] == [2, 2]


def test_is_pair_needs_two_spread_holes():
    assert is_pair(np.array([[10, 5], [90, 6]], float), width=100)
    assert not is_pair(np.array([[50, 5], [55, 6]], float), width=100)
    assert not is_pair(np.array([[50, 5]], float), width=100)


def test_band_edges_bracket_each_row():
    edges = band_edges(np.array([15.0, 45.0, 75.0]), height=90)
    assert edges.tolist() == [0.0, 30.0, 60.0, 90.0]


def test_cut_rows_drops_slivers():
    assert cut_rows(100, np.array([3, 50, 52, 97]), period=50).tolist() == [0, 50, 100]


def test_summary():
    rows = [Row("a", 10, 10, 1.0), Row("b", 8, 10, 3.0)]
    s = summary(rows)
    assert s["mae"] == 1.0 and s["exact"] == 1 and s["total_accuracy"] == 0.9 and s["mean_ms"] == 2.0
