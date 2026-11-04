import numpy as np

from plank_counter.geometry import as_quad
from plank_counter.roi import crop_box, roi_crop
from plank_counter.yolo import contained, drop_contained, yolo_line


def test_half_plank_inside_full_plank_is_dropped():
    boxes = np.array([[0, 0, 100, 20], [50, 0, 100, 20], [0, 25, 100, 45]], float)
    kept = drop_contained(boxes, scores=np.array([0.9, 0.6, 0.8]))
    assert kept.tolist() == [[0, 0, 100, 20], [0, 25, 100, 45]]


def test_neighbouring_planks_survive():
    boxes = np.array([[0, 0, 100, 22], [0, 18, 100, 40]], float)  # small overlap at the seam
    assert len(drop_contained(boxes, scores=np.array([0.9, 0.9]))) == 2


def test_contained_share():
    assert contained(np.array([0, 0, 10, 10]), np.array([0, 0, 20, 20])) == 1.0
    assert contained(np.array([0, 0, 10, 10]), np.array([5, 0, 20, 20])) == 0.5


def test_yolo_line_normalises_to_crop():
    quad = as_quad([[20, 10], [60, 10], [60, 30], [20, 30]])
    assert yolo_line(quad, offset=(10, 0), size=(100, 40)) == "0 0.300000 0.500000 0.400000 0.500000"


def test_roi_crop_greys_outside_the_stack():
    image = np.full((400, 400, 3), 255, np.uint8)
    column = [[100, 100], [300, 100], [300, 300], [100, 300]]
    crop, (ox, oy) = roi_crop(image, [column])
    assert (ox, oy) == crop_box([column], image.shape)[:2]
    assert crop[0, 0].tolist() == [114, 114, 114]
    assert crop[200 - oy, 200 - ox].tolist() == [255, 255, 255]
