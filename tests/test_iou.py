from kalphacv import iou
from test_resources import *

def test_bbox_perfect_overlap():
    small_bbox1 = [0, 0, 5, 5]
    small_bbox2 = [0, 0, 5, 5]
    medium_bbox1 = [0, 0, 50, 50]
    medium_bbox2 = [0, 0, 50, 50]
    large_bbox1 = [0, 0, 500, 500]
    large_bbox2 = [0, 0, 500, 500]

    assert iou.calc_iou_bbox(small_bbox1, small_bbox2) == 1
    assert iou.calc_iou_bbox(medium_bbox1, medium_bbox2) == 1
    assert iou.calc_iou_bbox(large_bbox1, large_bbox2) == 1


def test_bbox_positive_partial_overlap():
    # iou > 0.5
    small_bbox1 = [0, 0, 5, 5]
    small_bbox2 = [1, 0, 5, 5]
    medium_bbox1 = [0, 5, 50, 50]
    medium_bbox2 = [5, 0, 50, 50]
    large_bbox1 = [10, 10, 500, 500]
    large_bbox2 = [0, 0, 500, 500]

    assert round(iou.calc_iou_bbox(small_bbox1, small_bbox2), 4) == 0.6667
    assert round(iou.calc_iou_bbox(medium_bbox1, medium_bbox2), 4) == 0.6807
    assert round(iou.calc_iou_bbox(large_bbox1, large_bbox2), 4) == 0.9238


def test_bbox_negative_partial_overlap():
    # iou < 0.5
    small_bbox1 = [0, 0, 5, 5]
    small_bbox2 = [4, 4, 5, 5]
    medium_bbox1 = [0, 0, 50, 50]
    medium_bbox2 = [40, 40, 50, 50]
    large_bbox1 = [70, 80, 500, 300]
    large_bbox2 = [0, 0, 500, 500]

    assert round(iou.calc_iou_bbox(small_bbox1, small_bbox2), 4) == 0.0204
    assert round(iou.calc_iou_bbox(medium_bbox1, medium_bbox2), 4) == 0.0204
    assert round(iou.calc_iou_bbox(large_bbox1, large_bbox2), 4) == 0.4760


def test_bbox_no_overlap():
    small_bbox1 = [0, 0, 5, 5]
    small_bbox2 = [5, 5, 5, 5]
    medium_bbox1 = [100, 100, 50, 50]
    medium_bbox2 = [0, 0, 50, 50]
    large_bbox1 = [0, 0, 500, 500]
    large_bbox2 = [600, 700, 300, 520]

    assert iou.calc_iou_bbox(small_bbox1, small_bbox2) == 0
    assert iou.calc_iou_bbox(medium_bbox1, medium_bbox2) == 0
    assert iou.calc_iou_bbox(large_bbox1, large_bbox2) == 0


def test_seg_perfect_overlap():
    small_seg1 = [0, 0, 5, 2, 4, 5, 1, 4]
    small_seg2 = [0, 0, 5, 2, 4, 5, 1, 4]
    medium_seg1 = [0, 0, 30, 40, 50, 20, 40, 50, 10, 35]
    medium_seg2 = [0, 0, 30, 40, 50, 20, 40, 50, 10, 35]
    large_seg1 = [0, 0, 400, 350, 500, 250, 450, 500, 20, 368]
    large_seg2 = [0, 0, 400, 350, 500, 250, 450, 500, 20, 368]

    assert iou.calc_iou_seg(small_seg1, small_seg2) == 1
    assert iou.calc_iou_seg(medium_seg1, medium_seg2) == 1
    assert iou.calc_iou_seg(large_seg1, large_seg2) == 1


def test_seg_positive_partial_overlap():
    # iou > 0.5
    small_seg1 = [0, 0, 3, 0, 3, 3, 0, 3]
    small_seg2 = [0, 0, 3, 0, 5, 2, 0, 3]
    medium_seg1 = [0, 0, 30, 0, 30, 30, 0, 30]
    medium_seg2 = [0, 0, 30, 0, 50, 20, 0, 30]
    large_seg1 = [200, 400, 321, 598, 468, 600, 645, 550, 512, 435, 671, 345, 397, 304]
    large_seg2 = [236, 600, 394, 343, 625, 335, 546, 442, 611, 543, 400, 600]
    assert round(iou.calc_iou_seg(small_seg1, small_seg2), 4) == 0.6842
    assert round(iou.calc_iou_seg(medium_seg1, medium_seg2), 4) == 0.7127
    assert round(iou.calc_iou_seg(large_seg1, large_seg2), 4) == 0.6444

    pass


def test_seg_negative_partial_overlap():
    # iou < 0.5
    small_seg1 = [0, 0, 3, 0, 3, 3, 0, 3]
    small_seg2 = [0, 0, 6, 0, 4, 3]
    medium_seg1 = [0, 0, 0, 30, 14, 8, 30, 30, 30, 0]
    medium_seg2 = [55, 15, 45, 15, 45, 0, 35, 0, 20, 25, 55, 40]
    large_seg1 = [0, 0, 0, 140, 160, 160, 240, 0]
    large_seg2 = [100, 250, 100, 100, 450, 100, 250, 300]
    assert round(iou.calc_iou_seg(small_seg1, small_seg2), 4) == 0.3333
    assert round(iou.calc_iou_seg(medium_seg1, medium_seg2), 4) == 0.0712
    assert round(iou.calc_iou_seg(large_seg1, large_seg2), 4) == 0.0599


def test_seg_no_overlap():
    small_bbox1 = [10, 10, 15, 12, 14, 15, 11, 14]
    small_bbox2 = [0, 0, 5, 2, 4, 5, 1, 4]
    medium_bbox1 = [100, 100, 130, 140, 150, 120, 140, 150, 110, 135]
    medium_bbox2 = [0, 0, 30, 40, 50, 20, 40, 50, 10, 35]
    large_bbox1 = [0, 0, 400, 350, 500, 250, 450, 500, 20, 368]
    large_bbox2 = [500, 500, 1400, 1350, 1500, 1250, 1450, 1500, 620, 1368]

    assert iou.calc_iou_seg(small_bbox1, small_bbox2) == 0
    assert iou.calc_iou_seg(medium_bbox1, medium_bbox2) == 0
    assert iou.calc_iou_seg(large_bbox1, large_bbox2) == 0
