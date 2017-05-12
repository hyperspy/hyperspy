
import numpy as np

import hyperspy.api as hs
from hyperspy.gui_ipywidgets.tests.utils import KWARGS


def test_span_roi():
    roi = hs.roi.SpanROI(left=0, right=10)
    wd = roi.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["left"].value == 0
    assert wd["right"].value == 10
    wd["left"].value = -10
    wd["right"].value = 0
    assert roi.left == -10
    assert roi.right == 0


def test_point_1d_roi():
    roi = hs.roi.Point1DROI(value=5.5)
    wd = roi.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["value"].value == 5.5
    wd["value"].value = 0
    assert roi.value == 0


def test_point2d():
    roi = hs.roi.Point2DROI(x=0, y=10)
    wd = roi.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["x"].value == 0
    assert wd["y"].value == 10
    wd["x"].value = -10
    wd["y"].value = 0
    assert roi.x == -10
    assert roi.y == 0


def test_rectangular_roi():
    roi = hs.roi.RectangularROI(left=0, right=10, top=-10, bottom=0)
    wd = roi.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["left"].value == 0
    assert wd["right"].value == 10
    assert wd["top"].value == -10
    assert wd["bottom"].value == 0
    wd["left"].value = -10
    wd["right"].value = 0
    wd["bottom"].value = 1.2
    wd["top"].value = 1.1
    assert roi.left == -10
    assert roi.right == 0
    assert roi.top == 1.1
    assert roi.bottom == 1.2


def test_circle_roi():
    roi = hs.roi.CircleROI(cx=0, cy=0, r=1, r_inner=0.5)
    wd = roi.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["cx"].value == 0
    assert wd["cy"].value == 0
    assert wd["radius"].value == 1
    assert wd["inner_radius"].value == 0.5
    wd["cx"].value = 1
    wd["cy"].value = 2
    wd["radius"].value = 4
    wd["inner_radius"].value = 1.5
    assert roi.cx == 1
    assert roi.cy == 2
    assert roi.r == 4
    assert roi.r_inner == 1.5


def test_line2d_roi():
    roi = hs.roi.Line2DROI(x1=0, x2=10, y1=0, y2=10, linewidth=2)
    wd = roi.gui(**KWARGS)["ipywidgets"]["wdict"]
    assert wd["x1"].value == 0
    assert wd["x2"].value == 10
    assert wd["y1"].value == 0
    assert wd["y2"].value == 10
    assert wd["linewidth"].value == 2
    wd["x1"].value = 12
    wd["x2"].value = 23
    wd["y1"].value = -12
    wd["y2"].value = -23
    wd["linewidth"].value = 100
    assert roi.x1 == 12
    assert roi.x2 == 23
    assert roi.y1 == -12
    assert roi.y2 == -23
    assert roi.linewidth == 100
