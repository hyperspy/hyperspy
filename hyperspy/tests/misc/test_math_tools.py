
import numpy as np

from hyperspy.misc import math_tools


def test_isfloat_float():
    assert math_tools.isfloat(3.)


def test_isfloat_int():
    assert not math_tools.isfloat(3)


def test_isfloat_npfloat():
    assert math_tools.isfloat(np.float32(3.))


def test_isfloat_npint():
    assert not math_tools.isfloat(np.int16(3))
