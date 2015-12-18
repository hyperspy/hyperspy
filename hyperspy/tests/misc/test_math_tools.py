import nose.tools
import numpy as np

from hyperspy.misc import math_tools


def test_isfloat_float():
    nose.tools.assert_true(math_tools.isfloat(3.))


def test_isfloat_int():
    nose.tools.assert_false(math_tools.isfloat(3))


def test_isfloat_npfloat():
    nose.tools.assert_true(math_tools.isfloat(np.float32(3.)))


def test_isfloat_npint():
    nose.tools.assert_false(math_tools.isfloat(np.int16(3)))
