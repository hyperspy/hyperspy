# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

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
