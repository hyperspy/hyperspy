# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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
from numpy.testing import assert_allclose

from hyperspy.components1d import Logistic


def test_function():
    g = Logistic()
    g.a.value = 1
    g.b.value = 2
    g.c.value = 3
    g.origin.value = 4
    assert_allclose(g.function(10), 1)
    assert_allclose(g.function(4), 1/3)
    assert_allclose(g.function(0), 3.07209674e-06)
