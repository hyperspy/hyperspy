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

from hyperspy.components1d import Bleasdale


def test_function():
    g = Bleasdale()
    g.a.value = 1
    g.b.value = 2
    g.c.value = 2
    assert g.function(-0.5) == 0
    assert g.function(0) == 1
    assert g.function(12) == 0.2
    np.testing.assert_allclose(g.function(-.48),5)
    assert g.grad_a(0) == -0.5
    assert g.grad_b(0) == 0
    assert g.grad_c(0) == 0
    assert g.grad_a(-1) == 0
    assert g.grad_b(-1) == 0
