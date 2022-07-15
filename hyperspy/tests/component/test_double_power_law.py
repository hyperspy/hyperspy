# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np

from hyperspy.components1d import DoublePowerLaw


def test_function():
    g = DoublePowerLaw()
    g.A.value = 3
    g.r.value = 2
    g.origin.value = 1
    g.shift.value = 2
    g.ratio.value = 2
    assert np.isinf(g.function(1))
    assert np.isinf(g.function(3))
    assert g.function(-1) == 0
    assert g.function(0) == 0
    assert g.function(2) == 9
    np.testing.assert_allclose(g.function(10), 0.15948602)
    assert g.grad_A(2) == 3
    np.testing.assert_allclose(g.grad_r(4), -0.3662041)
    assert g.grad_origin(2)  == -6
    assert g.grad_shift(2)  == -12
    assert g.grad_ratio(2)  == 3
