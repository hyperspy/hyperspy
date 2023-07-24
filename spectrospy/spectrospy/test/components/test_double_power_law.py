# -*- coding: utf-8 -*-
# Copyright 2007-2023 The SpectroSpy developers
#
# This file is part of SpectroSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SpectroSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SpectroSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np

from spectrospy.components import DoublePowerLaw


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
