# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

from hyperspy.components1d import Arctan


def test_function():
    g = Arctan()
    g.A.value = 10
    g.k.value = 2
    g.x0.value = 1
    np.testing.assert_allclose(g.function(0), -11.07148718)
    np.testing.assert_allclose(g.function(1), 0)
    np.testing.assert_allclose(g.function(1e4), 10 * np.pi / 2, 1e-4)
