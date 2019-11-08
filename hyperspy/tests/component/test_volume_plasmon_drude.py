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

from hyperspy.components1d import VolumePlasmonDrude


def test_function():
    g = VolumePlasmonDrude()
    g.intensity.value = 3.0
    g.plasmon_energy.value = 8.0
    g.fwhm.value = 2.0
    assert g.function(0) == 0
    assert_allclose(g.function(8), 12)
    assert_allclose(g.function(1), 9.66524e-2, rtol=1e-6)
    assert_allclose(g.function(30), 1.639867e-2, rtol=1e-6)
