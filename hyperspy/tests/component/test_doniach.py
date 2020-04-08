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
from hyperspy.components1d import Doniach

sqrt2pi = np.sqrt(2 * np.pi)
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))


def test_function():
    g = Doniach()
    g.centre.value = 1
    g.sigma.value = 2 / sigma2fwhm
    g.A.value = 3 * sqrt2pi * g.sigma.value
    g.alpha.value = 1.0e-7
    assert_allclose(g.function(2), 3.151281311482424)
    assert_allclose(g.function(1), 7.519884823893001)


