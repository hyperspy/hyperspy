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


from packaging.version import Version

import pytest
import sympy
import numpy as np

from hyperspy.components1d import Erf

pytestmark = pytest.mark.skipif(Version(sympy.__version__) < Version("1.3"),
                                reason="This test requires SymPy >= 1.3")

def test_function():
    g = Erf()
    g.A.value = 1
    g.sigma.value = 2
    g.origin.value = 3
    assert g.function(3) == 0.
    np.testing.assert_allclose(g.function(15),0.5)
    np.testing.assert_allclose(g.function(1.951198),-0.2,rtol=1e-6)
