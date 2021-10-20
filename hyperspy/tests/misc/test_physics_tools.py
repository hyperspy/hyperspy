# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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
import pytest


from hyperspy.misc.physics_tools import bragg_scattering_angle, effective_Z


def test_bragg_angle():
    np.testing.assert_allclose(bragg_scattering_angle(1.0e-9), 0.00370087636)
    np.testing.assert_allclose(bragg_scattering_angle(1.0e-9, E0=100.0), 0.00370087636)
    np.testing.assert_allclose(bragg_scattering_angle(1.0e-9, E0=1.0), 0.0387581632)


def test_effectiveZ():
    np.testing.assert_allclose(effective_Z([(1, 1), (2, 2), (3, 3)]), 1.8984805)
    np.testing.assert_allclose(
        effective_Z([(1, 1), (2, 2), (3, 3)], exponent=1.5), 1.3616755
    )


def test_effectiveZ_errors():
    with pytest.raises(ValueError, match="list of tuples"):
        _ = effective_Z(1)

    with pytest.raises(ValueError, match="list of tuples"):
        _ = effective_Z([1, 2, 3])

