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

from hyperspy.signal_tools import SpikesRemovalInteractive
from hyperspy.signals import Signal1D


def test_spikes_removal_tool():
    s = Signal1D(np.ones((2, 3, 30)))
    np.random.seed(1)
    s.add_gaussian_noise(1e-5)
    # Add three spikes
    s.data[1, 0, 1] += 2
    s.data[0, 2, 29] += 1
    s.data[1, 2, 14] += 1

    sr = SpikesRemovalInteractive(s)
    sr.threshold = 1.5
    sr.find()
    assert s.axes_manager.indices == (0, 1)
    sr.threshold = 0.5
    assert s.axes_manager.indices == (0, 0)
    sr.find()
    assert s.axes_manager.indices == (2, 0)
    sr.find()
    assert s.axes_manager.indices == (0, 1)
    sr.find(back=True)
    assert s.axes_manager.indices == (2, 0)
    sr.add_noise = False
    sr.apply()
    np.testing.assert_almost_equal(s.data[0, 2, 29], 1, decimal=5)
    assert s.axes_manager.indices == (0, 1)
    sr.apply()
    np.testing.assert_almost_equal(s.data[1, 0, 1], 1, decimal=5)
    assert s.axes_manager.indices == (2, 1)
    np.random.seed(1)
    sr.add_noise = True
    sr.default_spike_width = 3
    sr.interpolator_kind = "Spline"
    sr.spline_order = 3
    sr.apply()
    np.testing.assert_almost_equal(s.data[1, 2, 14], 1, decimal=5)
    assert s.axes_manager.indices == (0, 0)
