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
import pytest

from hyperspy.signal_tools import SpikesRemoval, SpikesRemovalInteractive
from hyperspy.signals import Signal1D


def test_spikes_removal_tool():
    s = Signal1D(np.ones((2, 3, 30)))
    s.add_gaussian_noise(1e-5, random_state=1)
    # Add three spikes
    s.data[1, 0, 1] += 2
    s.data[0, 2, 29] += 1
    s.data[1, 2, 14] += 1

    sr = SpikesRemovalInteractive(s, random_state=1)
    sr._show_derivative_histogram_fired()
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
    np.testing.assert_almost_equal(s.data[0, 2, 29], 1, decimal=4)
    assert s.axes_manager.indices == (0, 1)
    sr.apply()
    np.testing.assert_almost_equal(s.data[1, 0, 1], 1, decimal=4)
    assert s.axes_manager.indices == (2, 1)
    sr.add_noise = True
    sr.default_spike_width = 3
    sr.interpolator_kind = "Spline"
    sr.spline_order = 3
    sr.apply()
    np.testing.assert_almost_equal(s.data[1, 2, 14], 2, decimal=4)
    assert s.axes_manager.indices == (0, 0)


def test_spikes_removal_tool_navigation_dimension_0():
    # Artificial Signal
    s = Signal1D(np.ones(1234))
    # Add a spike
    s.data[333] = 5
    s.add_gaussian_noise(0.01, random_state=1)
    assert s.axes_manager.navigation_dimension == 0

    sr = SpikesRemovalInteractive(s, random_state=1)
    sr.threshold = 1.5
    sr.add_noise = False
    sr.find()
    sr.apply()

    np.testing.assert_allclose(s.data[333], 1, atol=0.02)


@pytest.mark.parametrize(("add_noise, decimal"), [(True, 1), (False, 4)])
def test_spikes_removal_tool_non_interactive(add_noise, decimal):
    s = Signal1D(np.ones((2, 3, 30)))
    s.add_gaussian_noise(1e-5, random_state=1)
    # Add three spikes
    s.data[1, 0, 1] += 2
    s.data[0, 2, 29] += 1
    s.data[1, 2, 14] += 1
    s.metadata.Signal.set_item("Noise_properties.variance", 1e-5)

    sr = s.spikes_removal_tool(threshold=0.5, interactive=False, add_noise=add_noise)
    np.testing.assert_almost_equal(s.data[1, 0, 1], 1, decimal=decimal)
    np.testing.assert_almost_equal(s.data[0, 2, 29], 1, decimal=decimal)
    np.testing.assert_almost_equal(s.data[1, 2, 14], 1, decimal=decimal)
    assert isinstance(sr, SpikesRemoval)


def test_spikes_removal_tool_non_interactive_masking():
    s = Signal1D(np.ones((2, 3, 30)))
    s.add_gaussian_noise(1e-5, random_state=1)
    # Add three spikes
    s.data[1, 0, 1] += 2
    s.data[0, 2, 29] += 1
    s.data[1, 2, 14] += 1

    navigation_mask = np.zeros((2, 3), dtype="bool")
    navigation_mask[1, 0] = True
    signal_mask = np.zeros((30,), dtype="bool")
    signal_mask[28:] = True
    s.spikes_removal_tool(
        threshold=0.5,
        interactive=False,
        add_noise=False,
        navigation_mask=navigation_mask,
        signal_mask=signal_mask,
    )
    np.testing.assert_almost_equal(s.data[1, 0, 1], 3, decimal=5)
    np.testing.assert_almost_equal(s.data[0, 2, 29], 2, decimal=5)
    np.testing.assert_almost_equal(s.data[1, 2, 14], 1, decimal=5)
