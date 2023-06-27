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
import pytest

import hyperspy.api as hs


def test_check_navigation_mask():
    s = hs.signals.Signal1D(np.arange(2*3*4).reshape(3, 2, 4))
    navigation_mask = s.sum(-1)
    s._check_navigation_mask(navigation_mask)
    with pytest.raises(ValueError):
        s._check_navigation_mask(navigation_mask.T)


def test_check_signal_mask():
    s = hs.signals.Signal1D(np.arange(2*3*4).reshape(3, 2, 4))
    signal_mask = s.sum([0, 1])
    s._check_signal_mask(signal_mask)
    with pytest.raises(ValueError):
        s._check_signal_mask(signal_mask.T)
