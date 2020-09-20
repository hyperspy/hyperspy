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
import pytest

import hyperspy.api as hs


def test_check_navigation_mask():
    s = hs.signals.Signal1D(np.ones(shape=(32, 32, 1024)))
    s.add_gaussian_noise(0.1)

    mask = (s.sum(-1) > 1)
    assert s._check_navigation_mask(mask) is None

    # wrong navigation shape
    error_message = '`mask` must be a signal with the same `navigation_shape`'
    with pytest.raises(ValueError, match=error_message):
        s._check_navigation_mask(mask.inav[:-2, :])

    # wrong array shape
    error_message = 'The shape of `mask` must match the shape of the `navigation_shape`.'
    with pytest.raises(ValueError, match=error_message):
        s._check_navigation_mask(mask.inav[:-2, :].data)

    # wrong signal dimenstion
    mask = (s > 1)
    error_message = '`mask` must be a signal with `signal_dimension`'
    with pytest.raises(ValueError, match=error_message):
        s._check_navigation_mask(mask)



def test_check_signal_mask():
    s = hs.signals.Signal1D(np.ones(shape=(32, 32, 1024)))
    s.add_gaussian_noise(0.1)

    mask = (s.inav[0, 0] > 1)
    assert s._check_signal_mask(mask) is None

    # wrong navigation shape
    error_message = '`mask` must be a signal with the same `signal_shape`'
    with pytest.raises(ValueError, match=error_message):
        s._check_signal_mask(mask.isig[:-2])

    # wrong array shape
    error_message = 'The shape of `mask` must match the shape of the `signal_shape`.'
    with pytest.raises(ValueError, match=error_message):
        s._check_signal_mask(mask.isig[:-2].data)

    # wrong signal dimenstion
    mask = (s > 1)
    error_message = '`mask` must be a signal with `navigation_dimension`'
    with pytest.raises(ValueError, match=error_message):
        s._check_signal_mask(mask)
