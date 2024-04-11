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

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass


@pytest.mark.parametrize("lazy", (True, False))
def test_setting_ragged_array(lazy):
    s = hs.signals.Signal1D(np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5))
    assert not s.ragged
    assert s.axes_manager.signal_shape == (5,)
    assert s.axes_manager.navigation_shape == (4, 3, 2)
    assert s.axes_manager.signal_indices_in_array == (3,)
    assert s.axes_manager.navigation_indices_in_array == (2, 1, 0)
    with pytest.raises(ValueError):
        s.ragged = True

    data = np.empty((2, 3, 4), dtype=object)
    data.fill(np.array([10, 20, 30, 40, 50]))
    s = hs.signals.BaseSignal(data, ragged=True)
    if lazy:
        s = s.as_lazy()
    assert s.ragged
    assert s.axes_manager.signal_shape == ()
    assert s.axes_manager.navigation_shape == (4, 3, 2)
    assert s.axes_manager.signal_indices_in_array == ()
    assert s.axes_manager.navigation_indices_in_array == (2, 1, 0)
    if lazy:
        with pytest.raises(NotImplementedError):
            s.ragged = False
    else:
        s.ragged = False
        assert not s.ragged
        assert s.axes_manager.signal_shape == (5,)
        assert s.axes_manager.navigation_shape == (4, 3, 2)
        assert s.axes_manager.signal_indices_in_array == (3,)
        assert s.axes_manager.navigation_indices_in_array == (2, 1, 0)


@lazifyTestClass
class TestRaggedArray:
    def setup_method(self, method):
        data = np.empty((3, 4), dtype=object)
        data.fill(np.array([10, 20]))
        s = hs.signals.BaseSignal(data, ragged=True)
        self.s = s

    def test_axes_manager(self):
        s = self.s
        class_ = "LazySignal" if s._lazy else "BaseSignal"
        assert s.__repr__() == f"<{class_}, title: , dimensions: (4, 3|ragged)>"

    def test_transpose(self):
        with pytest.raises(RuntimeError):
            self.s.T

    def test_slicing(self):
        s = self.s
        s2 = s.inav[0]
        assert s2.ragged
        assert s2.axes_manager.signal_dimension == 0
        assert s2.axes_manager.signal_shape == ()
        assert s2.axes_manager.navigation_shape == (3,)
        s3 = s.inav[0, 0]
        assert s3.ragged
        assert s3.axes_manager.signal_dimension == 0
        assert s3.axes_manager.signal_shape == ()
        assert s3.axes_manager.navigation_shape == ()

        with pytest.raises(RuntimeError):
            s.isig[0]


def test_create_ragged_array():
    data = np.array([[0, 1], [2, 3, 4]], dtype=object)
    s = hs.signals.BaseSignal(data, ragged=True)
    assert s.axes_manager.ragged

    with pytest.raises(ValueError):
        s.ragged = False

    data = np.empty((1,), dtype=object)
    data.fill(np.array([[0, 0], [25, -25], [-25, 25]]))

    s2 = hs.signals.BaseSignal(data, ragged=True)
    assert s2.axes_manager.ragged
    assert s2.__repr__() == "<BaseSignal, title: , dimensions: (|ragged)>"


def test_Signal1D_Signal2D_ragged():
    data = np.array((1, 2))
    with pytest.raises(ValueError):
        _ = hs.signals.Signal1D(data, ragged=True)

    with pytest.raises(ValueError):
        _ = hs.signals.Signal2D(data, ragged=True)


def test_conversion_signal():
    data = np.empty((2, 3, 4), dtype=object)
    data.fill(np.array([10, 20, 30, 40, 50]))
    s = hs.signals.BaseSignal(data, ragged=True)

    with pytest.raises(ValueError):
        s.axes_manager._set_signal_dimension(1)

    with pytest.raises(RuntimeError):
        s.as_signal1D(0)
