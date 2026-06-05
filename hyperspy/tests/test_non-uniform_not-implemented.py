# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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


def test_signal():
    s = hs.signals.Signal1D([10, 10])
    s.axes_manager[0].convert_to_non_uniform_axis()
    with pytest.raises(NotImplementedError):
        s.fft()
    with pytest.raises(NotImplementedError):
        s.ifft()
    with pytest.raises(NotImplementedError):
        s.diff(0)
    with pytest.raises(NotImplementedError):
        s.split(number_of_parts=2, axis=0)


def test_signal1d():
    s = hs.signals.Signal1D(([0, 1]))
    s.axes_manager[0].convert_to_non_uniform_axis()
    with pytest.raises(NotImplementedError):
        s.calibrate()
    with pytest.raises(NotImplementedError):
        s.shift1D([1])
    with pytest.raises(NotImplementedError):
        s.estimate_shift1D([1])
    with pytest.raises(NotImplementedError):
        s.smooth_savitzky_golay()
    with pytest.raises(NotImplementedError):
        s.smooth_tv()
    with pytest.raises(NotImplementedError):
        s.filter_butterworth()
    with pytest.raises(NotImplementedError):
        s.gaussian_filter(1)


def test_signal2d():
    s = hs.signals.Signal2D([[10, 10], [10, 10]])
    s.axes_manager[0].convert_to_non_uniform_axis()
    with pytest.raises(NotImplementedError):
        s.align2D()


def test_lazy():
    s = hs.signals.Signal1D([10, 10]).as_lazy()
    s.axes_manager[0].convert_to_non_uniform_axis()
    print(s)
    with pytest.raises(NotImplementedError):
        s.diff(0)


def test_rebin():
    s = hs.signals.Signal1D(np.arange(100).reshape(10, 10))
    s.axes_manager[-1].convert_to_non_uniform_axis()
    s.rebin(scale=(2, 1))
    s.rebin(new_shape=(5, 10))
    with pytest.raises(NotImplementedError):
        s.rebin(scale=(1, 2))
    with pytest.raises(NotImplementedError):
        s.rebin(new_shape=(1, 5))
