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

import pytest

from hyperspy.signals import (
    Signal1D,
    Signal2D,
)


def test_signal():
    s = Signal1D([10, 10])
    s.axes_manager[0].convert_to_non_uniform_axis()
    with pytest.raises(NotImplementedError):
        s.fft()
    with pytest.raises(NotImplementedError):
        s.ifft()
    with pytest.raises(NotImplementedError):
        s.diff(0)
    with pytest.raises(NotImplementedError):
        s.rebin(scale=[1])
    with pytest.raises(NotImplementedError):
        s.split(number_of_parts=2, axis=0)


def test_signal1d():
    s = Signal1D(([0, 1]))
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
    s = Signal2D([[10, 10], [10, 10]])
    s.axes_manager[0].convert_to_non_uniform_axis()
    with pytest.raises(NotImplementedError):
        s.align2D()


def test_lazy():
    s = Signal1D([10, 10]).as_lazy()
    s.axes_manager[0].convert_to_non_uniform_axis()
    print(s)
    with pytest.raises(NotImplementedError):
        s.diff(0)
