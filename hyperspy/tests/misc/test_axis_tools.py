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


from numpy import arange

from hyperspy.axes import DataAxis, FunctionalDataAxis, UniformDataAxis
from hyperspy.misc.axis_tools import check_axes_calibration


def test_check_axes_calibration():
    axisDA1 = DataAxis(axis=arange(10) ** 2)
    axisDA2 = DataAxis(axis=arange(10))
    axisDA3 = DataAxis(axis=arange(10), units="s")
    axisDA4 = DataAxis(axis=arange(10), units="seconds")
    expression = "x ** power"
    axisFDA1 = FunctionalDataAxis(size=10, expression=expression, power=2)
    axisFDA2 = FunctionalDataAxis(size=12, expression=expression, power=2)
    axisUDA1 = UniformDataAxis(size=10, scale=1, offset=10)
    axisUDA2 = UniformDataAxis(size=10, scale=1, offset=0)
    assert check_axes_calibration(axisDA1, axisDA1)
    assert not check_axes_calibration(axisDA1, axisDA2)
    assert check_axes_calibration(axisDA3, axisDA4)
    assert check_axes_calibration(axisFDA1, axisFDA1)
    assert not check_axes_calibration(axisFDA1, axisFDA2)
    assert check_axes_calibration(axisUDA1, axisUDA1)
    assert not check_axes_calibration(axisUDA1, axisUDA2)
    assert not check_axes_calibration(axisDA1, axisUDA1)
    assert not check_axes_calibration(axisFDA1, axisUDA1)
    assert check_axes_calibration(axisDA1, axisFDA1)
    assert check_axes_calibration(axisDA2, axisUDA2)
