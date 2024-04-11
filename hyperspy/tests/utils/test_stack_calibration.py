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

import logging

import numpy as np
import pytest

from hyperspy import signals
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc.utils import stack


@lazifyTestClass
class TestMismatchedCalibration1D:
    def setup_method(self, method):
        self.s1 = signals.Signal1D([1.0, 2.0, 3.0])
        self.s2 = signals.Signal1D([1.0, 2.0, 3.0])

        self.s1.axes_manager[0].scale = 1.0
        self.s1.axes_manager[0].offset = 0.0
        self.s1.axes_manager[0].units = ""

        self.s2.axes_manager[0].scale = 2.0
        self.s2.axes_manager[0].offset = 0.00001
        self.s2.axes_manager[0].units = "mm"

    @pytest.mark.parametrize("axis", [0, None])
    def test_no_warning(self, axis, caplog):
        with caplog.at_level(logging.WARNING):
            _ = stack([self.s1, self.s1], axis=axis)

        assert "Axis calibration mismatch detected along axis 0" not in caplog.text

    @pytest.mark.parametrize("axis", [0, None])
    def test_has_warning(self, axis, caplog):
        with caplog.at_level(logging.WARNING):
            _ = stack([self.s1, self.s2], axis=axis)

        assert "Axis calibration mismatch detected along axis 0" in caplog.text


class TestMismatchedCalibration2D:
    def setup_method(self, method):
        self.s1 = signals.Signal2D(np.ones((2, 3, 4, 5)))
        self.s2 = signals.Signal2D(np.ones((2, 3, 4, 5)))

        self.s1.axes_manager[1].scale = 1.0
        self.s1.axes_manager[1].offset = 0.0
        self.s1.axes_manager[1].units = ""

        self.s2.axes_manager[1].scale = 2.0
        self.s2.axes_manager[1].offset = 0.00001
        self.s2.axes_manager[1].units = "mm"

    @pytest.mark.parametrize("axis", [0, None])
    def test_no_warning(self, axis, caplog):
        with caplog.at_level(logging.WARNING):
            _ = stack([self.s1, self.s1], axis=axis)

        assert "Axis calibration mismatch detected along axis 1" not in caplog.text

    @pytest.mark.parametrize("axis", [0, None])
    def test_has_warning(self, axis, caplog):
        with caplog.at_level(logging.WARNING):
            _ = stack([self.s1, self.s2], axis=axis)

        assert "Axis calibration mismatch detected along axis 1" in caplog.text
