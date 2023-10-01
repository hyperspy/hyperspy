# -*- coding: utf-8 -*-
# Copyright 2007-2023 The exspy developers
#
# This file is part of exspy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# exspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with exspy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np
import pytest

import hyperspy.api as hs

from exspy.components import DoublePowerLaw


def test_function():
    g = DoublePowerLaw()
    g.A.value = 3
    g.r.value = 2
    g.origin.value = 1
    g.shift.value = 2
    g.ratio.value = 2
    assert np.isinf(g.function(1))
    assert np.isinf(g.function(3))
    assert g.function(-1) == 0
    assert g.function(0) == 0
    assert g.function(2) == 9
    np.testing.assert_allclose(g.function(10), 0.15948602)
    assert g.grad_A(2) == 3
    np.testing.assert_allclose(g.grad_r(4), -0.3662041)
    assert g.grad_origin(2)  == -6
    assert g.grad_shift(2)  == -12
    assert g.grad_ratio(2)  == 3


class TestDoublePowerLaw:

    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(1024))
        s.axes_manager[0].offset = 100
        s.axes_manager[0].scale = 0.1
        m = s.create_model()
        exspy = pytest.importorskip("exspy")
        from exspy.components import DoublePowerLaw
        m.append(DoublePowerLaw())
        m[0].A.value = 1000
        m[0].r.value = 4
        m[0].ratio.value = 200
        self.m = m

    @pytest.mark.parametrize(("binned"), (True, False))
    def test_fit(self, binned):
        self.m.signal.axes_manager[-1].is_binned = binned
        s = self.m.as_signal()
        assert s.axes_manager[-1].is_binned == binned
        exspy = pytest.importorskip("exspy")
        g = exspy.components.DoublePowerLaw()
        # Fix the ratio parameter to test the fit
        g.ratio.free = False
        g.shift.free = False
        g.origin.free = False
        g.ratio.value = 200
        m = s.create_model()
        m.append(g)
        m.fit_component(g, signal_range=(None, None))
        np.testing.assert_allclose(g.A.value, 1000.0)
        np.testing.assert_allclose(g.r.value, 4.0)
        np.testing.assert_allclose(g.ratio.value, 200.)