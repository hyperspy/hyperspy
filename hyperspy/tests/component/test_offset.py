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

import itertools

import numpy as np
import pytest

import hyperspy.api as hs

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]


class TestOffset:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.zeros(10))
        s.axes_manager[0].scale = 0.01
        m = s.create_model()
        m.append(hs.model.components1D.Offset())
        m[0].offset.value = 10
        self.m = m

    @pytest.mark.parametrize(("uniform"), (True, False))
    @pytest.mark.parametrize(("only_current", "binned"), TRUE_FALSE_2_TUPLE)
    def test_estimate_parameters(self, only_current, binned, uniform):
        self.m.signal.axes_manager[-1].is_binned = binned
        self.m.assign_current_values_to_all()
        s = self.m.as_signal()
        if not uniform:
            s.axes_manager[-1].convert_to_non_uniform_axis()
        assert s.axes_manager[-1].is_binned == binned
        o = hs.model.components1D.Offset()
        o.estimate_parameters(s, None, None, only_current=only_current)
        assert o._axes_manager[-1].is_binned == binned
        assert o._axes_manager[-1].is_uniform == uniform
        np.testing.assert_allclose(o.offset.value, 10)

    @pytest.mark.parametrize(("uniform"), (True, False))
    @pytest.mark.parametrize(("binned"), (True, False))
    def test_function_nd(self, binned, uniform):
        self.m.signal.axes_manager[-1].is_binned = binned
        self.m.assign_current_values_to_all()
        s = self.m.as_signal()
        s = hs.stack([s] * 2)
        o = hs.model.components1D.Offset()
        o.estimate_parameters(s, None, None, only_current=False)
        assert o._axes_manager[-1].is_binned == binned
        axis = s.axes_manager.signal_axes[0]
        factor = axis.scale if binned else 1
        np.testing.assert_allclose(o.function_nd(axis.axis) * factor, s.data)

    def test_constant_term(self):
        m = self.m
        o = m[0]
        o.offset.free = True
        assert o._constant_term == 0

        o.offset.free = False
        assert o._constant_term == o.offset.value
