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


class TestScalableFixedPattern:
    def setup_method(self, method):
        s = hs.signals.Signal1D(np.linspace(0.0, 100.0, 10))
        s1 = hs.signals.Signal1D(np.linspace(0.0, 1.0, 10))
        s.axes_manager[0].scale = 0.1
        s1.axes_manager[0].scale = 0.1
        self.s = s
        self.pattern = s1

    def test_position(self):
        s1 = self.pattern
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        assert fp._position is fp.shift

    def test_both_unbinned(self):
        s = self.s
        s1 = self.pattern
        s.axes_manager[-1].is_binned = False
        s1.axes_manager[-1].is_binned = False
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        fp.xscale.free = False
        fp.shift.free = False
        m.fit()
        np.testing.assert_allclose(fp.yscale.value, 100)

    @pytest.mark.parametrize(("uniform"), (True, False))
    def test_both_binned(self, uniform):
        s = self.s
        s1 = self.pattern
        s.axes_manager[-1].is_binned = True
        s1.axes_manager[-1].is_binned = True
        if not uniform:
            s.axes_manager[0].convert_to_non_uniform_axis()
            s1.axes_manager[0].convert_to_non_uniform_axis()
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        fp.xscale.free = False
        fp.shift.free = False
        m.fit()
        np.testing.assert_allclose(fp.yscale.value, 100)

    def test_pattern_unbinned_signal_binned(self):
        s = self.s
        s1 = self.pattern
        s.axes_manager[-1].is_binned = True
        s1.axes_manager[-1].is_binned = False
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        fp.xscale.free = False
        fp.shift.free = False
        m.fit()
        np.testing.assert_allclose(fp.yscale.value, 1000)

    def test_pattern_binned_signal_unbinned(self):
        s = self.s
        s1 = self.pattern
        s.axes_manager[-1].is_binned = False
        s1.axes_manager[-1].is_binned = True
        m = s.create_model()
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        m.append(fp)
        fp.xscale.free = False
        fp.shift.free = False
        m.fit()
        np.testing.assert_allclose(fp.yscale.value, 10)

    def test_function(self):
        s = self.s
        s1 = self.pattern
        fp = hs.model.components1D.ScalableFixedPattern(s1, interpolate=False)
        m = s.create_model()
        m.append(fp)
        m.fit(grad="analytical")
        x = s.axes_manager[0].axis
        np.testing.assert_allclose(s.data, fp.function(x))
        np.testing.assert_allclose(fp.function(x), fp.function_nd(x))

    def test_function_nd(self):
        s = self.s
        s1 = self.pattern
        fp = hs.model.components1D.ScalableFixedPattern(s1)
        s_multi = hs.stack([s] * 3)
        m = s_multi.create_model()
        m.append(fp)
        fp.yscale.map["values"] = [1.0, 0.5, 1.0]
        fp.xscale.map["values"] = [1.0, 1.0, 0.75]
        results = fp.function_nd(s.axes_manager[0].axis)
        expected = np.array([s1.data * v for v in [1, 0.5, 0.75]])
        np.testing.assert_allclose(results, expected)

    @pytest.mark.parametrize("interpolate", [True, False])
    def test_recreate_component(self, interpolate):
        s = self.s
        s1 = self.pattern
        fp = hs.model.components1D.ScalableFixedPattern(s1, interpolate=interpolate)
        assert fp.yscale._linear
        assert not fp.xscale._linear
        assert not fp.shift._linear

        m = s.create_model()
        m.append(fp)
        model_dict = m.as_dictionary()

        m2 = s.create_model()
        m2._load_dictionary(model_dict)
        assert m2[0].interpolate == interpolate
        np.testing.assert_allclose(m2[0].signal.data, s1.data)
        assert m2[0].yscale._linear
        assert not m2[0].xscale._linear
        assert not m2[0].shift._linear

    def test_estimate_parameters_yscale_ratio_only_current(self):
        scale = 3.0

        signal = hs.signals.Signal1D(self.pattern.data * scale)
        signal.axes_manager[0].scale = self.pattern.axes_manager[0].scale

        fp = hs.model.components1D.ScalableFixedPattern(self.pattern)
        fp.estimate_parameters(signal, 0.0, 0.9, only_current=True, yscale=True)

        assert fp.xscale.value == 1.0
        assert fp.shift.value == 0.0
        np.testing.assert_allclose(fp.yscale.value, scale)

    def test_estimate_parameters_yscale_ratio_map(self):
        signal = hs.signals.Signal1D(
            np.stack([self.pattern.data * 2.0, self.pattern.data * 4.0])
        )
        signal.axes_manager[-1].scale = self.pattern.axes_manager[0].scale

        fp = hs.model.components1D.ScalableFixedPattern(self.pattern)
        fp.estimate_parameters(signal, 0.0, 0.9, only_current=False, yscale=True)

        np.testing.assert_allclose(fp.xscale.map["values"], [1.0, 1.0])
        np.testing.assert_allclose(fp.shift.map["values"], [0.0, 0.0])
        np.testing.assert_allclose(fp.yscale.map["values"], [2.0, 4.0])

    @pytest.mark.parametrize("only_current", [True, False])
    @pytest.mark.parametrize(
        ("yscale", "expected_input_scale"),
        [
            (False, 5.0),
            (True, 5.0),
            (7.0, 3.0),
        ],
    )
    def test_estimate_paramters(self, only_current, yscale, expected_input_scale):
        pattern = self.pattern

        if only_current:
            signal = hs.signals.Signal1D(pattern.data * expected_input_scale)
            signal.axes_manager[0].scale = pattern.axes_manager[0].scale
        else:
            signal = hs.signals.Signal1D(
                np.stack(
                    [
                        pattern.data * expected_input_scale,
                        pattern.data * expected_input_scale,
                    ]
                )
            )
            signal.axes_manager[-1].scale = pattern.axes_manager[0].scale

        fp = hs.model.components1D.ScalableFixedPattern(pattern)
        fp.estimate_parameters(
            signal, 0.0, 0.9, only_current=only_current, yscale=yscale
        )

        if only_current:
            assert fp.xscale.value == 1.0
            assert fp.shift.value == 0.0

            if yscale is False:
                np.testing.assert_allclose(fp.yscale.value, 1.0)
            elif yscale is True:
                np.testing.assert_allclose(fp.yscale.value, expected_input_scale)
            else:
                np.testing.assert_allclose(fp.yscale.value, yscale)
        else:
            np.testing.assert_allclose(fp.xscale.map["values"], [1.0, 1.0])
            np.testing.assert_allclose(fp.shift.map["values"], [0.0, 0.0])

            if yscale is False:
                np.testing.assert_allclose(fp.yscale.map["values"], [1.0, 1.0])
            elif yscale is True:
                np.testing.assert_allclose(
                    fp.yscale.map["values"],
                    [expected_input_scale, expected_input_scale],
                )
            else:
                np.testing.assert_allclose(fp.yscale.map["values"], [yscale, yscale])

            np.testing.assert_allclose(fp.yscale.map["is_set"], [True, True])
