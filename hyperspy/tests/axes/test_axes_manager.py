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

from unittest import mock

import numpy as np
import pytest

from hyperspy.axes import (
    AxesManager,
    BaseDataAxis,
    GeneratorLen,
    _flyback_iter,
    _serpentine_iter,
)
from hyperspy.defaults_parser import preferences
from hyperspy.signals import BaseSignal, Signal1D, Signal2D


def generator():
    for i in range(3):
        yield ((0, 0, i))


class TestAxesManager:
    def setup_method(self, method):
        axes_list = [
            {
                "name": "a",
                "navigate": True,
                "offset": 0.0,
                "scale": 1.3,
                "size": 2,
                "units": "aa",
            },
            {
                "name": "b",
                "navigate": False,
                "offset": 1.0,
                "scale": 6.0,
                "size": 3,
                "units": "bb",
            },
            {
                "name": "c",
                "navigate": False,
                "offset": 2.0,
                "scale": 100.0,
                "size": 4,
                "units": "cc",
            },
            {
                "name": "d",
                "navigate": True,
                "offset": 3.0,
                "scale": 1000000.0,
                "size": 5,
                "units": "dd",
            },
        ]

        self.am = AxesManager(axes_list)

    def test_reprs(self):
        repr(self.am)
        self.am._repr_html_
        self.am[0].convert_to_non_uniform_axis()
        self.am[-1].convert_to_non_uniform_axis()
        repr(self.am)
        self.am._repr_html_

    def test_update_from(self):
        am = self.am
        am2 = self.am.deepcopy()
        m = mock.Mock()
        am.events.any_axis_changed.connect(m.changed)
        am.update_axes_attributes_from(am2._axes)
        assert not m.changed.called
        am2[0].scale = 0.5
        am2[1].units = "km"
        am2[2].offset = 50
        am2[3].size = 1
        am.update_axes_attributes_from(am2._axes, attributes=["units", "scale"])
        assert m.changed.called
        assert am2[0].scale == am[0].scale
        assert am2[1].units == am[1].units
        assert am2[2].offset != am[2].offset
        assert am2[3].size != am[3].size

    def test_create_axis_from_object(self):
        am = self.am
        axis = am[-1].copy()
        am.create_axes([axis])
        assert am[-3].offset == am[-1].offset
        assert am[-3].scale == am[-1].scale

    def test_set_axis(self):
        am = self.am
        axis = am[-1].copy()
        am.set_axis(axis, 2)
        assert am[-2].offset == am[-1].offset
        assert am[-2].scale == am[-1].scale

    def test_all_uniform(self):
        assert self.am.all_uniform is True
        self.am[-1].convert_to_non_uniform_axis()
        assert self.am.all_uniform is False

    def test_get_axis(self):
        am = self.am
        assert am[0] == am["d"]
        assert am[-1] == am["b"]
        axis = am[1]
        assert am[axis] == axis
        with pytest.raises(ValueError):
            axis = BaseDataAxis()
            am[axis]


class TestAxesManagerScaleOffset:
    def test_low_high_value(self):
        data = np.arange(11)
        s = BaseSignal(data)
        axes = s.axes_manager[0]
        assert axes.low_value == data[0]
        assert axes.high_value == data[-1]

    def test_change_scale(self):
        data = np.arange(132)
        s = BaseSignal(data)
        axes = s.axes_manager[0]
        scale_value_list = [0.07, 76, 1]
        for scale_value in scale_value_list:
            axes.scale = scale_value
            assert axes.low_value == data[0] * scale_value
            assert axes.high_value == data[-1] * scale_value

    def test_change_offset(self):
        data = np.arange(81)
        s = BaseSignal(data)
        axes = s.axes_manager[0]
        offset_value_list = [12, -216, 1, 0]
        for offset_value in offset_value_list:
            axes.offset = offset_value
            assert axes.low_value == (data[0] + offset_value)
            assert axes.high_value == (data[-1] + offset_value)

    def test_change_offset_scale(self):
        data = np.arange(11)
        s = BaseSignal(data)
        axes = s.axes_manager[0]
        scale, offset = 0.123, -314
        axes.offset = offset
        axes.scale = scale
        assert axes.low_value == (data[0] * scale + offset)
        assert axes.high_value == (data[-1] * scale + offset)


class TestAxesManagerExtent:
    def test_1d_basesignal(self):
        s = BaseSignal(np.arange(10))
        assert len(s.axes_manager.signal_extent) == 2
        signal_axis = s.axes_manager.signal_axes[0]
        signal_extent = (signal_axis.low_value, signal_axis.high_value)
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 0
        assert () == s.axes_manager.navigation_extent

    def test_1d_signal1d(self):
        s = Signal1D(np.arange(10))
        assert len(s.axes_manager.signal_extent) == 2
        signal_axis = s.axes_manager.signal_axes[0]
        signal_extent = (signal_axis.low_value, signal_axis.high_value)
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 0
        assert () == s.axes_manager.navigation_extent

    def test_2d_signal1d(self):
        s = Signal1D(np.arange(100).reshape(10, 10))
        assert len(s.axes_manager.signal_extent) == 2
        signal_axis = s.axes_manager.signal_axes[0]
        signal_extent = (signal_axis.low_value, signal_axis.high_value)
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 2
        nav_axis = s.axes_manager.navigation_axes[0]
        nav_extent = (nav_axis.low_value, nav_axis.high_value)
        assert nav_extent == s.axes_manager.navigation_extent

    def test_3d_signal1d(self):
        s = Signal1D(np.arange(1000).reshape(10, 10, 10))
        assert len(s.axes_manager.signal_extent) == 2
        signal_axis = s.axes_manager.signal_axes[0]
        signal_extent = (signal_axis.low_value, signal_axis.high_value)
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 4
        nav_axis0 = s.axes_manager.navigation_axes[0]
        nav_axis1 = s.axes_manager.navigation_axes[1]
        nav_extent = (
            nav_axis0.low_value,
            nav_axis0.high_value,
            nav_axis1.low_value,
            nav_axis1.high_value,
        )
        assert nav_extent == s.axes_manager.navigation_extent

    def test_2d_signal2d(self):
        s = Signal2D(np.arange(100).reshape(10, 10))
        assert len(s.axes_manager.signal_extent) == 4
        signal_axis0 = s.axes_manager.signal_axes[0]
        signal_axis1 = s.axes_manager.signal_axes[1]
        signal_extent = (
            signal_axis0.low_value,
            signal_axis0.high_value,
            signal_axis1.low_value,
            signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 0
        assert () == s.axes_manager.navigation_extent

    def test_3d_signal2d(self):
        s = Signal2D(np.arange(1000).reshape(10, 10, 10))
        assert len(s.axes_manager.signal_extent) == 4
        signal_axis0 = s.axes_manager.signal_axes[0]
        signal_axis1 = s.axes_manager.signal_axes[1]
        signal_extent = (
            signal_axis0.low_value,
            signal_axis0.high_value,
            signal_axis1.low_value,
            signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 2
        nav_axis = s.axes_manager.navigation_axes[0]
        nav_extent = (nav_axis.low_value, nav_axis.high_value)
        assert nav_extent == s.axes_manager.navigation_extent

    def test_changing_scale_offset(self):
        s = Signal2D(np.arange(100).reshape(10, 10))
        signal_axis0 = s.axes_manager.signal_axes[0]
        signal_axis1 = s.axes_manager.signal_axes[1]
        signal_extent = (
            signal_axis0.low_value,
            signal_axis0.high_value,
            signal_axis1.low_value,
            signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent
        signal_axis0.scale = 0.2
        signal_axis1.scale = 0.7
        signal_extent = (
            signal_axis0.low_value,
            signal_axis0.high_value,
            signal_axis1.low_value,
            signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent
        signal_axis0.offset = -11
        signal_axis1.scale = 23
        signal_extent = (
            signal_axis0.low_value,
            signal_axis0.high_value,
            signal_axis1.low_value,
            signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent


def test_setting_indices_coordinates():
    s = Signal1D(np.arange(1000).reshape(10, 10, 10))

    m = mock.Mock()
    s.axes_manager.events.indices_changed.connect(m, [])

    # both indices are changed but the event is triggered only once
    s.axes_manager.indices = (5, 5)
    assert s.axes_manager.indices == (5, 5)
    assert m.call_count == 1

    # indices not changed, so the event is not triggered
    s.axes_manager.indices == (5, 5)
    assert s.axes_manager.indices == (5, 5)
    assert m.call_count == 1

    # both indices changed again, call only once
    s.axes_manager.indices = (2, 3)
    assert s.axes_manager.indices == (2, 3)
    assert m.call_count == 2

    # single index changed, call only once
    s.axes_manager.indices = (2, 2)
    assert s.axes_manager.indices == (2, 2)
    assert m.call_count == 3

    # both coordinates are changed but the event is triggered only once
    s.axes_manager.coordinates = (5, 5)
    assert s.axes_manager.coordinates == (5, 5)
    assert m.call_count == 4

    # coordinates not changed, so the event is not triggered
    s.axes_manager.indices == (5, 5)
    assert s.axes_manager.indices == (5, 5)
    assert m.call_count == 4

    # both coordinates changed again, call only once
    s.axes_manager.coordinates = (2, 3)
    assert s.axes_manager.coordinates == (2, 3)
    assert m.call_count == 5

    # single coordinate changed, call only once
    s.axes_manager.indices = (2, 2)
    assert s.axes_manager.indices == (2, 2)
    assert m.call_count == 6


class TestAxesHotkeys:
    def setup_method(self, method):
        s = Signal1D(np.zeros(7 * (5,)))
        self.am = s.axes_manager

    def test_hotkeys_in_six_dimensions(self):
        "Step twice increasing and once decreasing all axes"

        mod01 = preferences.Plot.modifier_dims_01
        mod23 = preferences.Plot.modifier_dims_23
        mod45 = preferences.Plot.modifier_dims_45

        dim0_decrease = mod01 + "+" + preferences.Plot.dims_024_decrease
        dim0_increase = mod01 + "+" + preferences.Plot.dims_024_increase
        dim1_decrease = mod01 + "+" + preferences.Plot.dims_135_decrease
        dim1_increase = mod01 + "+" + preferences.Plot.dims_135_increase
        dim2_decrease = mod23 + "+" + preferences.Plot.dims_024_decrease
        dim2_increase = mod23 + "+" + preferences.Plot.dims_024_increase
        dim3_decrease = mod23 + "+" + preferences.Plot.dims_135_decrease
        dim3_increase = mod23 + "+" + preferences.Plot.dims_135_increase
        dim4_decrease = mod45 + "+" + preferences.Plot.dims_024_decrease
        dim4_increase = mod45 + "+" + preferences.Plot.dims_024_increase
        dim5_decrease = mod45 + "+" + preferences.Plot.dims_135_decrease
        dim5_increase = mod45 + "+" + preferences.Plot.dims_135_increase

        steps = [
            dim0_increase,
            dim0_increase,
            dim0_decrease,
            dim1_increase,
            dim1_increase,
            dim1_decrease,
            dim2_increase,
            dim2_increase,
            dim2_decrease,
            dim3_increase,
            dim3_increase,
            dim3_decrease,
            dim4_increase,
            dim4_increase,
            dim4_decrease,
            dim5_increase,
            dim5_increase,
            dim5_decrease,
        ]

        class fake_key_event:
            "Fake event handler for plot key press"

            def __init__(self, key):
                self.key = key

        for step in steps:
            self.am.key_navigator(fake_key_event(step))

        assert self.am.indices == (1, 1, 1, 1, 1, 1)


class TestIterPathScanPattern:
    def setup_method(self, method):
        s = Signal1D(np.zeros((3, 3, 3, 2)))
        self.am = s.axes_manager

    def test_iterpath_property(self):
        self.am._iterpath = "abc"  # with underscore
        assert self.am.iterpath == "abc"

        with pytest.raises(ValueError):
            self.am.iterpath = "blahblah"  # w/o underscore

        path = "flyback"
        self.am.iterpath = path
        assert self.am.iterpath == path
        assert self.am._iterpath == path

        path = "serpentine"
        self.am.iterpath = path
        assert self.am.iterpath == path
        assert self.am._iterpath == path

    def test_wrong_iterpath(self):
        with pytest.raises(ValueError):
            self.am.iterpath = ""

    def test_wrong_custom_iterpath(self):
        class A:
            pass

        with pytest.raises(TypeError):
            self.am.iterpath = A()

    def test_wrong_custom_iterpath2(self):
        with pytest.raises(TypeError):
            self.am.iterpath = [
                0,
                1,
                2,
                3,
                4,
            ]  # indices are not iterable

    def test_wrong_custom_iterpath3(self):
        with pytest.raises(ValueError):
            self.am.iterpath = [(0,)]  # not enough dimensions

    def test_flyback(self):
        self.am.iterpath = "flyback"
        for i, _ in enumerate(self.am):
            if i == 3:
                assert self.am.indices == (0, 1, 0)
            # Hits a new layer on index 9
            if i == 9:
                assert self.am.indices == (0, 0, 1)
            break

    def test_serpentine(self):
        self.am.iterpath = "serpentine"
        for i, _ in enumerate(self.am):
            if i == 3:
                assert self.am.indices == (2, 1, 0)
            # Hits a new layer on index 9
            if i == 9:
                assert self.am.indices == (2, 2, 1)
            break

    def test_custom_iterpath(self):
        iterpath = [(0, 1, 1), (1, 1, 1)]
        self.am.iterpath = iterpath
        assert self.am._iterpath == iterpath
        assert self.am.iterpath == iterpath
        assert self.am._iterpath_generator != iterpath

        for i, _ in enumerate(self.am):
            if i == 0:
                assert self.am.indices == iterpath[0]
            if i == 1:
                assert self.am.indices == iterpath[1]
            break

    def test_custom_iterpath_generator(self):
        iterpath = generator()
        self.am.iterpath = iterpath
        assert self.am._iterpath == iterpath
        assert self.am.iterpath == iterpath
        assert self.am._iterpath_generator == iterpath

        for i, _ in enumerate(self.am):
            if i == 0:
                assert self.am.indices == (0, 0, 0)
            if i == 1:
                assert self.am.indices == (0, 0, 1)
            break

    def test_get_iterpath_size1(self):
        assert self.am._get_iterpath_size() == self.am.navigation_size
        assert (
            self.am._get_iterpath_size(masked_elements=1) == self.am.navigation_size - 1
        )

    def test_get_iterpath_size2(self):
        self.am.iterpath = generator()
        assert self.am._get_iterpath_size() is None
        assert self.am._get_iterpath_size(masked_elements=1) is None

    def test_get_iterpath_size3(self):
        self.am.iterpath = [(0, 0, 0), (0, 0, 1)]
        assert self.am._get_iterpath_size() == 2
        assert self.am._get_iterpath_size(masked_elements=1) == 2

    def test_GeneratorLen(self):
        gen = GeneratorLen(gen=generator(), length=3)
        assert list(gen) == [(0, 0, 0), (0, 0, 1), (0, 0, 2)]

    def test_GeneratorLen_iterpath(self):
        gen = GeneratorLen(gen=generator(), length=3)
        assert len(gen) == 3
        self.am.iterpath = gen
        assert self.am._get_iterpath_size() == 3


class TestIterPathScanPatternSignal2D:
    def setup_method(self, method):
        s = Signal2D(np.zeros((3, 3, 3, 2, 1)))
        self.am = s.axes_manager
        self.s = s

    def test_wrong_iterpath_signal2D(self):
        with pytest.raises(ValueError):
            self.am.iterpath = "blahblah"

    def test_custom_iterpath_signal2D(self):
        indices = [(0, 1, 1), (1, 1, 1)]
        self.am.iterpath = indices
        for i, _ in enumerate(self.am):
            if i == 0:
                assert self.am.indices == indices[0]
            if i == 1:
                assert self.am.indices == indices[1]
            break

    def test_serpentine_signal2D(self):
        self.am.iterpath = "serpentine"
        for i, _ in enumerate(self.am):
            if i == 3:
                assert self.am.indices == (2, 1, 0)
            # Hits a new layer on index 9
            if i == 9:
                assert self.am.indices == (2, 2, 1)
            break

    def test_switch_iterpath(self):
        s = self.s
        s.axes_manager.iterpath = "serpentine"
        with s.axes_manager.switch_iterpath("flyback"):
            assert s.axes_manager.iterpath == "flyback"
            for i, _ in enumerate(s.axes_manager):
                if i == 3:
                    assert self.am.indices == (0, 1, 0)
                # Hits a new layer on index 9
                if i == 9:
                    assert self.am.indices == (0, 0, 1)
                break
        assert s.axes_manager.iterpath == "serpentine"


def test_iterpath_function_flyback():
    for i, indices in enumerate(_flyback_iter((3, 3, 3))):
        if i == 3:
            assert indices == (0, 1, 0)


def test_iterpath_function_serpentine():
    for i, indices in enumerate(_serpentine_iter((3, 3, 3))):
        if i == 3:
            assert indices == (2, 1, 0)


def TestAxesManagerRagged():
    def setup_method(self, method):
        axes_list = [
            {
                "name": "a",
                "navigate": True,
                "offset": 0.0,
                "scale": 1.3,
                "size": 2,
                "units": "aa",
            },
        ]

        self.am = AxesManager(axes_list)
        self.am._ragged = True

    def test_ragged_property(self):
        assert self.am.ragged
        with pytest.raises(AttributeError):
            self.am.ragged = False
        self.am._ragged = False
        assert not self.am.ragged

    def test_reprs(self):
        expected_string = "<Axes manager, axes: (2|ragged)>\n"
        "            Name |   size |  index |  offset |   scale |  units \n"
        "================ | ====== | ====== | ======= | ======= | ====== \n"
        "               a |      2 |      0 |       0 |     1.3 |     aa \n"
        "---------------- | ------ | ------ | ------- | ------- | ------ \n"
        "     Ragged axis |               Variable length"
        assert self.am.__repr__() == expected_string
