
# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

from unittest import mock

from hyperspy.axes import AxesManager
from hyperspy.signals import BaseSignal, Signal1D, Signal2D
from hyperspy.defaults_parser import preferences
from numpy import arange, zeros


class TestAxesManager:

    def setup_method(self, method):
        axes_list = [
            {'name': 'a',
             'navigate': True,
             'offset': 0.0,
             'scale': 1.3,
             'size': 2,
             'units': 'aa'},
            {'name': 'b',
             'navigate': False,
             'offset': 1.0,
             'scale': 6.0,
             'size': 3,
             'units': 'bb'},
            {'name': 'c',
             'navigate': False,
             'offset': 2.0,
             'scale': 100.0,
             'size': 4,
             'units': 'cc'},
            {'name': 'd',
             'navigate': True,
             'offset': 3.0,
             'scale': 1000000.0,
             'size': 5,
             'units': 'dd'}]

        self.am = AxesManager(axes_list)

    def test_reprs(self):
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
        am.update_axes_attributes_from(am2._axes,
                                       attributes=["units", "scale"])
        assert m.changed.called
        assert am2[0].scale == am[0].scale
        assert am2[1].units == am[1].units
        assert am2[2].offset != am[2].offset
        assert am2[3].size != am[3].size


class TestAxesManagerScaleOffset:

    def test_low_high_value(self):
        data = arange(11)
        s = BaseSignal(data)
        axes = s.axes_manager[0]
        assert axes.low_value == data[0]
        assert axes.high_value == data[-1]

    def test_change_scale(self):
        data = arange(132)
        s = BaseSignal(data)
        axes = s.axes_manager[0]
        scale_value_list = [0.07, 76, 1]
        for scale_value in scale_value_list:
            axes.scale = scale_value
            assert axes.low_value == data[0] * scale_value
            assert axes.high_value == data[-1] * scale_value

    def test_change_offset(self):
        data = arange(81)
        s = BaseSignal(data)
        axes = s.axes_manager[0]
        offset_value_list = [12, -216, 1, 0]
        for offset_value in offset_value_list:
            axes.offset = offset_value
            assert axes.low_value == (data[0] + offset_value)
            assert axes.high_value == (data[-1] + offset_value)

    def test_change_offset_scale(self):
        data = arange(11)
        s = BaseSignal(data)
        axes = s.axes_manager[0]
        scale, offset = 0.123, -314
        axes.offset = offset
        axes.scale = scale
        assert axes.low_value == (data[0] * scale + offset)
        assert axes.high_value == (data[-1] * scale + offset)


class TestAxesManagerExtent:

    def test_1d_basesignal(self):
        s = BaseSignal(arange(10))
        assert len(s.axes_manager.signal_extent) == 2
        signal_axis = s.axes_manager.signal_axes[0]
        signal_extent = (signal_axis.low_value, signal_axis.high_value)
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 0
        assert () == s.axes_manager.navigation_extent

    def test_1d_signal1d(self):
        s = Signal1D(arange(10))
        assert len(s.axes_manager.signal_extent) == 2
        signal_axis = s.axes_manager.signal_axes[0]
        signal_extent = (signal_axis.low_value, signal_axis.high_value)
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 0
        assert () == s.axes_manager.navigation_extent

    def test_2d_signal1d(self):
        s = Signal1D(arange(100).reshape(10, 10))
        assert len(s.axes_manager.signal_extent) == 2
        signal_axis = s.axes_manager.signal_axes[0]
        signal_extent = (signal_axis.low_value, signal_axis.high_value)
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 2
        nav_axis = s.axes_manager.navigation_axes[0]
        nav_extent = (nav_axis.low_value, nav_axis.high_value)
        assert nav_extent == s.axes_manager.navigation_extent

    def test_3d_signal1d(self):
        s = Signal1D(arange(1000).reshape(10, 10, 10))
        assert len(s.axes_manager.signal_extent) == 2
        signal_axis = s.axes_manager.signal_axes[0]
        signal_extent = (signal_axis.low_value, signal_axis.high_value)
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 4
        nav_axis0 = s.axes_manager.navigation_axes[0]
        nav_axis1 = s.axes_manager.navigation_axes[1]
        nav_extent = (
            nav_axis0.low_value, nav_axis0.high_value,
            nav_axis1.low_value, nav_axis1.high_value,
        )
        assert nav_extent == s.axes_manager.navigation_extent

    def test_2d_signal2d(self):
        s = Signal2D(arange(100).reshape(10, 10))
        assert len(s.axes_manager.signal_extent) == 4
        signal_axis0 = s.axes_manager.signal_axes[0]
        signal_axis1 = s.axes_manager.signal_axes[1]
        signal_extent = (
            signal_axis0.low_value, signal_axis0.high_value,
            signal_axis1.low_value, signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 0
        assert () == s.axes_manager.navigation_extent

    def test_3d_signal2d(self):
        s = Signal2D(arange(1000).reshape(10, 10, 10))
        assert len(s.axes_manager.signal_extent) == 4
        signal_axis0 = s.axes_manager.signal_axes[0]
        signal_axis1 = s.axes_manager.signal_axes[1]
        signal_extent = (
            signal_axis0.low_value, signal_axis0.high_value,
            signal_axis1.low_value, signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent
        assert len(s.axes_manager.navigation_extent) == 2
        nav_axis = s.axes_manager.navigation_axes[0]
        nav_extent = (nav_axis.low_value, nav_axis.high_value)
        assert nav_extent == s.axes_manager.navigation_extent

    def test_changing_scale_offset(self):
        s = Signal2D(arange(100).reshape(10, 10))
        signal_axis0 = s.axes_manager.signal_axes[0]
        signal_axis1 = s.axes_manager.signal_axes[1]
        signal_extent = (
            signal_axis0.low_value, signal_axis0.high_value,
            signal_axis1.low_value, signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent
        signal_axis0.scale = 0.2
        signal_axis1.scale = 0.7
        signal_extent = (
            signal_axis0.low_value, signal_axis0.high_value,
            signal_axis1.low_value, signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent
        signal_axis0.offset = -11
        signal_axis1.scale = 23
        signal_extent = (
            signal_axis0.low_value, signal_axis0.high_value,
            signal_axis1.low_value, signal_axis1.high_value,
        )
        assert signal_extent == s.axes_manager.signal_extent


def test_setting_indices_coordinates():
    s = Signal1D(arange(1000).reshape(10, 10, 10))

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
        s = Signal1D(zeros(7 * (5,)))
        self.am = s.axes_manager

    def test_hotkeys_in_six_dimensions(self):
        'Step twice increasing and once decreasing all axes'

        mod01 = preferences.Plot.modifier_dims_01
        mod23 = preferences.Plot.modifier_dims_23
        mod45 = preferences.Plot.modifier_dims_45

        dim0_decrease = mod01 + '+' + preferences.Plot.dims_024_decrease
        dim0_increase = mod01 + '+' + preferences.Plot.dims_024_increase
        dim1_decrease = mod01 + '+' + preferences.Plot.dims_135_decrease
        dim1_increase = mod01 + '+' + preferences.Plot.dims_135_increase
        dim2_decrease = mod23 + '+' + preferences.Plot.dims_024_decrease
        dim2_increase = mod23 + '+' + preferences.Plot.dims_024_increase
        dim3_decrease = mod23 + '+' + preferences.Plot.dims_135_decrease
        dim3_increase = mod23 + '+' + preferences.Plot.dims_135_increase
        dim4_decrease = mod45 + '+' + preferences.Plot.dims_024_decrease
        dim4_increase = mod45 + '+' + preferences.Plot.dims_024_increase
        dim5_decrease = mod45 + '+' + preferences.Plot.dims_135_decrease
        dim5_increase = mod45 + '+' + preferences.Plot.dims_135_increase

        steps = [dim0_increase, dim0_increase, dim0_decrease, dim1_increase,
                 dim1_increase, dim1_decrease, dim2_increase, dim2_increase, dim2_decrease,
                 dim3_increase, dim3_increase, dim3_decrease, dim4_increase,
                 dim4_increase, dim4_decrease, dim5_increase, dim5_increase, dim5_decrease]

        class fake_key_event():
            'Fake event handler for plot key press'

            def __init__(self, key):
                self.key = key

        for step in steps:
            self.am.key_navigator(fake_key_event(step))

        assert self.am.indices == (1, 1, 1, 1, 1, 1)
