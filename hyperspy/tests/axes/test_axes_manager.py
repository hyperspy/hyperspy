
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

from hyperspy.axes import DataAxis, AxesManager
from hyperspy.signals import BaseSignal, Signal1D, Signal2D
from numpy import arange


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
