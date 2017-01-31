
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
