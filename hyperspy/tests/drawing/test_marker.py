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

import pytest
import numpy as np

from hyperspy.signals import Signal1D, Signal2D
from hyperspy.utils import markers


class Test_markers:

    def test_get_data(self):
        s = Signal2D(np.zeros([3, 2, 2]))
        m = markers.line_segment(x1=list(range(3)),
                                 x2=list(range(3)),
                                 y1=1.3,
                                 y2=1.5)
        m.axes_manager = s.axes_manager
        assert m.get_data_position('x1') == 0
        assert m.get_data_position('y1') == 1.3
        s.axes_manager[0].index = 2
        assert m.get_data_position('x1') == 2
        assert m.get_data_position('y1') == 1.3

    def test_iterate_strings(self):
        s = Signal2D(np.zeros([3, 2, 2]))
        m = markers.text(x=list(range(3)),
                         y=list(range(3)),
                         text=['one', 'two', 'three'])
        m.axes_manager = s.axes_manager
        assert m.get_data_position('text') == 'one'
        s.axes_manager[0].index = 2
        assert m.get_data_position('text') == 'three'

    def test_get_one_string(self):
        s = Signal2D(np.zeros([3, 2, 2]))
        m = markers.text(x=list(range(3)),
                         y=list(range(3)),
                         text='one')
        m.axes_manager = s.axes_manager
        assert m.get_data_position('text') == 'one'
        s.axes_manager[0].index = 2
        assert m.get_data_position('text') == 'one'

    def test_get_data_array(self):
        s = Signal2D(np.zeros([2, 2, 2, 2]))
        m = markers.line_segment(x1=[[1.1, 1.2], [1.3, 1.4]], x2=1.1, y1=1.3,
                                 y2=1.5)
        m.axes_manager = s.axes_manager
        assert m.get_data_position('x1') == 1.1
        s.axes_manager[0].index = 1
        assert m.get_data_position('x1') == 1.2
        s.axes_manager[1].index = 1
        assert m.get_data_position('x1') == 1.4

    def test_set_get_data(self):
        m = markers.point(x=0, y=1.3)
        assert m.data['x1'] == 0
        assert m.data['y1'] == 1.3
        m.add_data(y1=0.3)
        assert m.data['x1'] == 0
        assert m.data['y1'] == 0.3
        m.set_data(y1=1.3)
        assert m.data['x1'][()][()] is None
        assert m.data['y1'] == 1.3
        assert m.data['x1'].dtype == np.dtype('O')
        m.add_data(y1=[1, 2])
        assert m.data['y1'][()].shape == (2,)

    def test_markers_properties(self):
        m = markers.text(x=1, y=2, text='a')
        m.set_marker_properties(fontsize=30, color='red')
        assert (m.marker_properties ==
                {'color': 'red', 'fontsize': 30})

    def test_auto_update(self):
        m = markers.text(y=1, x=2, text='a')
        assert m.auto_update is False
        m = markers.text(y=[1, 2], x=2, text='a')
        assert m.auto_update is True
        m.add_data(y1=1)
        assert m.auto_update is False
        m.add_data(y1=[1, 2])
        assert m.auto_update is True


class Test_permanent_markers:

    def test_add_permanent_marker(self):
        s = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        assert s.markers[0] == m
    
    def test_add_permanent_marker_twice(self):
        with pytest.raises(Exception):
            s = Signal1D(np.arange(10))
            m = markers.point(x=5, y=5)
            s.add_marker(m, permanent=True)
            s.add_marker(m, permanent=True)

    def test_add_permanent_marker_twice_different_signal(self):
        with pytest.raises(Exception):
            s0 = Signal1D(np.arange(10))
            s1 = Signal1D(np.arange(10))
            m = markers.point(x=5, y=5)
            s0.add_marker(m, permanent=True)
            s1.add_marker(m, permanent=True)

    def test_add_several_permanent_markers(self):
        s = Signal1D(np.arange(10))
        m_point = markers.point(x=5, y=5)
        m_line = markers.line_segment(x1=5, x2=10, y1=5, y2=10)
        m_vline = markers.vertical_line(x=5)
        m_vline_segment = markers.vertical_line_segment(x=4, y1=3, y2=6)
        m_hline = markers.horizontal_line(y=5)
        m_hline_segment = markers.horizontal_line_segment(x1=1, x2=9, y=5)
        m_rect = markers.rectangle(x1=1, x2=3, y1=5, y2=10)
        s.add_marker(m_point, permanent=True)
        s.add_marker(m_line, permanent=True)
        s.add_marker(m_vline, permanent=True)
        s.add_marker(m_vline_segment, permanent=True)
        s.add_marker(m_hline, permanent=True)
        s.add_marker(m_hline_segment, permanent=True)
        s.add_marker(m_rect, permanent=True)
        assert len(s.markers) == 7
        with pytest.raises(Exception):
            s.add_marker(m_rect, permanent=True)

    def test_add_permanent_marker_signal2d(self):
        s = Signal2D(np.arange(100).reshape(10,10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        assert s.markers[0] == m

