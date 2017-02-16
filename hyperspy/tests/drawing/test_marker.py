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
from hyperspy.io import load

import tempfile

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

    def test_get_data_shape_point(self):
        m0 = markers.point(5, 5)
        m1 = markers.point((5, 10), (5, 10))
        m2 = markers.point(((12, 2, 9), (1, 2, 3)), ((2, 5, 1), (3, 9, 2)))
        m3 = markers.vertical_line(((12, 2), (2, 5), (9, 2)))
        assert m0._get_data_shape() == ()
        assert m1._get_data_shape() == (2,)
        assert m2._get_data_shape() == (2, 3)
        assert m3._get_data_shape() == (3, 2)

    def test_add_marker_signal1d_navigation_dim(self):
        s = Signal1D(np.zeros((3, 50, 50)))
        m0 = markers.point(5, 5)
        m1 = markers.point((5, 10), (10, 15))
        m2 = markers.point(np.zeros((50, 3)), np.zeros((50, 3)))
        s.add_marker(m0)
        with pytest.raises(ValueError):
            s.add_marker(m1)
        s.add_marker(m2)

    def test_add_marker_signal2d_navigation_dim(self):
        s = Signal2D(np.zeros((3, 50, 50)))
        m0 = markers.point(5, 5)
        m1 = markers.point((5, 10), (10, 15))
        m2 = markers.point(np.zeros((3, )), np.zeros((3, )))
        s.add_marker(m0)
        with pytest.raises(ValueError):
            s.add_marker(m1)
        s.add_marker(m2)


class Test_permanent_markers:

    def test_add_permanent_marker(self):
        s = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        assert list(s.metadata.Markers)[0][1] == m

    def test_remove_permanent_marker_name(self):
        s = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        m.name = 'test'
        s.add_marker(m, permanent=True)
        assert list(s.metadata.Markers)[0][1] == m
        del s.metadata.Markers.test
        assert len(list(s.metadata.Markers)) == 0

    def test_permanent_marker_names(self):
        s = Signal1D(np.arange(10))
        m0 = markers.point(x=5, y=5)
        m1 = markers.point(x=5, y=5)
        m0.name = 'test'
        m1.name = 'test'
        s.add_marker(m0, permanent=True)
        s.add_marker(m1, permanent=True)
        assert s.metadata.Markers.test == m0
        assert m0.name == 'test'
        assert s.metadata.Markers.test1 == m1
        assert m1.name == 'test1'
    
    def test_add_permanent_marker_twice(self):
        s = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        with pytest.raises(ValueError):
            s.add_marker(m, permanent=True)

    def test_add_permanent_marker_twice_different_signal(self):
        s0 = Signal1D(np.arange(10))
        s1 = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        s0.add_marker(m, permanent=True)
        with pytest.raises(ValueError):
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
        assert len(list(s.metadata.Markers)) == 7
        with pytest.raises(ValueError):
            s.add_marker(m_rect, permanent=True)

    def test_add_permanent_marker_signal2d(self):
        s = Signal2D(np.arange(100).reshape(10,10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        assert list(s.metadata.Markers)[0][1] == m

    def test_save_permanent_marker(self):
        s = Signal2D(np.arange(100).reshape(10,10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        with tempfile.TemporaryDirectory() as tmp:
            filename = tmp + '/testsavefile.hdf5'
        s.save(filename)

    def test_load_permanent_marker(self):
        x, y = 5, 2
        color = 'red'
        size = 10
        name = 'testname'
        s = Signal2D(np.arange(100).reshape(10,10))
        m = markers.point(x=x, y=y, color=color, size=size)
        m.name = name
        s.add_marker(m, permanent=True)
        with tempfile.TemporaryDirectory() as tmp:
            filename = tmp + '/testloadfile.hdf5'
        s.save(filename)
        s1 = load(filename)
        assert s1.metadata.Markers.has_item(name)
        m1 = s1.metadata.Markers.get_item(name)
        assert m1.get_data_position('x1') == x
        assert m1.get_data_position('y1') == y
        assert m1.get_data_position('size') == size
        assert m1.marker_properties['color'] == color
        assert m1.name == name

    def test_load_permanent_marker_all_types(self):
        x1, y1, x2, y2 = 5, 2, 1, 8
        s = Signal2D(np.arange(100).reshape(10,10))
        m0 = markers.point(x=x1, y=y1)
        m1 = markers.horizontal_line(y=y1)
        m2 = markers.horizontal_line_segment(x1=x1, x2=x2, y=y1)
        m3 = markers.line_segment(x1=x1, x2=x2, y1=y1, y2=y2)
        m4 = markers.rectangle(x1=x1, x2=x2, y1=y1, y2=y2)
        m5 = markers.text(x=x1, y=y1, text="test")
        m6 = markers.vertical_line(x=x1)
        m7 = markers.vertical_line_segment(x=x1, y1=y1, y2=y2)
        s.add_marker(m0, permanent=True)
        s.add_marker(m1, permanent=True)
        s.add_marker(m2, permanent=True)
        s.add_marker(m3, permanent=True)
        s.add_marker(m4, permanent=True)
        s.add_marker(m5, permanent=True)
        s.add_marker(m6, permanent=True)
        s.add_marker(m7, permanent=True)
        with tempfile.TemporaryDirectory() as tmp:
            filename = tmp + '/testallmarkersfile.hdf5'
        s.save(filename)
        s1 = load(filename)
        assert len(list(s1.metadata.Markers)) == 8
