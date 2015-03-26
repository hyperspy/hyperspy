# Copyright 2007-2011 The HyperSpy developers
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


import numpy as np
import nose.tools
from matplotlib.pyplot import vlines

from hyperspy.drawing.marker import Marker as marker
from hyperspy.signals import Image


class Test_markers:

    def test_marker_type(self):
        m_types = ['line', 'axvline', 'axhline', 'text', 'pointer', 'rect']
        t = []
        for m_type in m_types:
            m = marker(m_type)
            t.append(m.type)
        nose.tools.assert_true(m_types == t)

    def test_get_data(self):
        s = Image(np.zeros([3, 2, 2]))
        m = marker('text')
        m.set_data(x1=range(3))
        m.add_data(y1=1.3)
        m.axes_manager = s.axes_manager
        nose.tools.assert_true(m.get_data_position('x1') == 0)
        nose.tools.assert_true(m.get_data_position('y1') == 1.3)
        s.axes_manager[0].index = 2
        nose.tools.assert_true(m.get_data_position('x1') == 2)
        nose.tools.assert_true(m.get_data_position('y1') == 1.3)

    def test_markers_properties(self):
        m = marker('text')
        m.set_marker_properties(fontsize=30, color='red')
        nose.tools.assert_true(m.marker_properties ==
                               {'color': 'red', 'fontsize': 30})

    def test_marker_lines(self):
        im = Image(np.zeros((100, 100)))
        m = marker('line')
        m.orientation = 'v'
        m.set_marker_properties(linewidth=4, color='red', linestyle='dotted')
        m.set_data(x1=20, x2=70, y1=20, y2=70)
        m.axes_manager = im.axes_manager
        m.marker = vlines(0, 0, 0)
        m.set_line_segment()
        nose.tools.assert_true(np.allclose(m.marker.get_segments(),
                                           [np.array([[20.,  20.],
                                                      [20.,  70.]])]))
