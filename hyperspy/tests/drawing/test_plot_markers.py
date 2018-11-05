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

import numpy as np
import pytest

from hyperspy.misc.test_utils import update_close_figure, sanitize_dict
from hyperspy.signals import Signal2D, Signal1D, BaseSignal
from hyperspy.utils import markers, stack
from hyperspy.drawing.marker import dict2marker
from hyperspy.datasets.example_signals import EDS_TEM_Spectrum


default_tol = 2.0
baseline_dir = 'plot_markers'
style_pytest_mpl = 'default'


class TestMarkers:

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
        assert m.marker_properties == {'color': 'red', 'fontsize': 30}

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
        m4 = markers.point(5, 5)
        m4.data['x1'][()] = np.array(None, dtype=np.object)
        m4.data['y1'][()] = np.array(None, dtype=np.object)
        m5 = markers.vertical_line(9)
        m6 = markers.rectangle(1, 5, 6, 8)
        m7 = markers.rectangle((1, 2), (5, 6), (6, 7), (8, 9))
        m8 = markers.point(
            np.arange(256).reshape(2, 2, 2, 2, 2, 2, 2, 2),
            np.arange(256).reshape(2, 2, 2, 2, 2, 2, 2, 2))
        assert m0._get_data_shape() == ()
        assert m1._get_data_shape() == (2,)
        assert m2._get_data_shape() == (2, 3)
        assert m3._get_data_shape() == (3, 2)
        with pytest.raises(ValueError):
            assert m4._get_data_shape() == ()
        assert m5._get_data_shape() == ()
        assert m6._get_data_shape() == ()
        assert m7._get_data_shape() == (2,)
        assert m8._get_data_shape() == (2, 2, 2, 2, 2, 2, 2, 2)

    def test_add_marker_not_plot(self):
        # This will do nothing, since plot_marker=False and permanent=False
        # So this test will return a _logger warning
        s = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, plot_marker=False)

    def test_add_marker_signal1d_navigation_dim(self, mpl_cleanup):
        s = Signal1D(np.zeros((3, 50, 50)))
        m0 = markers.point(5, 5)
        m1 = markers.point((5, 10), (10, 15))
        m2 = markers.point(np.zeros((50, 3)), np.zeros((50, 3)))
        s.add_marker(m0)
        with pytest.raises(ValueError):
            s.add_marker(m1)
        s.add_marker(m2)

    def test_add_marker_signal2d_navigation_dim(self, mpl_cleanup):
        s = Signal2D(np.zeros((3, 50, 50)))
        m0 = markers.point(5, 5)
        m1 = markers.point((5, 10), (10, 15))
        m2 = markers.point(np.zeros((3, )), np.zeros((3, )))
        s.add_marker(m0)
        with pytest.raises(ValueError):
            s.add_marker(m1)
        s.add_marker(m2)

    def test_add_markers_as_list(self, mpl_cleanup):
        s = Signal1D(np.arange(10))
        marker_list = []
        for i in range(12):
            marker_list.append(markers.point(4, 8))
        s.add_marker(marker_list)


class Test_permanent_markers:

    def test_add_permanent_marker(self, mpl_cleanup):
        s = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        assert list(s.metadata.Markers)[0][1] == m

    def test_add_permanent_marker_not_plot(self):
        s = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True, plot_marker=False)
        assert list(s.metadata.Markers)[0][1] == m

    def test_remove_permanent_marker_name(self, mpl_cleanup):
        s = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        m.name = 'test'
        s.add_marker(m, permanent=True)
        assert list(s.metadata.Markers)[0][1] == m
        del s.metadata.Markers.test
        assert len(list(s.metadata.Markers)) == 0

    def test_permanent_marker_names(self, mpl_cleanup):
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

    def test_add_permanent_marker_twice(self, mpl_cleanup):
        s = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        with pytest.raises(ValueError):
            s.add_marker(m, permanent=True)

    def test_add_permanent_marker_twice_different_signal(self, mpl_cleanup):
        s0 = Signal1D(np.arange(10))
        s1 = Signal1D(np.arange(10))
        m = markers.point(x=5, y=5)
        s0.add_marker(m, permanent=True)
        with pytest.raises(ValueError):
            s1.add_marker(m, permanent=True)

    def test_add_several_permanent_markers(self, mpl_cleanup):
        s = Signal1D(np.arange(10))
        m_point = markers.point(x=5, y=5)
        m_line = markers.line_segment(x1=5, x2=10, y1=5, y2=10)
        m_vline = markers.vertical_line(x=5)
        m_vline_segment = markers.vertical_line_segment(x=4, y1=3, y2=6)
        m_hline = markers.horizontal_line(y=5)
        m_hline_segment = markers.horizontal_line_segment(x1=1, x2=9, y=5)
        m_rect = markers.rectangle(x1=1, x2=3, y1=5, y2=10)
        m_text = markers.text(x=1, y=5, text="test")
        s.add_marker(m_point, permanent=True)
        s.add_marker(m_line, permanent=True)
        s.add_marker(m_vline, permanent=True)
        s.add_marker(m_vline_segment, permanent=True)
        s.add_marker(m_hline, permanent=True)
        s.add_marker(m_hline_segment, permanent=True)
        s.add_marker(m_rect, permanent=True)
        s.add_marker(m_text, permanent=True)
        assert len(list(s.metadata.Markers)) == 8
        with pytest.raises(ValueError):
            s.add_marker(m_rect, permanent=True)

    def test_add_markers_as_list(self, mpl_cleanup):
        s = Signal1D(np.arange(10))
        marker_list = []
        for i in range(10):
            marker_list.append(markers.point(1, 2))
        s.add_marker(marker_list, permanent=True)
        assert len(s.metadata.Markers) == 10

    def test_add_markers_as_list_add_same_twice(self, mpl_cleanup):
        s = Signal1D(np.arange(10))
        marker_list = []
        for i in range(10):
            marker_list.append(markers.point(1, 2))
        s.add_marker(marker_list, permanent=True)
        with pytest.raises(ValueError):
            s.add_marker(marker_list, permanent=True)

    def test_add_markers_as_list_add_different_twice(self, mpl_cleanup):
        s = Signal1D(np.arange(10))
        marker_list0 = []
        for i in range(10):
            marker_list0.append(markers.point(1, 2))
        s.add_marker(marker_list0, permanent=True)
        assert len(s.metadata.Markers) == 10
        marker_list1 = []
        for i in range(10):
            marker_list1.append(markers.point(4, 8))
        s.add_marker(marker_list1, permanent=True)
        assert len(s.metadata.Markers) == 20

    def test_add_permanent_marker_signal2d(self, mpl_cleanup):
        s = Signal2D(np.arange(100).reshape(10, 10))
        m = markers.point(x=5, y=5)
        s.add_marker(m, permanent=True)
        assert list(s.metadata.Markers)[0][1] == m

    def test_deepcopy_permanent_marker(self, mpl_cleanup):
        x, y, color, name = 2, 9, 'blue', 'test_point'
        s = Signal2D(np.arange(100).reshape(10, 10))
        m = markers.point(x=x, y=y, color=color)
        m.name = name
        s.add_marker(m, permanent=True)
        s1 = s.deepcopy()
        m1 = s1.metadata.Markers.get_item(name)
        assert m.get_data_position('x1') == m1.get_data_position('x1')
        assert m.get_data_position('y1') == m1.get_data_position('y1')
        assert m.name == m1.name
        assert m.marker_properties['color'] == m1.marker_properties['color']

    def test_dict2marker(self):
        m_point0 = markers.point(x=5, y=5)
        m_point1 = markers.point(x=(5, 10), y=(1, 5))
        m_line = markers.line_segment(x1=5, x2=10, y1=5, y2=10)
        m_vline = markers.vertical_line(x=5)
        m_vline_segment = markers.vertical_line_segment(x=4, y1=3, y2=6)
        m_hline = markers.horizontal_line(y=5)
        m_hline_segment = markers.horizontal_line_segment(x1=1, x2=9, y=5)
        m_rect = markers.rectangle(x1=1, x2=3, y1=5, y2=10)
        m_text = markers.text(x=1, y=5, text="test")

        m_point0_new = dict2marker(m_point0._to_dictionary(), m_point0.name)
        m_point1_new = dict2marker(m_point1._to_dictionary(), m_point1.name)
        m_line_new = dict2marker(m_line._to_dictionary(), m_line.name)
        m_vline_new = dict2marker(m_vline._to_dictionary(), m_vline.name)
        m_vline_segment_new = dict2marker(
            m_vline_segment._to_dictionary(), m_vline_segment.name)
        m_hline_new = dict2marker(m_hline._to_dictionary(), m_hline.name)
        m_hline_segment_new = dict2marker(
            m_hline_segment._to_dictionary(), m_hline_segment.name)
        m_rect_new = dict2marker(m_rect._to_dictionary(), m_rect.name)
        m_text_new = dict2marker(m_text._to_dictionary(), m_text.name)

        m_point0_dict = sanitize_dict(m_point0._to_dictionary())
        m_point1_dict = sanitize_dict(m_point1._to_dictionary())
        m_line_dict = sanitize_dict(m_line._to_dictionary())
        m_vline_dict = sanitize_dict(m_vline._to_dictionary())
        m_vline_segment_dict = sanitize_dict(m_vline_segment._to_dictionary())
        m_hline_dict = sanitize_dict(m_hline._to_dictionary())
        m_hline_segment_dict = sanitize_dict(m_hline_segment._to_dictionary())
        m_rect_dict = sanitize_dict(m_rect._to_dictionary())
        m_text_dict = sanitize_dict(m_text._to_dictionary())

        m_point0_new_dict = sanitize_dict(m_point0_new._to_dictionary())
        m_point1_new_dict = sanitize_dict(m_point1_new._to_dictionary())
        m_line_new_dict = sanitize_dict(m_line_new._to_dictionary())
        m_vline_new_dict = sanitize_dict(m_vline_new._to_dictionary())
        m_vline_segment_new_dict = sanitize_dict(
            m_vline_segment_new._to_dictionary())
        m_hline_new_dict = sanitize_dict(m_hline_new._to_dictionary())
        m_hline_segment_new_dict = sanitize_dict(
            m_hline_segment_new._to_dictionary())
        m_rect_new_dict = sanitize_dict(m_rect_new._to_dictionary())
        m_text_new_dict = sanitize_dict(m_text_new._to_dictionary())
        assert m_point0_dict == m_point0_new_dict
        assert m_point1_dict == m_point1_new_dict
        assert m_line_dict == m_line_new_dict
        assert m_vline_dict == m_vline_new_dict
        assert m_vline_segment_dict == m_vline_segment_new_dict
        assert m_hline_dict == m_hline_new_dict
        assert m_hline_segment_dict == m_hline_segment_new_dict
        assert m_rect_dict == m_rect_new_dict
        assert m_text_dict == m_text_new_dict


def _test_plot_rectange_markers():
    # Create test image 100x100 pixels:
    im = Signal2D(np.arange(100).reshape([10, 10]))

    # Add four line markers:
    m1 = markers.line_segment(
        x1=2, y1=2, x2=7, y2=2, color='red', linewidth=3)
    m2 = markers.line_segment(
        x1=2, y1=2, x2=2, y2=7, color='red', linewidth=3)
    m3 = markers.line_segment(
        x1=2, y1=7, x2=7, y2=7, color='red', linewidth=3)
    m4 = markers.line_segment(
        x1=7, y1=2, x2=7, y2=7, color='red', linewidth=3)

    # Add rectangle marker at same position:
    m = markers.rectangle(x1=2, x2=7, y1=2, y2=7,
                          linewidth=4, color='blue', ls='dotted')

    # Plot image and add markers to img:
    im.plot()
    im.add_marker(m)
    im.add_marker(m1)
    im.add_marker(m2)
    im.add_marker(m3)
    im.add_marker(m4)
    return im


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_rectange_markers(mpl_cleanup):
    im = _test_plot_rectange_markers()
    return im._plot.signal_plot.figure


@update_close_figure
def test_plot_rectange_markers_close():
    return _test_plot_rectange_markers()  # return for @update_close_figure


def _test_plot_point_markers():
    width = 100
    data = np.arange(width * width).reshape((width, width))
    s = Signal2D(data)

    x, y = 10 * np.arange(4), 15 * np.arange(4)
    color = ['yellow', 'green', 'red', 'blue']
    for xi, yi, c in zip(x, y, color):
        m = markers.point(x=xi, y=yi, color=c)
        s.add_marker(m)
    return s


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_point_markers(mpl_cleanup):
    s = _test_plot_point_markers()
    return s._plot.signal_plot.figure


@update_close_figure
def test_plot_point_markers_close():
    return _test_plot_point_markers()


def _test_plot_text_markers():
    s = Signal1D(np.arange(100).reshape([10, 10]))
    s.plot(navigator='spectrum')
    for i in range(s.axes_manager.shape[0]):
        m = markers.text(y=s.sum(-1).data[i] + 5, x=i, text='abcdefghij'[i])
        s.add_marker(m, plot_on_signal=False)
    x = s.axes_manager.shape[-1] / 2  # middle of signal plot
    m = markers.text(x=x, y=s.inav[x].data + 2, text=[i for i in 'abcdefghij'])
    s.add_marker(m)
    return s


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_text_markers_nav(mpl_cleanup):
    s = _test_plot_text_markers()
    return s._plot.navigator_plot.figure


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_text_markers_sig(mpl_cleanup):
    s = _test_plot_text_markers()
    return s._plot.signal_plot.figure


@update_close_figure
def test_plot_text_markers_close():
    return _test_plot_text_markers()


def _test_plot_line_markers():
    im = Signal2D(np.arange(100 * 100).reshape((100, 100)))
    m0 = markers.vertical_line_segment(x=20, y1=30, y2=70, linewidth=4,
                                       color='red', linestyle='dotted')
    im.add_marker(m0)
    m1 = markers.horizontal_line_segment(x1=30, x2=20, y=80, linewidth=8,
                                         color='blue', linestyle='-')
    im.add_marker(m1)
    m2 = markers.vertical_line(50, linewidth=12, color='green')
    im.add_marker(m2)
    m3 = markers.horizontal_line(50, linewidth=10, color='yellow')
    im.add_marker(m3)
    return im


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_line_markers(mpl_cleanup):
    im = _test_plot_line_markers()
    return im._plot.signal_plot.figure


@update_close_figure
def test_plot_line_markers_close():
    return _test_plot_line_markers()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_eds_lines():
    a = EDS_TEM_Spectrum()
    s = stack([a, a * 5])
    s.plot(True)
    s.axes_manager.navigation_axes[0].index = 1
    return s._plot.signal_plot.figure


def test_iterate_markers():
    from skimage.feature import peak_local_max
    import scipy.misc
    ims = BaseSignal(scipy.misc.face()).as_signal2D([1, 2])
    index = np.array([peak_local_max(im.data, min_distance=100,
                                     num_peaks=4) for im in ims])
    # Add multiple markers
    for i in range(4):
        xs = index[:, i, 1]
        ys = index[:, i, 0]
        m = markers.point(x=xs, y=ys, color='red')
        ims.add_marker(m, plot_marker=True, permanent=True)
        m = markers.text(x=10 + xs, y=10 + ys, text=str(i), color='k')
        ims.add_marker(m, plot_marker=True, permanent=True)
    xs = index[:, :, 1]
    ys = index[:, :, 0]
    m = markers.rectangle(np.min(xs, 1),
                          np.min(ys, 1),
                          np.max(xs, 1),
                          np.max(ys, 1),
                          color='green')
    ims.add_marker(m, plot_marker=True, permanent=True)

    for im in ims:
        m_original = ims.metadata.Markers
        m_iterated = im.metadata.Markers
        for key in m_original.keys():
            mo = m_original[key]
            mi = m_iterated[key]
            assert mo.__class__.__name__ == mi.__class__.__name__
            assert mo.name == mi.name
            assert mo.get_data_position('x1') == mi.get_data_position('x1')
            assert mo.get_data_position('y1') == mi.get_data_position('y1')
            assert mo.get_data_position('text') == mi.get_data_position('text')
            assert mo.marker_properties['color'] == \
                mi.marker_properties['color']


@update_close_figure
def test_plot_eds_markers_close():
    s = EDS_TEM_Spectrum()
    s.plot(True)
    return s


def test_plot_eds_markers_no_energy():
    s = EDS_TEM_Spectrum()
    del s.metadata.Acquisition_instrument.TEM.beam_energy
    s.plot(True)
