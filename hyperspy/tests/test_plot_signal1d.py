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
import traits.api as t
from matplotlib.testing.decorators import image_comparison, cleanup

import hyperspy.api as hs
from hyperspy.misc.test_utils import get_matplotlib_version_label, update_close_figure

mplv = get_matplotlib_version_label()
scalebar_color = 'blue'
default_tol = 0.05


def _set_navigation_axes(axes_manager, name=t.Undefined, units=t.Undefined,
                         scale=1.0, offset=0.0):
    for nav_axis in axes_manager.navigation_axes:
        nav_axis.units = units
        nav_axis.scale = scale
        nav_axis.offset = offset
    return axes_manager


def _set_signal_axes(axes_manager, name=t.Undefined, units=t.Undefined,
                     scale=1.0, offset=0.0):
    for sig_axis in axes_manager.signal_axes:
        sig_axis.name = name
        sig_axis.units = units
        sig_axis.scale = scale
        sig_axis.offset = offset
    return axes_manager


@image_comparison(baseline_images=['%s_plot_spectra_cascade' % mplv,
                                   '%s_plot_spectra_heatmap' % mplv,
                                   '%s_plot_spectra_mosaic' % mplv],
                  extensions=['png'], tol=default_tol)
def test_plot_spectra():
    import scipy.misc
    s = hs.signals.Signal1D(scipy.misc.ascent()[100:160:10])
    hs.plot.plot_spectra(s, style='cascade', legend='auto')
    hs.plot.plot_spectra(s, style='heatmap', legend='auto')
    hs.plot.plot_spectra(s, style='mosaic', legend='auto')


@image_comparison(baseline_images=['%s_plot_spectra_sync_1nav' % mplv,
                                   '%s_plot_spectra_sync_1sig' % mplv,
                                   '%s_plot_spectra_sync_2nav' % mplv,
                                   '%s_plot_spectra_sync_2sig' % mplv],
                  extensions=['png'], tol=default_tol)
def test_plot_spectra_sync():
    import scipy.misc
    s1 = hs.signals.Signal1D(scipy.misc.face()).as_signal1D(0).inav[:, :3]
    s2 = s1.deepcopy() * -1
    hs.plot.plot_signals([s1, s2])

""" Navigation 0, Signal 1 """


def _test_plot_nav0_sig1():
    s = hs.signals.Signal1D(np.arange(20))
    s.axes_manager = _set_signal_axes(s.axes_manager, name='Energy',
                                      units='eV', scale=500.0, offset=300.0)
    s.metadata.General.title = '1: Nav 0, Sig 1'
    s.plot()
    return s


@image_comparison(baseline_images=['%s_nav0_signal1_sig' % mplv],
                  extensions=['png'], tol=default_tol)
def test_plot_nav0_sig1():
    _test_plot_nav0_sig1()


@cleanup
@update_close_figure
def test_plot_nav0_sig1_close():
    return _test_plot_nav0_sig1()  # return for @update_close_figure


""" Navigation 1, Signal 1 """


def _setup_nav1_sig1(complex_data=False):
    data = np.arange(10 * 20).reshape((10, 20))
    if complex_data:
        data = data + 1j * (data + 9)
    s = hs.signals.Signal1D(data)
    s.axes_manager = _set_signal_axes(s.axes_manager, name='Energy',
                                      units='eV', scale=500.0, offset=300.0)
    s.axes_manager = _set_navigation_axes(s.axes_manager, name='',
                                          units='m', scale=1E-6, offset=5E-6)
    return s


def _test_plot_nav1_sig1():
    s = _setup_nav1_sig1()
    s.metadata.General.title = '1: Nav 1, Sig 1'
    s.plot()
    return s


@image_comparison(baseline_images=['%s_nav1_signal1_1nav' % mplv,
                                   '%s_nav1_signal1_1sig' % mplv],
                  extensions=['png'], tol=default_tol)
def test_plot_nav1_sig1():
    _test_plot_nav1_sig1()


@cleanup
@update_close_figure
def test_plot_nav1_sig1_close():
    return _test_plot_nav1_sig1()


def _test_plot_nav1_sig1_complex():
    s = _setup_nav1_sig1(complex_data=True)
    s.metadata.General.title = '2: Nav 1, Sig 1 complex'
    s.plot()
    return s


@image_comparison(baseline_images=['%s_nav1_signal1_2nav_complex' % mplv,
                                   '%s_nav1_signal1_2sig_complex' % mplv],
                  extensions=['png'], tol=default_tol)
def test_plot_nav1_sig1_complex():
    _test_plot_nav1_sig1_complex()


@cleanup
@update_close_figure
def test_plot_nav1_sig1_complex_close():
    return _test_plot_nav1_sig1_complex()

""" Navigation 2, Signal 1 """


def _setup_nav2_sig1():
    data = np.arange(5 * 10 * 20).reshape((5, 10, 20))
    s = hs.signals.Signal1D(data)
    s.axes_manager = _set_signal_axes(s.axes_manager, name='Energy',
                                      units='eV', scale=500.0, offset=300.0)
    s.axes_manager = _set_navigation_axes(s.axes_manager, name='',
                                          units='m', scale=1E-6, offset=5E-6)
    return s


def _test_plot_nav2_sig1():
    s = _setup_nav2_sig1()
    s.metadata.General.title = '1: Nav 2, Sig 1'
    s.plot()
    return s


@image_comparison(baseline_images=['%s_nav2_signal1_1nav' % mplv,
                                   '%s_nav2_signal1_1sig' % mplv],
                  extensions=['png'], tol=default_tol)
def test_plot_nav2_sig1():
    _test_plot_nav2_sig1()


@cleanup
@update_close_figure
def test_plot_nav2_sig1_close():
    return _test_plot_nav2_sig1()


def _test_plot_nav2_sig1_two_cursors():
    s = _setup_nav2_sig1()
    s.metadata.General.title = '2: Nav 2, Sig 1, two cursor'
    s.axes_manager[0].index = 5
    s.axes_manager[1].index = 2
    s.plot()
    s._plot.add_right_pointer()
    s._plot.right_pointer.axes_manager[0].index = 2
    s._plot.right_pointer.axes_manager[1].index = 2
    return s


@image_comparison(baseline_images=['%s_nav2_signal1_2nav_two_cursors' % mplv,
                                   '%s_nav2_signal1_2sig_two_cursors' % mplv],
                  extensions=['png'], tol=default_tol)
def test_plot_nav2_sig1_two_cursors():
    _test_plot_nav2_sig1_two_cursors()


@cleanup
@update_close_figure
def test_plot_nav2_sig1_two_cursors_close():
    return _test_plot_nav2_sig1_two_cursors()
