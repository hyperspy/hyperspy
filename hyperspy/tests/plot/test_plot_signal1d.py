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

import os
import numpy as np
import scipy.misc
import traits.api as t
import pytest
from matplotlib.testing.decorators import cleanup

import hyperspy.api as hs
from hyperspy.misc.test_utils import get_matplotlib_version_label, update_close_figure

mplv = get_matplotlib_version_label()
scalebar_color = 'blue'
default_tol = 2.0
path = os.path.join('plot_signal1d-%s' % mplv)


class TestPlotSpectra():

    def _test_plot_spectra(self):
        return hs.signals.Signal1D(scipy.misc.ascent()[100:160:10])

    @pytest.mark.parametrize("style", ['default', 'overlap', 'cascade', 'mosaic',
                                       'heatmap'])
    @pytest.mark.mpl_image_compare(baseline_dir=path, tolerance=default_tol)
    def test_plot_spectra(self, style):
        ax = hs.plot.plot_spectra(self._test_plot_spectra(), style=style,
                                  legend='auto')
        if style == 'mosaic':
            ax = ax[0]
        return ax.figure

    @pytest.mark.parametrize("figure", ['1nav', '1sig', '2nav', '2sig'])
    @pytest.mark.mpl_image_compare(baseline_dir=path, tolerance=default_tol)
    def test_plot_spectra_sync(self, figure):
        s1 = hs.signals.Signal1D(scipy.misc.face()).as_signal1D(0).inav[:, :3]
        s2 = s1.deepcopy() * -1
        hs.plot.plot_signals([s1, s2])
        if figure == '1nav':
            return s1._plot.signal_plot.figure
        if figure == '1sig':
            return s1._plot.navigator_plot.figure
        if figure == '2nav':
            return s2._plot.navigator_plot.figure
        if figure == '2sig':
            return s2._plot.navigator_plot.figure


class _TestPlot:

    def __init__(self, ndim=0, data_type='real'):
        shape = np.arange(1, ndim + 2) * 5
        n = 1
        for i in shape:
            n *= i
        data = np.arange(n).reshape(shape)
        title = 'Signal: 1, Navigator: %i' % ndim
        if data_type == 'complex':
            data = data + 1j * (data + 9)
            title += ', complex'
        s = hs.signals.Signal1D(data)
        s.axes_manager = self._set_signal_axes(s.axes_manager, name='Energy',
                                               units='eV', scale=500.0, offset=300.0)
        if ndim > 0:
            s.axes_manager = self._set_navigation_axes(s.axes_manager, name='',
                                                       units='m', scale=1E-6,
                                                       offset=5E-6)
        s.metadata.General.title = title
        s.plot()
        self.signal = s

    def _set_navigation_axes(self, axes_manager, name=t.Undefined,
                             units=t.Undefined, scale=1.0, offset=0.0):
        for nav_axis in axes_manager.navigation_axes:
            nav_axis.units = units
            nav_axis.scale = scale
            nav_axis.offset = offset
        return axes_manager

    def _set_signal_axes(self, axes_manager, name=t.Undefined,
                         units=t.Undefined, scale=1.0, offset=0.0):
        for sig_axis in axes_manager.signal_axes:
            sig_axis.name = name
            sig_axis.units = units
            sig_axis.scale = scale
            sig_axis.offset = offset
        return axes_manager


def _generate_parameter():
    parameters = []
    for ndim in [0, 1, 2]:
        for plot_type in ['nav', 'sig']:
            for data_type in ['complex', 'real']:
                if ndim == 0 and plot_type == "nav":  # in this case, no nav figure
                    pass
                else:
                    parameters.append([ndim, plot_type, data_type])
    return parameters


@pytest.mark.parametrize(("ndim", "plot_type", "data_type"),
                         _generate_parameter())
@pytest.mark.mpl_image_compare(baseline_dir=path, tolerance=default_tol)
def test_plot_sig1_nav(ndim, plot_type, data_type):
    test_plot = _TestPlot(ndim, data_type)
    if plot_type == "sig":
        return test_plot.signal._plot.signal_plot.figure
    else:
        return test_plot.signal._plot.navigator_plot.figure


@cleanup
@update_close_figure
def test_plot_nav0_close():
    test_plot = _TestPlot(ndim=0)
    return test_plot.signal


@cleanup
@update_close_figure
def test_plot_nav1_close():
    test_plot = _TestPlot(ndim=1)
    return test_plot.signal


@cleanup
@update_close_figure
def test_plot_nav2_close():
    test_plot = _TestPlot(ndim=2)
    return test_plot.signal


def _test_plot_nav2_sig1_two_cursors():
    test_plot = _TestPlot(ndim=2)
    s = test_plot.signal
    s.metadata.General.title = 'Nav 2, Sig 1, two cursor'
    s.axes_manager[0].index = 5
    s.axes_manager[1].index = 2
    s.plot()
    s._plot.add_right_pointer()
    s._plot.right_pointer.axes_manager[0].index = 2
    s._plot.right_pointer.axes_manager[1].index = 2
    return s


@pytest.mark.parametrize("plot_type", ['nav', 'sig'])
@pytest.mark.mpl_image_compare(baseline_dir=path, tolerance=default_tol)
def test_plot_nav2_sig1_two_cursors(plot_type):
    s = _test_plot_nav2_sig1_two_cursors()
    if plot_type == "sig":
        return s._plot.signal_plot.figure
    else:
        return s._plot.navigator_plot.figure


@cleanup
@update_close_figure
def test_plot_nav2_sig1_two_cursors_close():
    return _test_plot_nav2_sig1_two_cursors()
