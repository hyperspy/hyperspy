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
import pytest
import matplotlib.pyplot as plt

from hyperspy.misc.test_utils import update_close_figure
import hyperspy.api as hs


scalebar_color = 'blue'
default_tol = 2.0
baseline_dir = 'plot_signal'
style_pytest_mpl = 'default'


class _TestPlot:

    def __init__(self, ndim, sdim, data_type='real'):
        shape = np.arange(1, ndim + sdim + 1) * 5
        n = 1
        for i in shape:
            n *= i
        data = np.arange(n).reshape(shape)
        title = 'Signal: %i, Navigator: %i' % (sdim, ndim)
        dtype = ''
        if 'complex' in data_type:
            data = data + 1j * (data + 9)
            title += ', complex'
            dtype = 'Complex'
        s = hs.signals.__dict__['%sSignal%iD' % (dtype, sdim)](data)
        if sdim == 1:
            s.axes_manager = self._set_signal_axes(s.axes_manager, name='Energy',
                                                   units='keV', scale=.5, offset=0.3)
        elif sdim == 2:
            s.axes_manager = self._set_signal_axes(s.axes_manager, name='Reciprocal distance',
                                                   units='1/nm', scale=1, offset=0.0)
        if ndim > 0:
            s.axes_manager = self._set_navigation_axes(s.axes_manager, name='',
                                                       units='nm', scale=1.0,
                                                       offset=5.0)
        s.metadata.General.title = title
        # workaround to be able to access the figure in case of complex 2d
        # signals
        if 'complex' in data_type and sdim == 2:
            real = s.real
            real.plot()
            self.real_plot = real._plot
            imag = s.imag
            imag.plot()
            self.imag_plot = imag._plot
        self.signal = s
        self.sdim = sdim

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
        for sdim in [1, 2]:
            for plot_type in ['nav', 'sig']:
                # For complex 2D, there are 4 figures generated, some of these
                # tests are redondants
                for data_type in ['real', 'complex_real', 'complex_imag']:
                    if ndim == 0 and plot_type == "nav":  # in this case, no nav figure
                        pass
                    else:
                        parameters.append([ndim, sdim, plot_type, data_type])
    return parameters


@pytest.mark.parametrize(("ndim", "sdim", "plot_type", "data_type"),
                         _generate_parameter())
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_sig_nav(mpl_cleanup, ndim, sdim, plot_type, data_type):
    test_plot = _TestPlot(ndim, sdim, data_type)
    test_plot.signal.plot()
    return _get_figure(test_plot, data_type, plot_type)


@pytest.mark.parametrize("sdim", [1, 2])
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_data_changed_event(sdim):
    if sdim == 2:
        s = hs.signals.Signal2D(np.arange(25).reshape((5, 5)))
    else:
        s = hs.signals.Signal1D(np.arange(25))
    s.plot()
    s.data *= -2
    s.events.data_changed.trigger(obj=s)
    return plt.gcf()


def _get_figure(test_plot, data_type, plot_type):
    if plot_type == "sig":
        plot = "signal_plot"
    elif plot_type == "nav":
        plot = "navigator_plot"

    if "complex" in data_type and test_plot.sdim == 2:
        if data_type == "complex_real":
            fig = test_plot.real_plot.__dict__[plot].figure
        elif data_type == "complex_imag":
            fig = test_plot.real_plot.__dict__[plot].figure
    else:
        fig = test_plot.signal._plot.__dict__[plot].figure
    return fig


@update_close_figure
def test_plot_nav0_sig1_close():
    test_plot = _TestPlot(ndim=0, sdim=1, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure
def test_plot_nav1_sig1_close():
    test_plot = _TestPlot(ndim=1, sdim=1, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure
def test_plot_nav2_sig1_close():
    test_plot = _TestPlot(ndim=2, sdim=1, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure
def test_plot_nav0_sig2_close():
    test_plot = _TestPlot(ndim=0, sdim=2, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure
def test_plot_nav1_sig2_close():
    test_plot = _TestPlot(ndim=1, sdim=2, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure
def test_plot_nav2_sig2_close():
    test_plot = _TestPlot(ndim=2, sdim=2, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal
