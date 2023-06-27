# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import matplotlib.pyplot as plt
import numpy as np
import pytest
import traits.api as t

import hyperspy.api as hs
from hyperspy.drawing.signal1d import Signal1DFigure, Signal1DLine
from hyperspy.drawing.image import ImagePlot
from hyperspy.misc.test_utils import update_close_figure, check_closing_plot


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
        s = getattr(hs.signals, f'{dtype}Signal{sdim}D')(data)
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
def test_plot_sig_nav(ndim, sdim, plot_type, data_type):
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
            plot_part = 'real_plot'
        elif data_type == "complex_imag":
            plot_part = 'real_plot'
        fig = getattr(getattr(test_plot, plot_part), plot).figure
    else:
        fig = getattr(test_plot.signal._plot, plot).figure
    return fig


@update_close_figure()
def test_plot_nav0_sig1_close():
    test_plot = _TestPlot(ndim=0, sdim=1, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure()
def test_plot_nav1_sig1_close():
    test_plot = _TestPlot(ndim=1, sdim=1, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure(check_data_changed_close=False)
def test_plot_nav2_sig1_close():
    test_plot = _TestPlot(ndim=2, sdim=1, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure()
def test_plot_nav0_sig2_close():
    test_plot = _TestPlot(ndim=0, sdim=2, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure(check_data_changed_close=False)
def test_plot_nav1_sig2_close():
    test_plot = _TestPlot(ndim=1, sdim=2, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure(check_data_changed_close=False)
def test_plot_nav2_sig2_close():
    test_plot = _TestPlot(ndim=2, sdim=2, data_type="real")
    test_plot.signal.plot()
    return test_plot.signal


@pytest.mark.parametrize("sdim", [1, 2])
def test_plot_close_cycle(sdim):
    test_plot = _TestPlot(ndim=2, sdim=sdim, data_type="real")
    s = test_plot.signal
    s.plot()
    s._plot.close()
    assert s._plot.signal_plot is None
    assert s._plot.navigator_plot is None
    s.plot()
    assert s._plot.signal_plot is not None
    assert s._plot.navigator_plot is not None
    s._plot.close()


@pytest.mark.parametrize('autoscale', ['', 'x', 'xv', 'v'])
@pytest.mark.parametrize("ndim", [1, 2])
def test_plot_navigator_kwds(ndim, autoscale):
    test_plot_nav1d = _TestPlot(ndim=ndim, sdim=2, data_type="real")
    s = test_plot_nav1d.signal
    s.plot(navigator_kwds={'norm':'log', 'autoscale':autoscale})
    if ndim == 1:
        assert isinstance(s._plot.navigator_plot, Signal1DFigure)
        plot = s._plot.navigator_plot.ax_lines[0]
        assert isinstance(plot, Signal1DLine)
    else:
        plot = s._plot.navigator_plot
        assert isinstance(plot, ImagePlot)

    assert plot.norm == 'log'
    assert plot.autoscale == autoscale
    s._plot.close()


def test_plot_signal_dim0():
    s = hs.signals.BaseSignal(np.arange(100)).T
    s.plot()
    assert s._plot.signal_plot is None
    assert s._plot.navigator_plot is not None
    s._plot.close()
    check_closing_plot(s)


@pytest.mark.parametrize('bool_value', [True, False])
@pytest.mark.parametrize("sdim", [1, 2])
def test_data_function_kwargs(sdim, bool_value):
    test_plot_nav1d = _TestPlot(ndim=1, sdim=sdim, data_type="complex")
    s = test_plot_nav1d.signal
    s.plot(power_spectrum=bool_value, fft_shift=bool_value)
    if sdim == 1:
        for key in ['power_spectrum', 'fft_shift']:
            assert s._plot.signal_data_function_kwargs[key] is bool_value
    else:
        for key in ['power_spectrum', 'fft_shift']:
            assert s._plot_kwargs[key] is bool_value


def test_plot_power_spectrum():
    s = hs.signals.Signal1D(np.arange(100))
    with pytest.raises(ValueError):
        s.plot(power_spectrum=True)

    s = hs.signals.ComplexSignal1D(np.arange(100))
    s.plot(power_spectrum=True)
    assert s._plot.signal_data_function_kwargs['power_spectrum'] is True


@pytest.mark.parametrize("sdim", [1, 2])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_plot_slider(ndim, sdim):
    test_plot_nav1d = _TestPlot(ndim=ndim, sdim=sdim, data_type="real")
    s = test_plot_nav1d.signal
    # Plot twice to check that the args of the second call are used.
    s.plot()
    s.plot(navigator='slider')
    assert s._plot.signal_plot is not None
    assert s._plot.navigator_plot is None
    s._plot.close()
    check_closing_plot(s, check_data_changed_close=False)

    if ndim > 1:
        s.plot(navigator='spectrum')
        assert s._plot.signal_plot is not None
        assert s._plot.navigator_plot is not None
        assert isinstance(s._plot.navigator_plot, Signal1DFigure)
        s._plot.close()
        check_closing_plot(s, check_data_changed_close=False)
    if ndim > 2:
        s.plot()
        assert s._plot.signal_plot is not None
        assert s._plot.navigator_plot is not None
        assert len(s.axes_manager.events.indices_changed.connected) >= 2
        s._plot.close()
        check_closing_plot(s, check_data_changed_close=False)


@pytest.mark.parametrize("tranpose", [True, False])
@pytest.mark.parametrize("ndim", [1, 2])
def test_plot_navigator_plot_signal(ndim, tranpose):
    test_plot_nav1d = _TestPlot(ndim=ndim, sdim=1, data_type="real")
    s = test_plot_nav1d.signal
    navigator = -s.sum(-1)
    if tranpose:
        navigator = navigator.T
    s.plot(navigator=navigator)
    if ndim == 1:
        navigator_data = s._plot.navigator_plot.ax_lines[0]._get_data()
    else:
        navigator_data = s._plot.navigator_plot._current_data
    np.testing.assert_allclose(navigator_data, navigator.data)
    s._plot.close()
    check_closing_plot(s)

    s.plot(navigator=None)
    assert s._plot.signal_plot is not None
    assert s._plot.navigator_plot is None
    s._plot.close()
    check_closing_plot(s)


@pytest.mark.parametrize("sdim", [1, 2])
def test_plot_autoscale(sdim):
    test_plot_nav1d = _TestPlot(ndim=1, sdim=sdim, data_type="real")
    s = test_plot_nav1d.signal
    with pytest.raises(ValueError):
        s.plot(autoscale='xa')

    s.change_dtype(bool)
    s.plot()


def test_plot_complex_representation():
    real_ref = np.arange(9).reshape((3, 3))
    imag_ref = np.arange(9).reshape((3, 3)) + 9
    comp_ref = real_ref + 1j * imag_ref
    s = hs.signals.ComplexSignal1D(comp_ref)
    s.plot()
    # change indices to trigger update
    s.axes_manager.indices = (1, )    
    s.plot(representation='polar', same_axes=True)
    s.plot(representation='polar', same_axes=False)
    with pytest.raises(ValueError):
        s.plot(representation='unsupported_argument')


def test_plot_signal_scalar():
    s = hs.signals.BaseSignal([1.0])
    s.plot()
    assert s._plot is None


@pytest.mark.parametrize('lazy', [True, False])
def test_plot_ragged_array(lazy):
    data = np.empty((2, 5), dtype=object)
    data.fill(np.array([10, 20]))

    s = hs.signals.BaseSignal(data, ragged=True)
    if lazy:
        s = s.as_lazy()
    with pytest.raises(RuntimeError):
        s.plot()
