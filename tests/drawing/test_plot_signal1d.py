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

import copy
import os
from shutil import copyfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, PickEvent
import numpy as np
import pytest
try:
    # scipy >=1.10
    from scipy.datasets import ascent, face
except ImportError:
    # scipy <1.10
    from scipy.misc import ascent, face

import hyperspy.api as hs
from hyperspy.misc.test_utils import update_close_figure
from hyperspy.signals import Signal1D
from hyperspy.drawing.signal1d import Signal1DLine
from hyperspy.tests.drawing.test_plot_signal import _TestPlot

scalebar_color = 'blue'
default_tol = 2.0
baseline_dir = 'plot_signal1d'
style_pytest_mpl = 'default'

style = ['default', 'overlap', 'cascade', 'mosaic', 'heatmap']


def _generate_filename_list(style):
    path = Path(__file__).resolve().parent
    baseline_path = path.joinpath(baseline_dir)

    filename_list = [f'test_plot_spectra_{s}' for s in style] + \
                    [f'test_plot_spectra_rev_{s}' for s in style]
    filename_list2 = []

    for filename in filename_list:
        for i in range(0, 4):
            filename_list2.append(
                baseline_path.joinpath(f'{filename}{i}.png')
            )

    return filename_list2


def _matplotlib_pick_event(figure, click, artist):
    try:
        # Introduced in matplotlib 3.6 and `pick_event` deprecated
        event = PickEvent('pick_event', figure, click, artist)
        figure.canvas.callbacks.process('pick_event', event)
    except: # Deprecated in matplotlib 3.6
        figure.canvas.pick_event(figure.canvas, click, artist)


@pytest.fixture
def setup_teardown(request, scope="class"):
    try:
        import pytest_mpl
        # This option is available only when pytest-mpl is installed
        mpl_generate_path_cmdopt = request.config.getoption("--mpl-generate-path")
    except ImportError:
        mpl_generate_path_cmdopt = None

    # SETUP
    # duplicate baseline images to match the test_name when the
    # parametrized 'test_plot_spectra' are run. For a same 'style', the
    # expected images are the same.
    if mpl_generate_path_cmdopt is None:
        for filename in _generate_filename_list(style):
            copyfile(f"{str(filename)[:-5]}.png", filename)
    yield
    # TEARDOWN
    # Create the baseline images: copy one baseline image for each test
    # and remove the other ones.
    if mpl_generate_path_cmdopt:
        for filename in _generate_filename_list(style):
            copyfile(filename, f"{str(filename)[:-5]}.png")
    # Delete the images that have been created in 'setup_class'
    for filename in _generate_filename_list(style):
        os.remove(filename)


@pytest.mark.usefixtures("setup_teardown")
class TestPlotSpectra():

    def setup_method(self, method):
        s = hs.signals.Signal1D(ascent()[100:160:10])

        # Add a test signal with decreasing axis
        s_reverse = s.deepcopy()
        s_reverse.axes_manager[1].offset = 512
        s_reverse.axes_manager[1].scale = -1
        self.s = s
        self.s_reverse = s_reverse

    def _generate_parameters(style):
        parameters = []
        for s in style:
            for fig in [True, None]:
                for ax in [True, None]:
                    parameters.append([s, fig, ax])
        return parameters

    def _generate_ids(style, duplicate=4):
        ids = []
        for s in style:
            ids.extend([s] * duplicate)
        return ids

    @pytest.mark.parametrize(("style", "fig", "ax"),
                             _generate_parameters(style),
                             ids=_generate_ids(style))
    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_spectra(self, style, fig, ax):
        if fig:
            fig = plt.figure()
        if ax:
            if fig:
                ax = fig.add_subplot(111)
            else:
                ax = plt.figure().add_subplot(111)

        ax = hs.plot.plot_spectra(self.s, style=style, legend='auto',
                                  fig=fig, ax=ax)
        if style == 'mosaic':
            ax = ax[0]
        return ax.figure

    @pytest.mark.parametrize(("style", "fig", "ax"),
                             _generate_parameters(style),
                             ids=_generate_ids(style))
    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_spectra_rev(self, style, fig, ax):
        if fig:
            fig = plt.figure()
        if ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax = hs.plot.plot_spectra(self.s_reverse, style=style, legend='auto',
                                  fig=fig, ax=ax)
        if style == 'mosaic':
            ax = ax[0]
        return ax.figure

    @pytest.mark.parametrize("figure", ['1nav', '1sig', '2nav', '2sig'])
    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_spectra_sync(self, figure):
        s1 = hs.signals.Signal1D(face()).as_signal1D(0).inav[:, :3]
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

    def test_plot_spectra_legend_pick(self):
        x = np.linspace(0., 2., 512)
        n = np.arange(1, 5)
        x_pow_n = x[None, :]**n[:, None]
        s = hs.signals.Signal1D(x_pow_n)
        my_legend = [r'x^' + str(io) for io in n]
        f = plt.figure()
        ax = hs.plot.plot_spectra(s, legend=my_legend, fig=f)
        leg = ax.get_legend()
        leg_artists = leg.get_lines()
        click = MouseEvent('button_press_event', f.canvas, 0, 0, 'left')
        for artist, li in zip(leg_artists, ax.lines[::-1]):
            _matplotlib_pick_event(f, click, artist)
            assert not li.get_visible()
            _matplotlib_pick_event(f, click, artist)
            assert li.get_visible()

    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_spectra_auto_update(self):
        s = hs.signals.Signal1D(np.arange(100))
        s2 = s / 2
        ax = hs.plot.plot_spectra([s, s2])
        s.data = -s.data
        s.events.data_changed.trigger(s)
        s2.data = -s2.data * 4 + 50
        s2.events.data_changed.trigger(s2)

        return ax.get_figure()


class TestPlotNonLinearAxis:

    def setup_method(self, method):
        dict0 = {'size': 10, 'name': 'Axis0', 'units': 'A', 'scale': 0.2,
                 'offset': 1, 'navigate': True}
        dict1 = {'axis': np.arange(100)**3, 'name': 'Axis1', 'units': 'O',
                 'navigate': False}
        np.random.seed(1)
        s = hs.signals.Signal1D(np.random.random((10, 100)),
                                axes=[dict0, dict1])
        self.s = s

    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_non_uniform_sig(self):
        self.s.plot()
        return self.s._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                                   tolerance=default_tol, style=style_pytest_mpl)
    def test_plot_non_uniform_nav(self):
        s2 = self.s.T
        s2.plot()
        return s2._plot.navigator_plot.figure


@update_close_figure()
def test_plot_nav0_close():
    test_plot = _TestPlot(ndim=0, sdim=1)
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure()
def test_plot_nav1_close():
    test_plot = _TestPlot(ndim=1, sdim=1)
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure(check_data_changed_close=False)
def test_plot_nav2_close():
    test_plot = _TestPlot(ndim=2, sdim=1)
    test_plot.signal.plot()
    return test_plot.signal


def _test_plot_two_cursors(ndim):
    test_plot = _TestPlot(ndim=ndim, sdim=1)  # sdim=2 not supported
    s = test_plot.signal
    s.metadata.General.title = 'Nav %i, Sig 1, two cursor' % ndim
    s.axes_manager[0].index = 4
    s.plot()
    s._plot.add_right_pointer()
    s._plot.navigator_plot.figure.canvas.draw()
    s._plot.signal_plot.figure.canvas.draw()
    s._plot.right_pointer.axes_manager[0].index = 2
    if ndim == 2:
        s.axes_manager[1].index = 2
        s._plot.right_pointer.axes_manager[1].index = 3
    return s


@pytest.mark.parametrize('autoscale', ['', 'x', 'xv', 'v'])
@pytest.mark.parametrize('norm', ['log', 'auto'])
def test_plot_two_cursos_parameters(autoscale, norm):
    kwargs = {'autoscale':autoscale, 'norm':norm}
    test_plot = _TestPlot(ndim=2, sdim=1)  # sdim=2 not supported
    s = test_plot.signal
    s.plot(**kwargs)
    s._plot.add_right_pointer(**kwargs)
    for line in s._plot.signal_plot.ax_lines:
        assert line.autoscale == autoscale


def _generate_parameter():
    parameters = []
    for ndim in [1, 2]:
        for plot_type in ['nav', 'sig']:
            parameters.append([ndim, plot_type])
    return parameters


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                               tolerance=default_tol, style=style_pytest_mpl)
def test_plot_log_scale():
    s = Signal1D(np.exp(-np.arange(100) / 5.0))
    s.plot(norm='log')
    return s._plot.signal_plot.figure


@pytest.mark.parametrize(("ndim", "plot_type"), _generate_parameter())
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                               tolerance=default_tol, style=style_pytest_mpl)
def test_plot_two_cursors(ndim, plot_type):
    s = _test_plot_two_cursors(ndim=ndim)

    if plot_type == "sig":
        f = s._plot.signal_plot.figure
    else:
        f= s._plot.navigator_plot.figure
    return f


@update_close_figure(check_data_changed_close=False)
def test_plot_nav2_sig1_two_cursors_close():
    return _test_plot_two_cursors(ndim=2)


def test_plot_with_non_finite_value():
    s = hs.signals.Signal1D(np.array([np.nan, 2.0]))
    s.plot()
    s.axes_manager.events.indices_changed.trigger(s.axes_manager)

    s = hs.signals.Signal1D(np.array([np.nan, np.nan]))
    s.plot()
    s.axes_manager.events.indices_changed.trigger(s.axes_manager)

    s = hs.signals.Signal1D(np.array([-np.inf, 2.0]))
    s.plot()
    s.axes_manager.events.indices_changed.trigger(s.axes_manager)

    s = hs.signals.Signal1D(np.array([np.inf, 2.0]))
    s.plot()
    s.axes_manager.events.indices_changed.trigger(s.axes_manager)


@pytest.mark.parametrize('ax', ['left', 'right'])
def test_plot_add_line_events(ax):
    s = hs.signals.Signal1D(np.arange(100))
    s.plot()
    assert len(s.axes_manager.events.indices_changed.connected) == 1
    plot = s._plot.signal_plot
    assert len(s._plot.signal_plot.figure.get_axes()) == 1

    def line_function(axes_manager=None):
        return 100 - np.arange(100)

    line = Signal1DLine()
    line.data_function = line_function
    line.set_line_properties(color='blue', type='line', scaley=False)

    if ax == 'right':
        plot.create_right_axis()
        plot.right_axes_manager = copy.deepcopy(s.axes_manager)
        expected_axis_number = 2
        expected_indices_changed_connected = 1
    else:
        expected_axis_number = 1
        expected_indices_changed_connected = 2
    plot.add_line(line, ax=ax, connect_navigation=True)

    assert len(s._plot.signal_plot.figure.get_axes()) == expected_axis_number
    line.plot()
    assert len(line.events.closed.connected) == 1
    # expected_indices_changed_connected is 2 only when adding line on the left
    # because for the right ax, we have a deepcopy of the axes_manager
    assert len(s.axes_manager.events.indices_changed.connected) == \
        expected_indices_changed_connected

    line.close()
    plot.close_right_axis()

    assert len(s._plot.signal_plot.figure.get_axes()) == 1
    assert len(line.events.closed.connected) == 0
    assert len(s.axes_manager.events.indices_changed.connected) == 1

    plot.close()
    assert len(s.axes_manager.events.indices_changed.connected) == 0
    assert s._plot.signal_plot is None


@pytest.mark.parametrize("autoscale", ['', 'x', 'xv', 'v'])
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                               tolerance=default_tol, style=style_pytest_mpl)
def test_plot_autoscale(autoscale):
    s = hs.datasets.artificial_data.get_core_loss_eels_line_scan_signal(
        add_powerlaw=True, add_noise=False)
    s.plot(autoscale=autoscale)
    ax = s._plot.signal_plot.ax
    ax.set_xlim(500.0, 700.0)
    ax.set_ylim(-10.0, 20.0)
    s.axes_manager.events.indices_changed.trigger(s.axes_manager)

    return s._plot.signal_plot.figure


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                               tolerance=default_tol, style=style_pytest_mpl)
@pytest.mark.parametrize('linestyle', [None, '-', ['-', '--']])
def test_plot_spectra_linestyle(linestyle):
    s = hs.signals.Signal1D(np.arange(100).reshape(2, 50))
    ax = hs.plot.plot_spectra(s, linestyle=linestyle)

    return ax.get_figure()


def test_plot_spectra_linestyle_error():
    from hyperspy.exceptions import VisibleDeprecationWarning
    s = hs.signals.Signal1D(np.arange(100).reshape(2, 50))
    with pytest.warns(VisibleDeprecationWarning):
        hs.plot.plot_spectra(s, line_style='--')

    with pytest.raises(ValueError):
        with pytest.warns(VisibleDeprecationWarning):
            hs.plot.plot_spectra(s, linestyle='-', line_style='--')

    with pytest.raises(ValueError):
        hs.plot.plot_spectra(s, linestyle='invalid')


def test_plot_empty_slice_autoscale():
    s = hs.signals.Signal1D(np.arange(100))
    s.plot()
    r = hs.roi.SpanROI()
    s_span = r.interactive(s)
    s_span.plot(autoscale='x')
    # change span selector to an "empty" slice and trigger update
    r.left = 24
    r.right = 24.1

    s_span.plot(autoscale='v')
    # change span selector to an "empty" slice and trigger update
    r.left = 23
    r.right = 23.1
