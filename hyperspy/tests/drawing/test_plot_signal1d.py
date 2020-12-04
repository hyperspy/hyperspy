# Copyright 2007-2020 The HyperSpy developers
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
from shutil import copyfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.misc

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


@pytest.fixture
def mpl_generate_path_cmdopt(request):
    return request.config.getoption("--mpl-generate-path")


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


@pytest.fixture
def setup_teardown(request, scope="class"):
    mpl_generate_path_cmdopt = request.config.getoption("--mpl-generate-path")
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

    s = hs.signals.Signal1D(scipy.misc.ascent()[100:160:10])

    # Add a test signal with decreasing axis
    s_reverse = s.deepcopy()
    s_reverse.axes_manager[1].offset = 512
    s_reverse.axes_manager[1].scale = -1

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
            fig = plt.figure()
            ax = fig.add_subplot(111)

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
        click = plt.matplotlib.backend_bases.MouseEvent(
            'button_press_event', f.canvas, 0, 0, 'left')
        for artist, li in zip(leg_artists, ax.lines[::-1]):
            plt.matplotlib.backends.backend_agg.FigureCanvasBase.pick_event(
                f.canvas, click, artist)
            assert not li.get_visible()
            plt.matplotlib.backends.backend_agg.FigureCanvasBase.pick_event(
                f.canvas, click, artist)

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


@update_close_figure
def test_plot_nav0_close():
    test_plot = _TestPlot(ndim=0, sdim=1)
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure
def test_plot_nav1_close():
    test_plot = _TestPlot(ndim=1, sdim=1)
    test_plot.signal.plot()
    return test_plot.signal


@update_close_figure
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


@pytest.mark.parametrize(("ndim", "plot_type"),
                         _generate_parameter())
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                               tolerance=default_tol, style=style_pytest_mpl)
def test_plot_two_cursors(ndim, plot_type):
    s = _test_plot_two_cursors(ndim=ndim)

    if plot_type == "sig":
        f = s._plot.signal_plot.figure
    else:
        f= s._plot.navigator_plot.figure
    f.canvas.draw()
    f.canvas.flush_events()
    return f


@update_close_figure
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


def test_plot_add_line_events():
    s = hs.signals.Signal1D(np.arange(100))
    s.plot()
    assert len(s.axes_manager.events.indices_changed.connected) == 1
    figure = s._plot.signal_plot

    def line_function(axes_manager=None):
        return 100 - np.arange(100)

    line = Signal1DLine()
    line.data_function = line_function
    line.set_line_properties(color='blue', type='line', scaley=False)
    figure.add_line(line, connect_navigation=True)
    line.plot()
    assert len(line.events.closed.connected) == 1
    assert len(s.axes_manager.events.indices_changed.connected) == 2

    line.close()
    figure.close_right_axis()

    assert len(line.events.closed.connected) == 0
    assert len(s.axes_manager.events.indices_changed.connected) == 1

    figure.close()
    assert len(s.axes_manager.events.indices_changed.connected) == 0


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
