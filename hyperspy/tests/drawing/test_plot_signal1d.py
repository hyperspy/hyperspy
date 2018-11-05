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
import scipy.misc
import pytest
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import numpy as np

import hyperspy.api as hs
from hyperspy.misc.test_utils import update_close_figure
from hyperspy.signals import Signal1D
from hyperspy.tests.drawing.test_plot_signal import _TestPlot


scalebar_color = 'blue'
default_tol = 2.0
baseline_dir = 'plot_signal1d'
style_pytest_mpl = 'default'

style = ['default', 'overlap', 'cascade', 'mosaic', 'heatmap']


def _generate_filename_list(style):
    path = os.path.dirname(__file__)
    filename_list = ['test_plot_spectra_%s' % s for s in style] + \
                    ['test_plot_spectra_rev_%s' % s for s in style]
    filename_list2 = []
    for filename in filename_list:
        for i in range(0, 4):
            filename_list2.append(os.path.join(path, baseline_dir,
                                               '%s%i.png' % (filename, i)))
    return filename_list2


class TestPlotSpectra():

    s = hs.signals.Signal1D(scipy.misc.ascent()[100:160:10])

    # Add a test signal with decreasing axis
    s_reverse = s.deepcopy()
    s_reverse.axes_manager[1].offset = 512
    s_reverse.axes_manager[1].scale = -1

    @classmethod
    def setup_class(cls):
        # duplicate baseline images to match the test_name when the
        # parametrized 'test_plot_spectra' are run. For a same 'style', the
        # expected images are the same.
        if pytest.config.getoption("--mpl-generate-path") is None:
            for filename in _generate_filename_list(style):
                copyfile("%s.png" % filename[:-5], filename)

    @classmethod
    def teardown_class(cls):
        # Create the baseline images: copy one baseline image for each test
        # and remove the other ones.
        if pytest.config.getoption("--mpl-generate-path"):
            for filename in _generate_filename_list(style):
                copyfile(filename, "%s.png" % filename[:-5])
        # Delete the images that have been created in 'setup_class'
        for filename in _generate_filename_list(style):
            os.remove(filename)

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
    def test_plot_spectra(self, mpl_cleanup, style, fig, ax):
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
    def test_plot_spectra_rev(self, mpl_cleanup, style, fig, ax):
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
    def test_plot_spectra_sync(self, mpl_cleanup, figure):
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

    def test_plot_spectra_legend_pick(self, mpl_cleanup):
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


def _generate_parameter():
    parameters = []
    for ndim in [1, 2]:
        for plot_type in ['nav', 'sig']:
            parameters.append([ndim, plot_type])
    return parameters


@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                               tolerance=default_tol, style=style_pytest_mpl)
def test_plot_log_scale(mpl_cleanup):
    s = Signal1D(np.exp(-np.arange(100) / 5.0))
    s.plot(norm='log')
    return s._plot.signal_plot.figure


@pytest.mark.parametrize(("ndim", "plot_type"),
                         _generate_parameter())
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                               tolerance=default_tol, style=style_pytest_mpl)
def test_plot_two_cursors(mpl_cleanup, ndim, plot_type):
    s = _test_plot_two_cursors(ndim=ndim)
    if plot_type == "sig":
        return s._plot.signal_plot.figure
    else:
        return s._plot.navigator_plot.figure


@update_close_figure
def test_plot_nav2_sig1_two_cursors_close():
    return _test_plot_two_cursors(ndim=2)
