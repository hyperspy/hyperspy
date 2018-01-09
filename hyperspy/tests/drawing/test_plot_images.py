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

from hyperspy.drawing.utils import make_cmap
import hyperspy.api as hs

import scipy

default_tol = 2.0
baseline_dir = 'plot_images'
style_pytest_mpl = 'default'


class _TestIteratedSignal:

    def __init__(self):
        s = hs.signals.Signal2D([scipy.misc.ascent()] * 6)
        angles = hs.signals.BaseSignal(range(00, 60, 10))
        s.map(scipy.ndimage.rotate, angle=angles.T, reshape=False)
        s.data = np.clip(s.data, 0, 255)  # prevent values outside
                                          # of integer range
        title = 'Ascent'

        s.axes_manager = self._set_signal_axes(s.axes_manager,
                                               name='spatial',
                                               units='nm', scale=1,
                                               offset=0.0)
        s.axes_manager = self._set_navigation_axes(s.axes_manager,
                                                   name='index',
                                                   units='images',
                                                   scale=1, offset=0)
        s.metadata.General.title = title

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


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_default(mpl_cleanup):
    test_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_plot.signal)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_list(mpl_cleanup):
    test_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_plot.signal,
                        axes_decor='off',
                        cmap=['viridis', 'gray'])
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_list_w_diverging(mpl_cleanup):
    test_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_plot.signal,
                        axes_decor='off',
                        cmap=['viridis', 'gray', 'RdBu_r'])
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_mpl_colors(mpl_cleanup):
    test_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_plot.signal,
                        axes_decor='off',
                        cmap='mpl_colors')
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_one_string(mpl_cleanup):
    test_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_plot.signal,
                        axes_decor='off',
                        cmap='RdBu_r')
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_make_cmap_bittrue(mpl_cleanup):
    test_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_plot.signal,
                        axes_decor='off',
                        cmap=make_cmap([(255, 255, 255),
                                        '#F5B0CB',
                                        (220, 106, 207),
                                        '#745C97',
                                        (57, 55, 91)],
                                       bit=True,
                                       name='test_cmap',
                                       register=True))
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_make_cmap_bitfalse(mpl_cleanup):
    test_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_plot.signal,
                        axes_decor='off',
                        cmap=make_cmap([(1, 1, 1),
                                        '#F5B0CB',
                                        (0.86, 0.42, 0.81),
                                        '#745C97',
                                        (0.22, 0.22, 0.36)],
                                       bit=False,
                                       name='test_cmap',
                                       register=True))
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_multi_signal(mpl_cleanup):
    test_plot1 = _TestIteratedSignal()

    test_plot2 = _TestIteratedSignal()
    test_plot2.signal *= 2  # change scale of second signal
    test_plot2.signal = test_plot2.signal.inav[::-1]
    test_plot2.signal.metadata.General.title = 'Descent'

    hs.plot.plot_images([test_plot1.signal,
                         test_plot2.signal],
                        axes_decor='off',
                        per_row=4,
                        cmap='mpl_colors')
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_multi_w_rgb(mpl_cleanup):
    test_plot1 = _TestIteratedSignal()

    test_plot2 = _TestIteratedSignal()
    test_plot2.signal *= 2  # change scale of second signal
    test_plot2.signal.metadata.General.title = 'Ascent-2'

    rgb_sig = hs.signals.Signal1D(scipy.misc.face())
    rgb_sig.change_dtype('rgb8')
    rgb_sig.metadata.General.title = 'Racoon!'

    hs.plot.plot_images([test_plot1.signal,
                         test_plot2.signal,
                         rgb_sig],
                        axes_decor='off',
                        per_row=4,
                        cmap='mpl_colors')
    return plt.gcf()
