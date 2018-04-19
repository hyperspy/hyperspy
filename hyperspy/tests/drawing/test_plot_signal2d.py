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
import scipy.ndimage
import traits.api as t
import pytest
import matplotlib.pyplot as plt

import hyperspy.api as hs
from hyperspy.drawing.utils import plot_RGB_map
from hyperspy.tests.drawing.test_plot_signal import _TestPlot


scalebar_color = 'blue'
default_tol = 2.0
baseline_dir = 'plot_signal2d'
style_pytest_mpl = 'default'


def _generate_image_stack_signal():
    image = hs.signals.Signal2D(np.random.random((2, 3, 512, 512)))
    for i in range(2):
        for j in range(3):
            image.data[i, j, :] = scipy.misc.ascent() * (i + 0.5 + j)
    axes = image.axes_manager
    axes[2].name = "x"
    axes[3].name = "y"
    axes[2].units = "nm"
    axes[3].units = "nm"

    return image


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


@pytest.mark.parametrize("normalization", ['single', 'global'])
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_rgb_image(mpl_cleanup, normalization):
    w = 20
    data = np.arange(1, w * w + 1).reshape(w, w)
    ch1 = hs.signals.Signal2D(data)
    ch1.axes_manager = _set_signal_axes(ch1.axes_manager)
    ch2 = hs.signals.Signal2D(data.T * 2)
    ch2.axes_manager = _set_signal_axes(ch2.axes_manager)
    plot_RGB_map([ch1, ch2], normalization=normalization)
    return plt.gcf()


def _generate_parameter():
    parameters = []
    for scalebar in [True, False]:
        for colorbar in [True, False]:
            for axes_ticks in [True, False]:
                for centre_colormap in [True, False]:
                    parameters.append([scalebar, colorbar, axes_ticks,
                                       centre_colormap])
    return parameters


@pytest.mark.parametrize(("scalebar", "colorbar", "axes_ticks",
                          "centre_colormap"),
                         _generate_parameter())
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot(mpl_cleanup, scalebar, colorbar, axes_ticks, centre_colormap):
    test_plot = _TestPlot(ndim=0, sdim=2)
    test_plot.signal.plot(scalebar=scalebar,
                          colorbar=colorbar,
                          axes_ticks=axes_ticks,
                          centre_colormap=centre_colormap)
    return test_plot.signal._plot.signal_plot.figure


def _generate_parameter_plot_images():
    # There are 9 images in total
    vmin, vmax = [None] * 9, [None] * 9
    vmin[1], vmax[2] = 30, 200
    return vmin, vmax


@pytest.mark.parametrize(("vmin", "vmax"), (_generate_parameter_plot_images(),
                                            (None, None)))
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_multiple_images_list(mpl_cleanup, vmin, vmax):
    # load red channel of raccoon as an image
    image0 = hs.signals.Signal2D(scipy.misc.face()[:, :, 0])
    image0.metadata.General.title = 'Rocky Raccoon - R'
    axes0 = image0.axes_manager
    axes0[0].name = "x"
    axes0[1].name = "y"
    axes0[0].units = "mm"
    axes0[1].units = "mm"

    # load ascent into 2x3 hyperimage
    image1 = _generate_image_stack_signal()

    # load green channel of raccoon as an image
    image2 = hs.signals.Signal2D(scipy.misc.face()[:, :, 1])
    image2.metadata.General.title = 'Rocky Raccoon - G'
    axes2 = image2.axes_manager
    axes2[0].name = "x"
    axes2[1].name = "y"
    axes2[0].units = "mm"
    axes2[1].units = "mm"

    # load rgb imimagesage
    rgb = hs.signals.Signal1D(scipy.misc.face())
    rgb.change_dtype("rgb8")
    rgb.metadata.General.title = 'RGB'
    axesRGB = rgb.axes_manager
    axesRGB[0].name = "x"
    axesRGB[1].name = "y"
    axesRGB[0].units = "nm"
    axesRGB[1].units = "nm"

    hs.plot.plot_images([image0, image1, image2, rgb], tight_layout=True,
                        # colorbar='single',
                        labelwrap=20, vmin=vmin, vmax=vmax)
    return plt.gcf()


def test_plot_images_single_image(mpl_cleanup):
    image0 = hs.signals.Signal2D(np.arange(100).reshape(10, 10))
    image0.isig[5, 5] = 200
    image0.metadata.General.title = 'This is the title from the metadata'
    ax = hs.plot.plot_images(image0, saturated_pixels=0.1)
    return ax[0].figure


@pytest.mark.parametrize("saturated_pixels", [5.0, [0.0, 20.0, 40.0],
                                              [10.0, 20.0], [10.0, None, 20.0]])
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_saturated_pixels(mpl_cleanup, saturated_pixels):
    image0 = hs.signals.Signal2D(np.arange(100).reshape(10, 10))
    image0.isig[5, 5] = 200
    image0.metadata.General.title = 'This is the title from the metadata'
    ax = hs.plot.plot_images([image0, image0, image0],
                             saturated_pixels=saturated_pixels,
                             axes_decor='off')
    return ax[0].figure


@pytest.mark.parametrize("colorbar", ['single', 'multi', None])
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_colorbar(mpl_cleanup, colorbar):
    image0 = hs.signals.Signal2D(np.arange(100).reshape(10, 10))
    image0.isig[5, 5] = 200
    image0.metadata.General.title = 'This is the title from the metadata'
    ax = hs.plot.plot_images([image0, image0], colorbar=colorbar,
                             vmin=[0, 10], vmax=[120, None],
                             axes_decor='ticks')
    return ax[0].figure


def test_plot_images_signal1D():
    image0 = hs.signals.Signal1D(np.arange(100).reshape(10, 10))
    with pytest.raises(ValueError):
        hs.plot.plot_images([image0, image0])


def test_plot_images_not_signal():
    data = np.arange(100).reshape(10, 10)
    with pytest.raises(ValueError):
        hs.plot.plot_images([data, data])

    with pytest.raises(ValueError):
        hs.plot.plot_images(data)

    with pytest.raises(ValueError):
        hs.plot.plot_images('not a list of signal')
