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
from matplotlib.testing.decorators import image_comparison

import hyperspy.api as hs


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


@image_comparison(baseline_images=['plot_multiple_images'],
                  extensions=['png'])
def test_plot_multiple_images():
    image = _generate_image_stack_signal()

    image.metadata.General.title = 'multi-dimensional Lena'
    hs.plot.plot_images(image, tight_layout=True)


@image_comparison(baseline_images=['plot_image_scalebar'],
                  extensions=['png'])
def test_plot_scalebar_image():
    image = hs.signals.Signal2D(
        np.arange(
            5 * 10,
            dtype=np.uint8).reshape(
            (5,
             10)))
    image.axes_manager[0].scale = 0.5
    image.axes_manager[1].scale = 0.5
    image.axes_manager[0].units = 'nm'
    image.axes_manager[1].units = 'nm'
    image.metadata.General.title = 'Image with scale bar'
    image.plot()


@image_comparison(baseline_images=['plot_image_scalebar_not_square_pixel'],
                  extensions=['png'])
def test_plot_image_scalebar_not_square_pixel():
    image = hs.signals.Signal2D(
        np.arange(
            5 * 10,
            dtype=np.uint8).reshape(
            (5,
             10)))
    image.axes_manager[0].scale = 0.25
    image.axes_manager[1].scale = 0.5
    image.axes_manager[0].units = 'nm'
    image.axes_manager[1].units = 'um'
    image.metadata.General.title = 'The scale bar can not be displayed'\
        ' because the pixel is not square'
    image.plot()


@image_comparison(baseline_images=['plot_multiple_images_label'],
                  extensions=['png'])
def test_plot_multiple_images_label():
    image = _generate_image_stack_signal()

    image.metadata.General.title = 'multi-dimensional Lena'
    hs.plot.plot_images(image, suptitle='Custom figure title',
                        label=['Signal2D 1', 'Signal2D 2', 'Signal2D 3',
                               'Signal2D 4', 'Signal2D 5', 'Signal2D 6'],
                        axes_decor=None, tight_layout=True)


@image_comparison(baseline_images=['plot_multiple_images_list'],
                  extensions=['png'])
def test_plot_multiple_images_list():
    # load red channel of raccoon as an image
    image0 = hs.signals.Signal2D(scipy.misc.face()[:, :, 0])

    image0.metadata.General.title = 'Rocky Raccoon - R'
    axes0 = image0.axes_manager
    axes0[0].name = "x"
    axes0[1].name = "y"
    axes0[0].units = "mm"
    axes0[1].units = "mm"

    # load lena into 2x3 hyperimage
    image1 = _generate_image_stack_signal()
    axes1 = image1.axes_manager
    axes1[2].name = "x"
    axes1[3].name = "y"
    axes1[2].units = "nm"
    axes1[3].units = "nm"

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
                        labelwrap=20)
