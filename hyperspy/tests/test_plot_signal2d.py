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
from matplotlib.testing.decorators import image_comparison

import hyperspy.api as hs
from hyperspy.misc.test_utils import get_matplotlib_version_label
from hyperspy.drawing.utils import plot_RGB_map

mplv = get_matplotlib_version_label()
scalebar_color = 'blue'

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


@image_comparison(baseline_images=['%s_rgb_image' % mplv],
                  extensions=['png'])    
def test_rgb_image():
    w = 20
    ch1 = hs.signals.Signal2D(np.arange(w * w).reshape(w, w))
    ch1.axes_manager = _set_signal_axes(ch1.axes_manager)
    ch2 = hs.signals.Signal2D(np.arange(w * w).reshape(w, w).T)
    ch2.axes_manager = _set_signal_axes(ch2.axes_manager)
    plot_RGB_map([ch1, ch2])
    

""" Navigation 0, Signal 2 """


def _setup_nav0_sig2():
    width = 20
    data = np.arange(width * width).reshape((width, width))
    s = hs.signals.Signal2D(data)
    scale = 1E9
    offset = -scale * width / 2
    s.axes_manager = _set_signal_axes(s.axes_manager, units='1/m',
                                      scale=scale, offset=offset)
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    return s

@image_comparison(baseline_images=['%s_nav0_signal2_1sig' % mplv],
                  extensions=['png'])
def test_plot_nav0_sig2():
    s = _setup_nav0_sig2()
    s.metadata.General.title = '1: Nav 0, Sig 2, with scalebar and no axis ticks'
    s.plot(scalebar_color=scalebar_color)


@image_comparison(baseline_images=['%s_nav0_signal2_2sig' % mplv],
                  extensions=['png'])
def test_plot_nav0_sig2_axes_ticks():
    s = _setup_nav0_sig2()
    s.metadata.General.title = '2: Nav 0, Sig 2 with axes_ticks=True'
    s.plot(scalebar_color=scalebar_color, axes_ticks=True)


@image_comparison(baseline_images=['%s_nav0_signal2_3sig' % mplv],
                  extensions=['png'])
def test_plot_nav0_sig2_no_scalebar():
    s = _setup_nav0_sig2()
    s.metadata.General.title = '3: Nav 0, Sig 2, without scalebar'
    s.plot(scalebar=False, scalebar_color=scalebar_color)

        
@image_comparison(baseline_images=['%s_nav0_signal2_4nav' % mplv,
                                   '%s_nav0_signal2_4nav' % mplv],
                  extensions=['png'])
def test_plot_nav0_sig2_different_signal_axes_scale():
    s = _setup_nav0_sig2()
    s.metadata.General.title = '4: Nav 0, Sig 2, without scalebar '\
        '(different axes scale)'
    s.axes_manager[0].scale = 5E9
    s.axes_manager[0].name = t.Undefined
    s.axes_manager[1].name = t.Undefined
    s.plot(scalebar_color=scalebar_color)


""" Navigation 2, Signal 2 """


def _setup_nav2_sig2():
    data = np.arange(5 * 7 * 10 * 20).reshape((5, 7, 10, 20))
    s = hs.signals.Signal2D(data)
    s.axes_manager = _set_signal_axes(s.axes_manager, name='Energy',
                                      units='1/m', scale=500.0, offset=0.0)
    s.axes_manager = _set_navigation_axes(s.axes_manager, name='',
                                          units='m', scale=1E-6, offset=5E-6)
    return s

    
@image_comparison(baseline_images=['%s_nav2_signal2_1nav' % mplv,
                                   '%s_nav2_signal2_1sig' % mplv],
                  extensions=['png'])
def test_plot_nav2_sig2():
    s = _setup_nav2_sig2()
    s.metadata.General.title = '1: Nav 2, Sig 2'
    s.plot(scalebar_color=scalebar_color)

    
@image_comparison(baseline_images=['%s_nav2_signal2_2nav' % mplv,
                                   '%s_nav2_signal2_2sig' % mplv],
                  extensions=['png'])
def test_plot_nav2_sig2_no_scalebar():
    s = _setup_nav2_sig2()
    s.metadata.General.title = '1: Nav 2, Sig 2, without scalebar'
    s.plot(scalebar=False, scalebar_color=scalebar_color)
    

@image_comparison(baseline_images=['%s_plot_multiple_images' % mplv],
                  extensions=['png'])
def test_plot_multiple_images():
    image = _generate_image_stack_signal()

    image.metadata.General.title = 'multi-dimensional Lena'
    hs.plot.plot_images(image, tight_layout=True,
                        scalebar_color=scalebar_color)


@image_comparison(baseline_images=['%s_plot_multiple_images_label' % mplv],
                  extensions=['png'])
def test_plot_multiple_images_label():
    image = _generate_image_stack_signal()

    image.metadata.General.title = 'multi-dimensional Lena'
    hs.plot.plot_images(image, suptitle='Custom figure title',
                        label=['Signal2D 1', 'Signal2D 2', 'Signal2D 3',
                               'Signal2D 4', 'Signal2D 5', 'Signal2D 6'],
                        axes_decor=None, tight_layout=True)


@image_comparison(baseline_images=['%s_plot_multiple_images_list' % mplv],
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
