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
from hyperspy.drawing.utils import make_cmap

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


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_log_scale(mpl_cleanup):
    test_plot = _TestPlot(ndim=0, sdim=2)
    test_plot.signal += 1  # need to avoid zeros in log
    test_plot.signal.plot(norm='log')
    return test_plot.signal._plot.signal_plot.figure


@pytest.mark.parametrize("fft_shift", [True, False])
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_FFT(mpl_cleanup, fft_shift):
    s = hs.datasets.example_signals.object_hologram()
    s2 = s.isig[:128, :128].fft()
    s2.plot(fft_shift=fft_shift, axes_ticks=True, power_spectrum=True)
    return s2._plot.signal_plot.figure


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
                        labelwrap=20, vmin=vmin, vmax=vmax)
    return plt.gcf()


class _TestIteratedSignal:

    def __init__(self):
        s = hs.signals.Signal2D([scipy.misc.ascent()] * 6)
        angles = hs.signals.BaseSignal(range(00, 60, 10))
        s.map(scipy.ndimage.rotate, angle=angles.T, reshape=False)
        # prevent values outside of integer range
        s.data = np.clip(s.data, 0, 255)
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
    test_im_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_im_plot.signal)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_list(mpl_cleanup):
    test_im_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_im_plot.signal,
                        axes_decor='off',
                        cmap=['viridis', 'gray'])
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_list_w_diverging(mpl_cleanup):
    test_im_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_im_plot.signal,
                        axes_decor='off',
                        cmap=['viridis', 'gray', 'RdBu_r'])
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_mpl_colors(mpl_cleanup):
    test_im_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_im_plot.signal,
                        axes_decor='off',
                        cmap='mpl_colors')
    return plt.gcf()


def test_plot_images_cmap_mpl_colors_w_single_cbar():
    # This should give an error, so test for that
    test_im_plot = _TestIteratedSignal()
    with pytest.raises(ValueError) as val_error:
        hs.plot.plot_images(test_im_plot.signal,
                            axes_decor='off',
                            cmap='mpl_colors',
                            colorbar='single')
    assert str(val_error.value) == 'Cannot use a single colorbar with ' \
                                   'multiple colormaps. Please check for ' \
                                   'compatible arguments.'


def test_plot_images_bogus_cmap():
    # This should give an error, so test for that
    test_im_plot = _TestIteratedSignal()
    with pytest.raises(ValueError) as val_error:
        hs.plot.plot_images(test_im_plot.signal,
                            axes_decor='off',
                            cmap=3.14159265359,
                            colorbar=None)
    assert str(val_error.value) == 'The provided cmap value was not ' \
                                   'understood. Please check input values.'


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_one_string(mpl_cleanup):
    test_im_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_im_plot.signal,
                        axes_decor='off',
                        cmap='RdBu_r',
                        colorbar='single')
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_cmap_make_cmap_bittrue(mpl_cleanup):
    test_im_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_im_plot.signal,
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
    test_im_plot = _TestIteratedSignal()
    hs.plot.plot_images(test_im_plot.signal,
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


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_single_image(mpl_cleanup):
    image0 = hs.signals.Signal2D(np.arange(100).reshape(10, 10))
    image0.isig[5, 5] = 200
    image0.metadata.General.title = 'This is the title from the metadata'
    ax = hs.plot.plot_images(image0, saturated_pixels=0.1)
    return plt.gcf()


@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol, style=style_pytest_mpl)
def test_plot_images_single_image_stack(mpl_cleanup):
    image0 = hs.signals.Signal2D(np.arange(200).reshape(2, 10, 10))
    image0.isig[5, 5] = 200
    image0.metadata.General.title = 'This is the title from the metadata'
    ax = hs.plot.plot_images(image0, saturated_pixels=0.1)
    return plt.gcf()


def test_plot_images_multi_signal_w_axes_replot(mpl_cleanup):
    imdata = np.random.rand(3, 5, 5)
    imgs = hs.signals.Signal2D(imdata)
    img_list = [imgs, imgs.inav[:2], imgs.inav[0]]
    subplots = hs.plot.plot_images(img_list, axes_decor=None)
    f = plt.gcf()
    f.canvas.draw()
    f.canvas.flush_events()

    tests = []
    for axi in subplots:
        imi = axi.images[0].get_array()
        x, y = axi.transData.transform((2, 2))
        # Calling base class method because of backends
        plt.matplotlib.backends.backend_agg.FigureCanvasBase.button_press_event(
            f.canvas, x, y, 'left', True)
        fn = plt.gcf()
        tests.append(
            np.allclose(imi, fn.axes[0].images[0].get_array().data))
        plt.close(fn)
    assert np.alltrue(tests)
    return f


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
