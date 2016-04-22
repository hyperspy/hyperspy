# -*- coding: utf-8 -*-
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


from hyperspy.signals import Image


class PhaseImage(Image):
    def plot(self,
             colorbar=True,
             scalebar=True,
             scalebar_color="white",
             axes_ticks=None,
             auto_contrast=True,
             saturated_pixels=0,
             vmin=None,
             vmax=None,
             no_nans=False,
             centre_colormap="auto",
             **kwargs
             ):
        """Plot image.

        For multidimensional datasets an optional figure,
        the "navigator", with a cursor to navigate that data is
        raised. In any case it is possible to navigate the data using
        the sliders. Currently only signals with signal_dimension equal to
        0, 1 and 2 can be plotted.

        Parameters
        ----------
        navigator : {"auto", None, "slider", "spectrum", Signal}
            If "auto", if navigation_dimension > 0, a navigator is
            provided to explore the data.
            If navigation_dimension is 1 and the signal is an image
            the navigator is a spectrum obtained by integrating
            over the signal axes (the image).
            If navigation_dimension is 1 and the signal is a spectrum
            the navigator is an image obtained by stacking horizontally
            all the spectra in the dataset.
            If navigation_dimension is > 1, the navigator is an image
            obtained by integrating the data over the signal axes.
            Additionaly, if navigation_dimension > 2 a window
            with one slider per axis is raised to navigate the data.
            For example,
            if the dataset consists of 3 navigation axes X, Y, Z and one
            signal axis, E, the default navigator will be an image
            obtained by integrating the data over E at the current Z
            index and a window with sliders for the X, Y and Z axes
            will be raised. Notice that changing the Z-axis index
            changes the navigator in this case.
            If "slider" and the navigation dimension > 0 a window
            with one slider per axis is raised to navigate the data.
            If "spectrum" and navigation_dimension > 0 the navigator
            is always a spectrum obtained by integrating the data
            over all other axes.
            If None, no navigator will be provided.
            Alternatively a Signal instance can be provided. The signal
            dimension must be 1 (for a spectrum navigator) or 2 (for a
            image navigator) and navigation_shape must be 0 (for a static
            navigator) or navigation_shape + signal_shape must be equal
            to the navigator_shape of the current object (for a dynamic
            navigator).
            If the signal dtype is RGB or RGBA this parameters has no
            effect and is always "slider".
        axes_manager : {None, axes_manager}
            If None `axes_manager` is used.
        colorbar : bool, optional
             If true, a colorbar is plotted for non-RGB images.
        scalebar : bool, optional
            If True and the units and scale of the x and y axes are the same a
            scale bar is plotted.
        scalebar_color : str, optional
            A valid MPL color string; will be used as the scalebar color.
        axes_ticks : {None, bool}, optional
            If True, plot the axes ticks. If None axes_ticks are only
            plotted when the scale bar is not plotted. If False the axes ticks
            are never plotted.
        auto_contrast : bool, optional
            If True, the contrast is stretched for each image using the
            `saturated_pixels` value. Default True.
        saturated_pixels: scalar
            The percentage of pixels that are left out of the bounds.
            For example, the low and high bounds of a value of 1 are the 0.5%
            and 99.5% percentiles. It must be in the [0, 100] range.
        vmin, vmax : scalar, optional
            `vmin` and `vmax` are used to normalize luminance data. If at
            least one of them is given `auto_contrast` is set to False and any
            missing values are calculated automatically.
        no_nans : bool, optional
            If True, set nans to zero for plotting.
        centre_colormap : {"auto", True, False}
            If True the centre of the color scheme is set to zero. This is
            specially useful when using diverging color schemes. If "auto"
            (default), diverging color schemes are automatically centred.
        **kwargs, optional
            Additional key word arguments passed to matplotlib.imshow()

        """
        super(Image, self).plot(
            colorbar=colorbar,
            scalebar=scalebar,
            scalebar_color=scalebar_color,
            axes_ticks=axes_ticks,
            auto_contrast=auto_contrast,
            saturated_pixels=saturated_pixels,
            vmin=vmin,
            vmax=vmax,
            no_nans=no_nans,
            centre_colormap=centre_colormap,
            **kwargs
        )


'''
Holography module

Dependencies
------------
ercpy.utils

'''
import numpy as np
from numpy.linalg import norm
from scipy.fftpack import ifft2, fftshift, fft2
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from PIL import Image
# import time
import utils

# from matplotlib.patches import Rectangle
# import hyperspy as hs

# PHASE UNWRAPPING #
####################
# 1) https://github.com/geggo/phase-unwrap
#   Algorithm based on:
#  M. A. Herraez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat,
#  "Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path",
#   Applied Optics, Vol. 41, Issue 35, pp. 7437-7444 (2002)
from skimage.restoration import unwrap_phase as unwrap

#   usage: phase_unw = holo.unwrap(phase)
# 2) Good C code with Python front end
#   http://www.cio.mx/~jestrada/phase_unwrapping2.html
#   https://github.com/trago/fringeproc
# x) Other sources:
#   http://nova.stanford.edu/sar_group/snaphu/
#   http://roipac.org/cgi-bin/moin.cgi/PhaseFiltUnwrap
#   http://community.dur.ac.uk/n.a.bharmal/phaseUnwrappingWithPython.html
#   http://www.lx.it.pt/~bioucas/code.htm
####################


__all__ = ['holo_reconstruct', 'unwrap']


def holo_reconstruct(holo_data, ref_data=None, rec_param=None, show_phase=False, **kwargs):
    '''Reconstruct holography data

    Parameters
    ----------
    holo_data : ndarray
        The object hologram array.
    ref_data : ndarray
        The refernce hologram array.
    rec_param : tuple
        Reconstruction parameters in sequence (SBrect(x0, y0, x1, y1), SB size)
    show_phase : boolean
        set True to plot phase after the reconstruction

    Returns
    -------
    wave : ndarray
        Reconstructed electron wave. By default object wave is devided by reference wave
    phase : ndarray
        Wrapped electron phase
    amp : ndarray
        Amplitude of the wave
    rec_param : tuple
        see the description in Parameters

    Notes
    -----
    The reconstruction parameters assigned interactively if rec_param is not givven

    See Also
    --------
    reconstruct_holo

    '''

    (sx, sy) = holo_data.shape
    eh_hw_fft = fftshift(fft2(holo_data))
    if rec_param is None:
        f, ax = plt.subplots(1, 1)
        ax.imshow(np.log(np.absolute(eh_hw_fft)), cmap=cm.binary_r)  # Magnification might be added;

        # getting rectangular ROI
        rect = utils.RoiRect()
        if hasattr(f.canvas.manager, 'window'):  f.canvas.manager.window.raise_()

        plt.waitforbuttonpress(100)
        plt.waitforbuttonpress(5)
        arect = eh_hw_fft[rect.y0:rect.y1,
                rect.x0:rect.x1]  # <----- use this one in the case of usage of rect_roi obj
        # Sideband position,find the max number and its [c,r]
        yR, xC = np.unravel_index(arect.argmax(), arect.shape)  # center of the selected sideBand
        sb_pos = [np.round(rect.y0) + yR, np.round(
            rect.x0) + xC]  # <----- use this one in the case of usage of rect_roi obj
        # Sideband position,find the max number and its [c,r]
        yR, xC = np.unravel_index(arect.argmax(), arect.shape)  # center of the selected sideBand
        sb_pos = [np.round(rect.y0) + yR, np.round(
            rect.x0) + xC]  # <----- use this one in the case of usage of rect_roi obj
        a = 1.0
        sb_size = np.round(np.array([a / 3, a / 2, a]) * (
        norm(np.subtract(sb_pos, [sx / 2, sy / 2])) * np.sqrt(2)))
        sb_size = sb_size - np.mod(sb_size,
                                   2)  # to be sure of even number, still questionable if it is needed
        print "Sideband range in pixels"
        print "%d %d %d" % (sb_size[0], sb_size[1], sb_size[2])
        # Choose SideBand size
        sb_size = input("Choose Sideband size  pixel = ")
        sb_size = sb_size - np.mod(sb_size, 2)  # to be sure of even number...
        print "Sideband Size in pixels"
        print "%d" % sb_size
        plt.close(f)
        rec_param = (rect.x0, rect.y0, rect.x1, rect.y1, sb_size)
    else:
        arect = eh_hw_fft[rec_param[1]:rec_param[3], rec_param[0]:rec_param[2]]
        # Sideband position,find the max number and its [c,r]
        yR, xC = np.unravel_index(arect.argmax(), arect.shape)  # center of the selected sideBand
        sb_pos = [np.round(rec_param[1]) + yR, np.round(
            rec_param[0]) + xC]  # <----- use this one in the case of usage of rect_roi obj
        sb_size = rec_param[4]

    # Reconstruction
    if ref_data is None:
        w_ref = 1
    else:
        w_ref = _reconstruct(ref_data, sb_size, sb_pos, [], [])  # reference electron wave

    w_obj = _reconstruct(holo_data, sb_size, sb_pos, [], [])  # object wave

    wave = w_obj / w_ref
    phase = np.angle(wave)
    amp = np.absolute(wave)

    if show_phase:
        f, ax = plt.subplots(1, 1)
        ax.imshow(phase, cmap=cm.binary_r)
        f.canvas.manager.window.raise_()

    return (wave, phase, amp, rec_param)


def _reconstruct(holo_data, sb_size, sb_pos, fresnel_ratio, fresnel_width):
    '''Core function for holographic reconstruction performing following steps:

    * 2D FFT without apodisation;

    * Cutting out sideband;

    * Centering sideband;

    * Applying round window;

    * Applying sinc filter;

    * Applying automatic filtering of Fresnel fringe (Fresnel filtering);

    * Inverse FFT.

    Parameters
    ----------
    holo_data : array_like
        Holographic data array
    sb_size : int
        Size of the sideband filter in px
    sb_pos : nparray (N=1)
        Vector of two elements with sideband coordinates [y,x]
    fresnel_ratio : float
        The ratio of Fresnel filter with respect to the sideband size
    fresnel_width : int
        Width of frsnel filter in px

    Returns
    -------
        wav : nparray
            Reconstructed electron wave

    Notes
    -----

    Disabeling of Fresnel filter is not implemented

    See Also
    --------

    reconstruct

    '''
    # TODO: Parsing of the input has to be redone
    # TODO: Add disabling of Fresnel filtering
    # TODO: Add smoothing of Fresnel filter
    # TODO: Add smoothing options?

    # Parse input
    if not fresnel_ratio:  # fresnenl_ratio is empty or 0
        fresnel_ratio = 0.3

    if not fresnel_width:  # fresnel_width is emty or 0
        fresnel_width = 6

    (sx, sy) = holo_data.shape
    holo_data = np.float64(holo_data);

    h_hw_fft = fftshift(fft2(holo_data))  # <---- NO Hanning filtering

    sb_roi = h_hw_fft[sb_pos[0] - sb_size / 2:sb_pos[0] + sb_size / 2,
             sb_pos[1] - sb_size / 2:sb_pos[1] + sb_size / 2]

    (sb_ny, sb_nx) = sb_roi.shape
    sb_l = min(sb_ny / 2, sb_nx / 2)
    cen_yx = [sb_ny / 2, sb_nx / 2]

    # Circular Aperture
    cgrid_y = np.arange(-sb_ny / 2, sb_ny / 2, 1)
    cgrid_x = np.arange(-sb_nx / 2, sb_nx / 2, 1)
    (sb_xx, sb_yy) = np.meshgrid(cgrid_x, cgrid_y)
    sb_r = np.sqrt(sb_xx ** 2 + sb_yy ** 2)
    c_mask = np.zeros((sb_nx, sb_ny))  # Original: cMask=zeros(SB_Ny,SB_Nx);

    c_mask[sb_r < sb_l] = 1

    # Fresnel Mask
    ang = np.arctan2((sx / 2 - sb_pos[0]), (sy / 2 - sb_pos[1]));  # [-pi pi]

    p_one = np.round([fresnel_ratio * sb_l * np.sin(ang), fresnel_ratio * sb_l * np.cos(ang)])
    p_two = np.round([sb_l * np.sin(ang), sb_l * np.cos(ang)])

    # ang_three = utils.wrap_to_pi(np.arctan2(p_two[0]-p_one[0], p_two[1]-p_one[1]))
    ang_one = utils.wrap(ang - np.pi / 2)
    ang_two = utils.wrap(ang + np.pi / 2)

    r = fresnel_width / 2
    aa = np.round([p_one[0] + r * np.sin(ang_one), p_one[1] + r * np.cos(ang_one)]) + cen_yx
    bb = np.round([p_one[0] + r * np.sin(ang_two), p_one[1] + r * np.cos(ang_two)]) + cen_yx
    cc = np.round([p_two[0] + r * np.sin(ang_two), p_two[1] + r * np.cos(ang_two)]) + cen_yx
    dd = np.round([p_two[0] + r * np.sin(ang_one), p_two[1] + r * np.cos(ang_one)]) + cen_yx

    abcd = np.array([aa, bb, cc, dd])
    f_mask = utils.poly_to_mask(abcd[:, 1], abcd[:, 0], sb_roi.shape)

    sinc_k = 5.0;  # Sink times SBsize
    w_one = np.sinc(np.linspace(-sb_size / 2, sb_size / 2, sb_size) * np.pi / (sinc_k * sb_size))
    w_one = w_one.reshape((sb_size, 1))
    w_two = w_one.dot(w_one.reshape((1, sb_size)))

    # IFFT
    wav = ifft2(fftshift(sb_roi * c_mask * np.logical_not(f_mask) * w_two))
    return wav
