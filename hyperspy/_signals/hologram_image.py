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

import numpy as np
from skimage import draw
from scipy.fftpack import ifft2, fftshift, fft2
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from hyperspy.signals import Signal2D
from collections import OrderedDict
import logging
_logger = logging.getLogger(__name__)


class HologramImage(Signal2D):
    """Image subclass for holograms acquired via electron holography."""

    _signal_type = 'hologram'

    def reconstruct_phase(self, reference=None, rec_param=None, show_phase=False,
                          fresnel_fringe_filter=False, verbose=False):
        """Reconstruct holography data

        Parameters
        ----------
        reference : ndarray, :class:`~hyperspy.signals.Signal2D
            Vacuum reference hologram.
        rec_param : tuple
            Reconstruction parameters in sequence (SBrect(x0, y0, x1, y1), SB size)
        show_phase : boolean
            set True to plot phase after the reconstruction
        fresnel_fringe_filter : boolean
            set True to apply an automatic filtering of biprisms fresnel fringe
        verbose : boolean
            set True to see details of the reconstruction (i.e. SB selection)

        Returns
        -------
        wave : :class:`~hyperspy.signals.WaveImage
            Reconstructed electron wave. By default object wave is devided by reference wave

        Notes
        -----
        The reconstruction parameters assigned interactively if rec_param is not given.
        Use wave.rec_param to extract reconstruction parameters, which can be used for batch processing

        See Also
        --------

        """

        holo_size = self.data.shape
        holo_data = self.data

        # Parsing reference input:
        if reference is None:  # to reconstruct hologram WITHOUT a reference
            eh_hw_fft = fftshift(fft2(holo_data))
        elif isinstance(reference, Signal2D):
            # to reconstruct hologram WITH a reference
            ref_data = reference.data
            eh_hw_fft = fftshift((fft2(ref_data)))
        else:
            ref_data = reference
            eh_hw_fft = fftshift((fft2(ref_data)))

        # Preparing reconstruction parameters:
        if rec_param is None:
            eh_hw_fft_half = eh_hw_fft[0:np.int(holo_size[0] / 2.02), :]  # selects half of the fft excluding
            # fraction of points in the middle
            sb_pos = tuple(np.unravel_index(eh_hw_fft_half.argmax(), eh_hw_fft_half.shape))  # Center of selected
            #  Side Band

            # Default SB size is 1/2 of the distance to main band
            sb_size = int(norm(np.subtract(sb_pos, (holo_size[0] / 2, holo_size[1] / 2)))) // 4 * 2  # To be sure of
            # even integer number, still questionable if it even number is needed...

            rec_param = (sb_pos[1] - 1, sb_pos[0] - 1, sb_pos[1] + 1, sb_pos[0] + 1, sb_size)

            _logger.info('Sideband size in pixels: {}'.format(sb_size))

            #
            # GUI based reconstruction, has to be redone using Hyperspy GUI:
            # f, ax = plt.subplots(1, 1)
            # ax.imshow(np.log(np.absolute(eh_hw_fft)),
            #           cmap=cm.binary_r)  # Magnification might be added;
            # # getting rectangular ROI
            # rect = RoiRect()
            # if hasattr(f.canvas.manager, 'window'):
            #     f.canvas.manager.window.raise_()
            # plt.waitforbuttonpress(100)
            # plt.waitforbuttonpress(5)
            # # Use this one in the case of usage of rect_roi obj:
            # arect = eh_hw_fft[rect.y0:rect.y1, rect.x0:rect.x1]
            # # Sideband position,find the max number and its [c,r]
            # sb_pos = np.unravel_index(arect.argmax(), arect.shape)  # Center of selected sideBand
            # # Use this one in the case of usage of rect_roi obj
            # sb_pos = [np.round(rect.y0) + sb_pos[0], np.round(rect.x0) + sb_pos[1]]
            # a = 1.0
            # sb_size = np.round(np.array([a / 3, a / 2, a]) *
            #                    (norm(np.subtract(sb_pos, [holo_size[0] / 2, holo_size[1] / 2])) * np.sqrt(2)))
            # # To be sure of even number, still questionable if it is needed:
            # sb_size = sb_size - np.mod(sb_size, 2)
            # print("Sideband range in pixels")
            # print("%d %d %d" % (sb_size[0], sb_size[1], sb_size[2]))
            # # Choose SideBand size
            # sb_size = input("Choose Sideband size  pixel = ")
            # sb_size = sb_size - np.mod(sb_size, 2)  # to be sure of even number...
            # print("Sideband Size in pixels")
            # print("%d" % sb_size)
            # plt.close(f)
            # rec_param = (rect.x0, rect.y0, rect.x1, rect.y1, sb_size)
        else:
            arect = eh_hw_fft[rec_param[1]:rec_param[3], rec_param[0]:rec_param[2]]
            sb_pos = tuple(np.unravel_index(arect.argmax(),
                                            arect.shape))  # Center of selected sideBand
            sb_pos = (rec_param[1] + sb_pos[1], rec_param[0] + sb_pos[0])
            sb_size = rec_param[4]

        if verbose:
            plt.figure()
            plt.imshow(np.log(np.abs(eh_hw_fft)))
            plt.hold(True)
            plt.plot(sb_pos[1], sb_pos[0], 'r+')

        # Reconstruction
        if fresnel_fringe_filter:
            fresnel_width = None
        else:
            fresnel_width = 0

        if ref_data is None:
            w_ref = 1
        else:
            w_ref = self._reconstruct(ref_data.data, sb_size, sb_pos, fresnel_width=fresnel_width)  # reference electron wave

        w_obj = self._reconstruct(holo_data, sb_size, sb_pos, fresnel_width=fresnel_width)  # object wave

        wave = w_obj / w_ref
        wave_image = self._deepcopy_with_new_data(wave)
        wave_image.set_signal_type('ComplexSignal2D')  # New signal is a complex image!
        rec_param_dict = OrderedDict([('sb_pos_x0', rec_param[0]), ('sb_pos_y0', rec_param[1]),
                                      ('sb_pos_x1', rec_param[2]), ('sb_pos_y1', rec_param[3]),
                                      ('sb_size', rec_param[4])])

        wave_image.metadata.Signal.add_node('holo_rec_param')
        wave_image.metadata.Signal.holo_rec_param.add_dictionary(rec_param_dict)
        wave_image.axes_manager[0].scale = wave_image.axes_manager[0].scale*holo_size[0]/wave.shape[0]
        wave_image.axes_manager[1].scale = wave_image.axes_manager[1].scale*holo_size[1]/wave.shape[1]
        if show_phase:
            phase = np.angle(wave)
            f, ax = plt.subplots(1, 1)
            ax.imshow(phase, cmap=cm.binary_r)
            # f.canvas.manager.window.raise_()

        return wave_image


    @staticmethod
    def _reconstruct(holo_data, sb_size, sb_pos, fresnel_ratio=0.3, fresnel_width=6):
        """Core function for holographic reconstruction performing following steps:

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
        sb_pos : tuple
            Vector of two elements with sideband coordinates [y,x]
        fresnel_ratio : float
            The ratio of Fresnel filter with respect to the sideband size
        fresnel_width : int
            Width of fresnel fringe filter in px, set to 0 to disable filtering

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

        """
        # TODO: Parsing of the input has to be redone
        # TODO: Add smoothing of Fresnel filter
        # TODO: Add other smoothing options?

        image_size = sb_size*2
        holo_size = holo_data.shape
        holo_data = np.float64(holo_data)

        h_hw_fft = fftshift(fft2(holo_data))  # <---- NO Hanning filtering

        sb_roi = h_hw_fft[(sb_pos[0]-image_size//2):(sb_pos[0]+image_size//2),
                 (sb_pos[1]-image_size//2):(sb_pos[1]+image_size//2)]

        (sb_ny, sb_nx) = sb_roi.shape
        sb_l = min(sb_ny / 2, sb_nx / 2)
        cen_yx = [sb_ny / 2, sb_nx / 2]

        # Circular Aperture
        cgrid_y = np.arange(-sb_ny / 2, sb_ny / 2, 1)
        cgrid_x = np.arange(-sb_nx / 2, sb_nx / 2, 1)
        (sb_xx, sb_yy) = np.meshgrid(cgrid_x, cgrid_y)
        sb_r = np.sqrt(sb_xx ** 2 + sb_yy ** 2)
        c_mask = np.zeros((sb_nx, sb_ny))  # Original: cMask=zeros(SB_Ny,SB_Nx);

        c_mask[sb_r < sb_size] = 1

        # Fresnel Mask
        ang = np.arctan2((holo_size[0] / 2 - sb_pos[0]), (holo_size[1] / 2 - sb_pos[1]))  # [-pi pi]

        p_one = np.round([fresnel_ratio * sb_l * np.sin(ang), fresnel_ratio * sb_l * np.cos(ang)])
        p_two = np.round([sb_l * np.sin(ang), sb_l * np.cos(ang)])

        ang_one = (ang - np.pi / 2) % (2 * np.pi )
        ang_two = (ang + np.pi / 2) % (2 * np.pi )

        r = fresnel_width / 2
        aa = np.round([p_one[0] + r * np.sin(ang_one), p_one[1] + r * np.cos(ang_one)]) + cen_yx
        bb = np.round([p_one[0] + r * np.sin(ang_two), p_one[1] + r * np.cos(ang_two)]) + cen_yx
        cc = np.round([p_two[0] + r * np.sin(ang_two), p_two[1] + r * np.cos(ang_two)]) + cen_yx
        dd = np.round([p_two[0] + r * np.sin(ang_one), p_two[1] + r * np.cos(ang_one)]) + cen_yx

        abcd = np.array([aa, bb, cc, dd])
        f_mask = _poly_to_mask(abcd[:, 1], abcd[:, 0], sb_roi.shape)

        sinc_k = 5.0  # Sink times SBsize
        w_one = np.sinc(
            np.linspace(-sb_ny/2, sb_ny/2, image_size) * np.pi / (sinc_k * sb_size))
        w_one = w_one.reshape(image_size, 1)
        w_two = np.sinc(
            np.linspace(-sb_nx/2, sb_nx/2, image_size) * np.pi / (sinc_k * sb_size))
        window = w_one.dot(w_two.reshape(1, image_size))

        # IFFT
        wav = ifft2(fftshift(sb_roi * c_mask * np.logical_not(f_mask) * window))
        return wav


def _poly_to_mask(vertex_row_coords, vertex_col_coords, shape):
    """
    Creates a polygon mask
    """
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask
