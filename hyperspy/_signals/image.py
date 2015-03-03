# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

from hyperspy.signal import Signal


class Image(Signal):

    """
    """
    _record_by = "image"

    def __init__(self, *args, **kw):
        super(Image, self).__init__(*args, **kw)
        self.axes_manager.set_signal_dimension(2)

    def to_spectrum(self):
        """Returns the image as a spectrum.

        See Also
        --------
        as_spectrum : a method for the same purpose with more options.
        signals.Image.to_spectrum : performs the inverse operation on images.

        """
        return self.as_spectrum(0 + 3j)

    def tomographic_reconstruction(self,
                                   algorithm='FBP',
                                   tilt_stages='auto',
                                   iteration=1,
                                   relaxation=0.15,
                                   **kwargs):
        """
        Reconstruct a 3D tomogram from a sinogram

        The siongram has x and y as signal axis and tilt as navigation axis

        Parameters
        ----------
        algorithm: {'FBP','SART'}
            FBP, filtered back projection
            SART, Simultaneous Algebraic Reconstruction Technique
        tilt_stages: list or 'auto'
            the angles of the sinogram. If 'auto', take the navigation axis
            value.
        iteration: int
            The numebr of iteration used for SART
        relaxation: float
            For SART: Relaxation parameter for the update step. A higher value
            can improve the convergence rate, but one runs the risk of
            instabilities. Values close to or higher than 1 are not
            recommended.

        Return
        ------
        The reconstruction as a 3D image

        Examples
        --------
        >>> tilt_series.change_dtype('float')
        >>> rec = tilt_series.tomographic_reconstruction()

        Notes
        -----
        See skimage.transform.iradon and skimage.transform.iradon_sart
        """
        from hyperspy._signals.spectrum import Spectrum
        sinogram = self.to_spectrum().data
        if tilt_stages == 'auto':
            tilt_stages = self.axes_manager[0].axis
        if algorithm == 'FBP':
            from skimage.transform import iradon
            rec = np.zeros([sinogram.shape[0], sinogram.shape[1],
                            sinogram.shape[1]])
            for i in range(sinogram.shape[0]):
                rec[i] = iradon(sinogram[i], theta=tilt_stages,
                                output_size=sinogram.shape[1], **kwargs)
        elif algorithm == 'SART':
            from skimage.transform import iradon_sart
            rec = np.zeros([sinogram.shape[0], sinogram.shape[1],
                            sinogram.shape[1]])
            for i in range(sinogram.shape[0]):
                rec[i] = iradon_sart(sinogram[i], theta=tilt_stages,
                                     **kwargs)
                for j in range(iteration - 1):
                    rec[i] = iradon_sart(sinogram[i], theta=tilt_stages,
                                         image=rec[i], **kwargs)

        rec = Spectrum(rec).as_image([2, 1])
        rec.axes_manager = self.axes_manager.deepcopy()
        rec.axes_manager[0].scale = rec.axes_manager[1].scale
        rec.axes_manager[0].offset = rec.axes_manager[1].offset
        rec.axes_manager[0].units = rec.axes_manager[1].units
        rec.axes_manager[0].name = 'z'
        rec.get_dimensions_from_data()
        return rec
