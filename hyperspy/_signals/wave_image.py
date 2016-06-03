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

from skimage.restoration import unwrap_phase as unwrap

from hyperspy.signals import Image


class WaveImage(Image):
    """
    """

    _signal_type = 'WAVE'

    @property
    def phase(self):
        return np.angle(self.data)

    @phase.setter
    def phase(self, phase):
        self.data = self.amplitude * np.exp(1j * phase)

    @property
    def amplitude(self):
        return np.abs(self.data)

    @amplitude.setter
    def amplitude(self, amplitude):
        self.data = amplitude * np.exp(1j * self.phase)

    @property
    def real(self):
        return np.real(self.data)

    @real.setter
    def real(self, real):
        self.data = real + 1j * self.imag

    @property
    def imag(self):
        return np.imag(self.data)

    @imag.setter
    def imag(self, imag):
        self.data = self.real + 1j * imag

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make sure data is 2D and complex
        self.data = self.data.astype(complex, copy=False)  # Avoid copy if data is already complex!
        # TODO: Add wave image specific constructor commands here if anything comes to mind!

    def get_unwrapped_phase(self):
        """Return the unwrapped phase as an :class:`~hyperspy._signals.Image`.

        Returns
        -------
        phase_image: :class:`~hyperspy._signals.Image`
            Unwrapped phase.

        """
        # TODO: For different unwrap algorithms, maybe add a 'mode' parameter and let user choose!
        # TODO: This single images (not stacks). How should I best iterate over the stacks?
        phase_image = self._deepcopy_with_new_data(unwrap(self.phase))
        phase_image.set_signal_type('')  # New signal is normal image without special signal type!
        phase_image._assign_subclass()  # Changes subclass to be Image!
        return phase_image  # TODO: Is this the correct way to generate image with same metadata?

    def normalize(self, normalization):
        """Normalize the wave. Takes the mean if input is an array.

        Parameters
        ----------
        normalization: float or :class:`~numpy.ndarray`

        """
        self.data /= np.mean(normalization)

    def subtract_reference(self, reference):
        """Subtract a reference wave.

        Parameters
        ----------
        reference: :class:`~hyperspy._signals.WaveImage`
            The reference wave, which should be subtracted.

        """
        assert isinstance(reference, WaveImage), 'Reference should be a WaveImage!'
        assert self.data.shape == reference.data.shape, 'Reference must have the same shape!'
        self.data /= reference.data

    def add_phase_ramp(self, ramp_x, ramp_y, offset=0):
        """Add a linear ramp to the wave.

        Parameters
        ----------
        ramp_x: float
            Slope of the ramp in x-direction.
        ramp_y: float
            Slope of the ramp in y-direction.
        offset: float, optional
            Offset of the ramp at the fulcrum.
        Notes
        -----
            The fulcrum of the linear ramp is at the origin and the slopes are given in units of
            the axis with the according scale taken into account. Both are available via the
            `axes_manager` of the signal.

        """
        yy, xx = np.indices(self.data.shape)
        phase_ramp = offset * np.ones(self.data.shape)
        phase_ramp += ramp_x * xx
        phase_ramp += ramp_y * yy
        self.phase += phase_ramp
        # TODO: Same problem as before, does not work for stacks. How to best iterate over them?

    # def fit_phase_ramp(self, roi=None, filter_order=5, subtract_ramp=True):
    #     """
    #
    #     Parameters
    #     ----------
    #     roi: ???, optional
    #
    #     filter_order: int, optional
    #         Default is 5.
    #     subtract_ramp: boolean, optional
    #         Default is True.
    #
    #     Returns
    #     -------
    #     (ramp_x, ramp_y): tuple of floats
    #
    #     """
    #     ramp_x, ramp_y = 1, 1
    #     if subtract_ramp:
    #         self.add_phase_ramp(-ramp_x, -ramp_y)
    #     return ramp_x, ramp_y
    # TODO: Florian, do your magic here and edit docstring!

# TODO: Some functions can't handle image stacks. Use wrapper functions which iterate over images?

# TODO: Overwrite plot function to be able to plot real, imag, phase, amplitude?

# TODO: Store applied operations somewhere in metadata?

# TODO: Will this work with image stacks? Should we ensure that the size is just 2D (no stacks)?
