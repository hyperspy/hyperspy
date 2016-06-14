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

from hyperspy._signals.image import Image


class WaveImage(Image):
    """Image subclass for complex electron wave data (e.g. reconstructed from holograms)."""

    _signal_type = 'WAVE'

    @property
    def phase(self):
        """Get/set the phase of the data. Returns an :class:`~hyperspy._signals.Image`."""
        phase = self._deepcopy_with_new_data(np.angle(self.data))
        phase.set_signal_type('')  # Result is a normal Image!
        return phase

    @phase.setter
    def phase(self, phase):
        if isinstance(phase, Image):
            phase = phase.data
        self.data = self.amplitude.data * np.exp(1j * phase)

    @property
    def amplitude(self):
        """Get/set the amplitude of the data. Returns an :class:`~hyperspy._signals.Image`."""
        amplitude = self._deepcopy_with_new_data(np.abs(self.data))
        amplitude.set_signal_type('')  # Result is a normal Image!
        return amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        if isinstance(amplitude, Image):
            amplitude = amplitude.data
        self.data = amplitude * np.exp(1j * self.phase.data)

    @property
    def real(self):
        """Get/set the real part of the data. Returns an :class:`~hyperspy._signals.Image`."""
        real = self._deepcopy_with_new_data(np.real(self.data))
        real.set_signal_type('')  # Result is a normal Image!
        return real

    @real.setter
    def real(self, real):
        if isinstance(real, Image):
            real = real.data
        self.data = real + 1j * self.imag.data

    @property
    def imag(self):
        """Get/set the imaginary part of the data. Returns an :class:`~hyperspy._signals.Image`."""
        imag = self._deepcopy_with_new_data(np.imag(self.data))
        imag.set_signal_type('')  # Result is a normal Image!
        return imag

    @imag.setter
    def imag(self, imag):
        if isinstance(imag, Image):
            imag = imag.data
        self.data = self.real.data + 1j * imag

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make sure data is complex:
        self.change_dtype(complex)

    def get_unwrapped_phase(self, wrap_around=False, seed=None):
        """Return the unwrapped phase as an :class:`~hyperspy._signals.Image`.

        Parameters
        ----------
        wrap_around : bool or sequence of bool, optional
            When an element of the sequence is  `True`, the unwrapping process
            will regard the edges along the corresponding axis of the image to be
            connected and use this connectivity to guide the phase unwrapping
            process. If only a single boolean is given, it will apply to all axes.
            Wrap around is not supported for 1D arrays.
        seed : int, optional
            Unwrapping 2D or 3D images uses random initialization. This sets the
            seed of the PRNG to achieve deterministic behavior.

        Returns
        -------
        phase_image: :class:`~hyperspy._signals.Image`
            Unwrapped phase.

        Notes
        -----
        Uses the :func:`~skimage.restoration.unwrap_phase` function from `skimage`.

        """
        phase_image = self._deepcopy_with_new_data(self.phase.data)  # Get copy of just the phase!
        phase_image.set_signal_type('')  # New signal is normal image without special signal type!
        phase_image.map(unwrap, wrap_around=wrap_around, seed=seed)  # Unwrap phase!
        return phase_image

    def normalize(self, normalization):
        """Normalize the wave. Takes the mean if input is an array.

        Parameters
        ----------
        normalization: complex or :class:`~numpy.ndarray`

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
        yy, xx = np.indices(self.axes_manager._signal_shape_in_array)
        phase = self.phase.data
        phase += offset * np.ones(self.data.shape)
        phase += ramp_x * xx
        phase += ramp_y * yy
        self.phase = phase
