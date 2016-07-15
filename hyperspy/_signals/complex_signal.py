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
import h5py

from hyperspy.signal import BaseSignal
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING, COMPLEX_DOCSTRING, KWARGS_DOCSTRING)


class ComplexSignal(BaseSignal):

    """BaseSignal subclass for complex data."""

    _dtype = "complex"

    @property
    def real(self):
        """Get/set the real part of the data. Returns an appropriate HyperSpy signal."""
        real = self._deepcopy_with_new_data(np.real(self.data))
        if real.metadata.General.title:
            title = real.metadata.General.title
        else:
            title = 'Untitled Signal'
        real.metadata.General.title = 'real({})'.format(title)
        real._assign_subclass()
        return real

    @real.setter
    def real(self, real):
        if isinstance(real, BaseSignal):
            self.real.isig[:] = real
        else:
            self.data.real[:] = real

    @property
    def imag(self):
        """Get/set imaginary part of the data. Returns an appropriate HyperSpy signal."""
        imag = self._deepcopy_with_new_data(np.imag(self.data))
        if imag.metadata.General.title:
            title = imag.metadata.General.title
        else:
            title = 'Untitled Signal'
        imag.metadata.General.title = 'imag({})'.format(title)
        imag._assign_subclass()
        return imag

    @imag.setter
    def imag(self, imag):
        if isinstance(imag, BaseSignal):
            self.imag.isig[:] = imag
        else:
            self.data.imag[:] = imag

    @property
    def amplitude(self):
        """Get/set the amplitude of the data. Returns an appropriate HyperSpy signal."""
        amplitude = np.abs(self)
        amplitude._assign_subclass()
        return amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        if isinstance(amplitude, BaseSignal):
            self.isig[:] = amplitude * np.exp(self.angle() * 1j)
        else:
            self.data[:] = amplitude * np.exp(1j * np.angle(self.data))

    @property
    def phase(self):
        """Get/set the phase of the data. Returns an appropriate HyperSpy signal."""
        phase = self._deepcopy_with_new_data(np.angle(self.data))
        if phase.metadata.General.title:
            title = phase.metadata.General.title
        else:
            title = 'Untitled Signal'
        phase.metadata.General.title = 'phase({})'.format(title)
        phase._assign_subclass()
        return phase

    @phase.setter
    def phase(self, phase):
        if isinstance(phase, BaseSignal):
            self.isig[:] = np.abs(self) * np.exp(phase * 1j)
        else:
            self.data[:] = np.abs(self.data) * np.exp(1j * phase)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.issubdtype(self.data.dtype, complex):
            self.data = self.data.astype(complex)

    def change_dtype(self, dtype):
        """Change the data type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast. For complex signals only other
            complex dtypes are allowed. If real valued properties are required use `real`,
            `imag`, `amplitude` and `phase` instead.
        """
        if np.issubdtype(dtype, complex):
            self.data = self.data.astype(dtype)
        else:
            raise AttributeError(
                'Complex data can only be converted into other complex dtypes!')

    def angle(self, deg=False):
        """Return the angle (also known as phase or argument). If the data is real, the angle is 0
        for positive values and 2$\pi$ for negative values.

        Parameters
        ----------
        deg : bool, optional
            Return angle in degrees if True, radians if False (default).

        Returns
        -------
        angle : HyperSpy signal
            The counterclockwise angle from the positive real axis on the complex plane,
            with dtype as numpy.float64.

        """
        sig = self._deepcopy_with_new_data(np.angle(self.data, deg))
        sig.set_signal_type('')
        if sig.metadata.General.title:
            title = sig.metadata.General.title
        else:
            title = 'Untitled Signal'
        sig.metadata.General.title = 'angle({})'.format(title)
        return sig

    def unwrapped_phase(self, wrap_around=False, seed=None,
                        show_progressbar=None):
        """Return the unwrapped phase as an appropriate HyperSpy signal.

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
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.

        Returns
        -------
        phase_image: :class:`~hyperspy._signals.BaseSignal` subclass
            Unwrapped phase.

        Notes
        -----
        Uses the :func:`~skimage.restoration.unwrap_phase` function from `skimage`.
        The algorithm is based on Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor,
        and Munther A. Gdeisat, “Fast two-dimensional phase-unwrapping algorithm based on sorting
        by reliability following a noncontinuous path”, Journal Applied Optics,
        Vol. 41, No. 35, pp. 7437, 2002

        """
        from skimage.restoration import unwrap_phase
        phase = self.phase
        phase.map(unwrap_phase, wrap_around=wrap_around, seed=seed,
                  show_progressbar=show_progressbar)
        phase.metadata.General.title = 'unwrapped {}'.format(
            phase.metadata.General.title)
        return phase  # Now unwrapped!

    def plot(self, navigator="auto", axes_manager=None,
             representation='cartesian', **kwargs):
        """%s
        %s
        %s

        """
        if representation == 'cartesian':
            self.real.plot(
                navigator=navigator,
                axes_manager=self.axes_manager,
                **kwargs)
            self.imag.plot(
                navigator=navigator,
                axes_manager=self.axes_manager,
                **kwargs)
        elif representation == 'polar':
            self.amplitude.plot(
                navigator=navigator,
                axes_manager=self.axes_manager,
                **kwargs)
            self.phase.plot(
                navigator=navigator,
                axes_manager=self.axes_manager,
                **kwargs)
        else:
            raise KeyError('{}'.format(representation) +
                           'is not a valid input for representation (use "cartesian" or "polar")!')
    plot.__doc__ %= BASE_PLOT_DOCSTRING, COMPLEX_DOCSTRING, KWARGS_DOCSTRING
