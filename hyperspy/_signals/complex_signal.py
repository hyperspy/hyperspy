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

from hyperspy.signal import BaseSignal


class ComplexSignal(BaseSignal):
    """BaseSignal subclass for complex data."""

    _signal_type = "complex"

    @property
    def real(self):
        """Get/set the real part of the data. Returns an appropriate HyperSpy signal."""
        real = self._deepcopy_with_new_data(np.real(self.data))
        if real.metadata.General.title:
            title = real.metadata.General.title
        else:
            title = 'Untitled Signal'
        real.metadata.General.title = 'real({})'.format(title)
        real.set_signal_type('')  # Result is no longer complex!
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
        imag.set_signal_type('')  # Result is no longer complex!
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
        amplitude.set_signal_type('')  # Result is no longer complex!
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
        phase.set_signal_type('')  # Result is no longer complex!
        return phase

    @phase.setter
    def phase(self, phase):
        if isinstance(phase, BaseSignal):
            self.isig[:] = np.abs(self) * np.exp(phase * 1j)
        else:
            self.data[:] = np.abs(self.data) * np.exp(1j * phase)

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

    def unwrapped_phase(self, wrap_around=False, seed=None, show_progressbar=None):
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
        phase_image: :class:`~hyperspy._signals.Signal2D`
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
        return phase  # Now unwrapped!

    def plot(self, navigator="auto", axes_manager=None, representation='cartesian', **kwargs):
        """Plot the signal at the current coordinates.

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

        representation : {'cartesian' or 'angular'}
            Determines if the real and imaginary part of the complex data is plotted ('cartesian',
            default), or if the amplitude and phase should be used ('angular').

        **kwargs : optional
            Any extra keyword arguments are passed to the signal plot.

        """
        if representation == 'cartesian':
            self.real.plot(navigator=navigator, axes_manager=self.axes_manager, **kwargs)
            self.imag.plot(navigator=navigator, axes_manager=self.axes_manager, **kwargs)
        elif representation == 'angular':
            self.amplitude.plot(navigator=navigator, axes_manager=self.axes_manager, **kwargs)
            self.phase.plot(navigator=navigator, axes_manager=self.axes_manager, **kwargs)
        else:
            raise KeyError('{} is not a valid input for representation (use "cartesion" or '
                           '"angular")!'.format(representation))
