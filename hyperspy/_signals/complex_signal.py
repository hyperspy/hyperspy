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

from functools import wraps

import numpy as np
import dask.array as da

from hyperspy.signal import BaseSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING, COMPLEX_DOCSTRING, KWARGS_DOCSTRING)


def format_title(thing):
    def title_decorator(func):
        @wraps(func)
        def signal_wrapper(*args, **kwargs):
            signal = func(*args, **kwargs)
            if signal.metadata.General.title:
                title = signal.metadata.General.title
            else:
                title = 'Untitled Signal'
            signal.metadata.General.title = thing + '({})'.format(title)
            return signal
        return signal_wrapper
    return title_decorator


class ComplexSignal_mixin:

    """BaseSignal subclass for complex data."""

    _dtype = "complex"

    @format_title('real')
    def _get_real(self):
        real = self._deepcopy_with_new_data(self.data.real)
        real._assign_subclass()
        return real

    real = property(lambda s: s._get_real(),
                    lambda s, v: s._set_real(v),
                    doc="""Get/set the real part of the data. Returns an
                    appropriate HyperSpy signal."""
                    )

    @format_title('imag')
    def _get_imag(self):
        imag = self._deepcopy_with_new_data(self.data.imag)
        imag._assign_subclass()
        return imag

    imag = property(lambda s: s._get_imag(),
                    lambda s, v: s._set_imag(v),
                    doc="""Get/set imaginary part of the data. Returns an
                    appropriate HyperSpy signal."""
                    )

    def _get_amplitude(self, amplitude):
        amplitude._assign_subclass()
        return amplitude

    amplitude = property(lambda s: s._get_amplitude(),
                         lambda s, v: s._set_amplitude(v),
                         doc="""Get/set the amplitude of the data. Returns an
                         appropriate HyperSpy signal.""")

    @format_title('phase')
    def _get_phase(self, phase):
        phase._assign_subclass()
        return phase

    phase = property(lambda s: s._get_phase(),
                     lambda s, v: s._set_phase(v),
                     doc="""Get/set the phase of the data. Returns an appropriate
                     HyperSpy signal."""
                     )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # _plot_kwargs store the plot kwargs argument for convenience when
        # plotting ROI in order to use the same plotting options than the
        # original plot
        self._plot_kwargs = {}
        if not np.issubdtype(self.data.dtype, np.complexfloating):
            self.data = self.data.astype(np.complexfloating)

    def change_dtype(self, dtype):
        """Change the data type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast. For complex signals only other
            complex dtypes are allowed. If real valued properties are required use `real`,
            `imag`, `amplitude` and `phase` instead.
        """
        if np.issubdtype(dtype, np.complexfloating):
            self.data = self.data.astype(dtype)
        else:
            raise AttributeError(
                'Complex data can only be converted into other complex dtypes!')

    @format_title('angle')
    def angle(self, angle, deg=False):
        r"""Return the angle (also known as phase or argument). If the data is real, the angle is 0
        for positive values and :math:`2\pi` for negative values.

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
        angle.set_signal_type('')
        return angle

    def unwrapped_phase(self, wrap_around=False, seed=None,
                        show_progressbar=None, parallel=None):
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
        parallel : {Bool, None, int}
            Perform the operation parallely

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
                  show_progressbar=show_progressbar, ragged=False,
                  parallel=parallel)
        phase.metadata.General.title = 'unwrapped {}'.format(
            phase.metadata.General.title)
        return phase  # Now unwrapped!

    def __call__(self, axes_manager=None, power_spectrum=False,
                 fft_shift=False):
        value = super().__call__(axes_manager=axes_manager,
                                 fft_shift=fft_shift)
        if power_spectrum:
            value = np.abs(value)**2
        return value

    def plot(self, power_spectrum=False, navigator="auto", axes_manager=None,
             representation='cartesian', norm="auto", fft_shift=False,
             same_axes=True, **kwargs):
        """%s
        %s
        %s

        """
        if norm is "auto":
            norm = 'log' if power_spectrum else 'linear'

        kwargs.update({'norm': norm,
                       'fft_shift': fft_shift,
                       'navigator': navigator,
                       'axes_manager': self.axes_manager})
        if representation == 'cartesian':
            if ((same_axes and self.axes_manager.signal_dimension == 1) or
                    power_spectrum):
                kwargs['power_spectrum'] = power_spectrum
                super().plot(**kwargs)
            else:
                self.real.plot(**kwargs)
                self.imag.plot(**kwargs)
        elif representation == 'polar':
            if same_axes and self.axes_manager.signal_dimension == 1:
                amp = self.amplitude
                amp.change_dtype("complex")
                amp.imag = self.phase
                amp.plot(**kwargs)
            else:
                self.amplitude.plot(**kwargs)
                self.phase.plot(**kwargs)
        else:
            raise ValueError('{}'.format(representation) +
                             'is not a valid input for representation (use "cartesian" or "polar")!')

        self._plot_kwargs = {'power_spectrum': power_spectrum,
                             'representation': representation,
                             'norm': norm,
                             'fft_shift': fft_shift,
                             'same_axes': same_axes}
    plot.__doc__ %= BASE_PLOT_DOCSTRING, COMPLEX_DOCSTRING, KWARGS_DOCSTRING


class ComplexSignal(ComplexSignal_mixin, BaseSignal):

    def _get_phase(self):
        phase = self._deepcopy_with_new_data(np.angle(self.data))
        return super()._get_phase(phase)

    def _get_amplitude(self):
        amplitude = np.abs(self)
        return super()._get_amplitude(amplitude)

    def _set_real(self, real):
        if isinstance(real, BaseSignal):
            self.real.isig[:] = real
        else:
            self.data.real[:] = real
        self.events.data_changed.trigger(self)

    def _set_imag(self, imag):
        if isinstance(imag, BaseSignal):
            self.imag.isig[:] = imag
        else:
            self.data.imag[:] = imag
        self.events.data_changed.trigger(self)

    def _set_amplitude(self, amplitude):
        if isinstance(amplitude, BaseSignal):
            self.isig[:] = amplitude * np.exp(self.angle() * 1j)
        else:
            self.data[:] = amplitude * np.exp(1j * np.angle(self.data))
        self.events.data_changed.trigger(self)

    def _set_phase(self, phase):
        if isinstance(phase, BaseSignal):
            self.isig[:] = np.abs(self) * np.exp(phase * 1j)
        else:
            self.data[:] = np.abs(self.data) * np.exp(1j * phase)
        self.events.data_changed.trigger(self)

    def angle(self, deg=False):
        angle = self._deepcopy_with_new_data(np.angle(self.data, deg))
        return super().angle(angle, deg=deg)
    angle.__doc__ = ComplexSignal_mixin.angle.__doc__


class LazyComplexSignal(ComplexSignal, LazySignal):

    @format_title('absolute')
    def _get_amplitude(self):
        amplitude = abs(self)
        return super(ComplexSignal, self)._get_amplitude(amplitude)

    def _get_phase(self):
        phase = self._deepcopy_with_new_data(da.angle(self.data))
        return super(ComplexSignal, self)._get_phase(phase)

    def _set_real(self, real):
        if isinstance(real, BaseSignal):
            real = real.data.real
        self.data = 1j * self.data.imag + real
        self.events.data_changed.trigger(self)

    def _set_imag(self, imag):
        if isinstance(imag, BaseSignal):
            imag = imag.data.real
        self.data = self.data.real + 1j * imag
        self.events.data_changed.trigger(self)

    def _set_amplitude(self, amplitude):
        if isinstance(amplitude, BaseSignal):
            amplitude = amplitude.data.real
        self.data = amplitude * da.exp(1j * da.angle(self.data))
        self.events.data_changed.trigger(self)

    def _set_phase(self, phase):
        if isinstance(phase, BaseSignal):
            phase = phase.data.real
        self.data = abs(self.data) * \
            da.exp(1j * phase)
        self.events.data_changed.trigger(self)

    def angle(self, deg=False):
        angle = self._deepcopy_with_new_data(da.angle(self.data, deg))
        return super(ComplexSignal, self).angle(angle, deg=deg)
    angle.__doc__ = ComplexSignal_mixin.angle.__doc__
