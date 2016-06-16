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

from hyperspy._signals.signal2d import Signal2D
from hyperspy.signal import BaseSignal


class WaveImage(Signal2D):
    """Signal2D subclass for complex electron wave data (e.g. reconstructed from holograms)."""

    _signal_type = 'wave'

    @property
    def phase(self):
        """Get/set the phase of the data. Returns an :class:`~hyperspy.signals.Signal2D`."""
        phase = self._deepcopy_with_new_data(np.angle(self.data))
        phase.set_signal_type('')
        return phase

    @phase.setter
    def phase(self, phase):
        if isinstance(phase, BaseSignal):
            phase = phase.data
        self.data = self.amplitude.data * np.exp(1j * phase)

    @property
    def amplitude(self):
        """Get/set the amplitude of the data. Returns an :class:`~hyperspy.signals.Signal2D`."""
        amplitude = self._deepcopy_with_new_data(np.abs(self.data))
        amplitude.set_signal_type('')
        return amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        if isinstance(amplitude, BaseSignal):
            amplitude = amplitude.data
        self.data = amplitude * np.exp(1j * self.phase.data)

    def get_unwrapped_phase(self, wrap_around=False, seed=None, show_progressbar=None):
        """Return the unwrapped phase as an :class:`~hyperspy.signals.Signal2D`.

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

        """
        from skimage.restoration import unwrap_phase
        phase = self.phase.deepcopy()
        phase.map(unwrap_phase, wrap_around=wrap_around, seed=seed,
                  show_progressbar=show_progressbar)
        return phase  # Now unwrapped!

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
