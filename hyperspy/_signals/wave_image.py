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
        phase = self.angle(deg=False)  # Phase is always in rad!
        phase.set_signal_type('')  # Go from WaveImage to Signal2D!
        return phase

    @phase.setter
    def phase(self, phase):
        self.isig[...] = np.abs(self) * np.exp(phase * 1j)

    @property
    def amplitude(self):
        """Get/set the amplitude of the data. Returns an :class:`~hyperspy.signals.Signal2D`."""
        amplitude = np.abs(self)
        amplitude.set_signal_type('')  # Go from WaveImage to Signal2D!
        return amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        self.isig[:] = amplitude * np.exp(self.angle() * 1j)
