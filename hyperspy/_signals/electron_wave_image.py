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

from hyperspy._signals.complex_signal import ComplexSignal2D
from hyperspy._signals.signal2d import Signal2D


class ElectronWaveImage(ComplexSignal2D):
    """Signal2D subclass for complex electron wave data (e.g. reconstructed from holograms)."""

    _signal_type = "electron_wave"

    def add_phase_ramp(self, ramp_x, ramp_y, offset=0):
        """Add a linear phase ramp to the wave.

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
        phase = self.phase
        phase.add_ramp(ramp_x, ramp_y, offset)
        self.phase = phase
