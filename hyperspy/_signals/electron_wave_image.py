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


from hyperspy.signals import BaseSignal, Signal1D, ComplexSignal2D


class ElectronWaveImage(ComplexSignal2D):

    """ComplexSignal2D subclass for electron wave images."""

    _signal_type = 'electron_wave'

    def display_reconstruction_parameters(self):
        assert self.metadata.Signal.has_item('Holography.Reconstruction_parameters'), \
            "No reconstruction parameters assigned to the wave"

        sb_position = self.metadata.Signal.Holography.Reconstruction_parameters.sb_position
        sb_size = self.metadata.Signal.Holography.Reconstruction_parameters.sb_size
        sb_smoothness = self.metadata.Signal.Holography.Reconstruction_parameters.sb_smoothness
        sb_unit = self.metadata.Signal.Holography.Reconstruction_parameters.sb_units

        if isinstance(sb_position, Signal1D):
            print('sb_position = ', sb_position.data)
        else:
            print('sb_position = ', sb_position)

        if isinstance(sb_size, BaseSignal):
            print('sb_size = ', sb_size.data)
        else:
            print('sb_size= ', sb_size)

        if isinstance(sb_smoothness, BaseSignal):
            print('sb_smoothness = ', sb_smoothness.data)
        else:
            print('sb_smoothness = ', sb_smoothness)

        print('sb_unit = ', sb_unit)
