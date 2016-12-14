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


from hyperspy._signals.complex_signal2d import ComplexSignal2D


class ElectronWaveImage(ComplexSignal2D):

    """ComplexSignal2D subclass for electron wave images."""

    _signal_type = 'electron_wave'

    # The class is empty at the moment, but some electron wave specific methods will be added later.

    # @property
    # def reconstruction_parameters(self):
    #     assert self.metadata.Signal.has_item('holo_reconstruction_parameters'), \
    #         "No reconstruction parameters assigned to the wave"
    #
    #     return self.metadata.Signal.holo_reconstruction_parameters.as_dictionary()
