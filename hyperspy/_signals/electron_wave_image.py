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

    _signal_dimension = 2
    _signal_type = 'ElectronWaveImage'

    @property
    def rec_param(self):
        assert self.metadata.Signal.has_item('holo_rec_param'), "No reconstruction parameters assigned to the wave"

        rec_param = (self.metadata.Signal.holo_rec_param.as_dictionary()['sb_pos_x0'],
                     self.metadata.Signal.holo_rec_param.as_dictionary()['sb_pos_y0'],
                     self.metadata.Signal.holo_rec_param.as_dictionary()['sb_pos_x1'],
                     self.metadata.Signal.holo_rec_param.as_dictionary()['sb_pos_y1'],
                     self.metadata.Signal.holo_rec_param.as_dictionary()['sb_size'])

        return rec_param
