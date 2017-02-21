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
import pytest

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass
from hyperspy.gui.egerton_quantification import SpikesRemoval


class _TestSpikesRemovalTool:

    def _add_spikes_to_data(self, data, coordinates, values):
        data2 = data.copy()
        for coordinate, value in zip(coordinates, values):
            data2[coordinate] *= value
        return data2
    
    def _get_index_from_coordinate(self, coordinate):
        return self.s.data.shape[1]*coordinate[0] + coordinate[1]

        
@lazifyTestClass
class TestSpikesRemovalTool(_TestSpikesRemovalTool):

    def setup_method(self, method):
        data = np.arange(5*10*20).reshape(5, 10, 20)
        self.coordinates, values = ((1, 6, 15), (3, 5, 10)), (5, 10)

        # Does not contain spikes
        self.s = hs.signals.Signal1D(data)
        # Does contain spikes, coordinates and values above
        self.s2 = hs.signals.Signal1D(self._add_spikes_to_data(data,
                                                               self.coordinates,
                                                               values))
        self.s.add_gaussian_noise(5)
        self.s2.add_gaussian_noise(5)
        self.sr = SpikesRemoval(self.s)
        self.sr2 = SpikesRemoval(self.s2)
    
    def test_detect_spikes(self):
        assert self.sr.detect_spike() == False

        for coordinate in self.coordinates:
            self.sr2.index = self._get_index_from_coordinate(coordinate)
            assert self.sr2.detect_spike() == True

    def test_find_spikes(self):
        # should raise an NotImplementedError; because it tries to display the
        # message 'End of dataset reached'
        with pytest.raises(NotImplementedError):
            self.sr.find()
            assert self.sr.index == 0 

        for coordinate in self.coordinates:
            self.sr2.find()
            assert self.sr2.index == self._get_index_from_coordinate(coordinate)
    
    def test_remove_spikes(self):
        self.sr2.find()
        for coordinate in self.coordinates:
            if coordinate == self.coordinates[-1]:
                # Reach the end of the dataset, will display a message
                with pytest.raises(NotImplementedError):
                    self.sr2.apply()
            else:
                self.sr2.apply()
        assert self.sr2.detect_spike() == False
        