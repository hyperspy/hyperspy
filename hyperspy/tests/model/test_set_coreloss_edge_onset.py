# Copyright 2007-2012 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from hyperspy.signals import EELSSpectrum
from hyperspy.components1d import Gaussian


class TestSetOnset:

    def setup_method(self, method):
        g = Gaussian()
        g.A.value = 10000.0
        g.centre.value = 5000.0
        g.sigma.value = 500.0
        axis = np.arange(10000)
        s = EELSSpectrum(g.function(axis))
        s.set_microscope_parameters(
                beam_energy=100,
                convergence_angle=10,
                collection_angle=10)
        s.add_elements(('O',))
        m = s.create_model(auto_background=False)
        self.model = m
        self.g = g
        self.top_point = s.data.max()
        self.rtol = 0.1

    def test_set_onset_100_percent(self):
        m = self.model
        g = self.g
        top_point = self.top_point
        percent_position = 1.0

        m.set_coreloss_edge_onset(
                m[0], signal_range=(1000, 5500),
                percent_position=percent_position)
        np.testing.assert_allclose(
                g.function(m[0].onset_energy.value),
                top_point*percent_position,
                rtol=self.rtol)

    def test_set_onset_50_percent(self):
        m = self.model
        g = self.g
        top_point = self.top_point
        percent_position = 0.5
        m.set_coreloss_edge_onset(
                m[0], signal_range=(1000, 5500),
                percent_position=percent_position)
        np.testing.assert_allclose(
                g.function(m[0].onset_energy.value),
                top_point*percent_position,
                rtol=self.rtol)

    def test_set_onset_10_percent(self):
        m = self.model
        g = self.g
        top_point = self.top_point
        percent_position = 0.1

        m.set_coreloss_edge_onset(
                m[0], signal_range=(1000, 5500),
                percent_position=percent_position)
        np.testing.assert_allclose(
                g.function(m[0].onset_energy.value),
                top_point*percent_position,
                rtol=self.rtol)

    def test_set_onset_1_percent(self):
        m = self.model
        g = self.g
        top_point = self.top_point
        percent_position = 0.01

        m.set_coreloss_edge_onset(
                m[0], signal_range=(1000, 5500),
                percent_position=percent_position)
        np.testing.assert_allclose(
                g.function(m[0].onset_energy.value),
                top_point*percent_position,
                rtol=self.rtol)
