# Copyright 2007-2012 The Hyperspy developers
#
# This file is part of Hyperspy.
#
# Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hyperspy. If not, see <http://www.gnu.org/licenses/>.


import os

import numpy as np

from nose.tools import assert_true, assert_equal, assert_not_equal
from hyperspy.signals.spectrum import Spectrum
from hyperspy.hspy import create_model 
from hyperspy.components import Gaussian


class TestFitOneComponent:
    def setUp(self):
        g = Gaussian()
        g.A.value = 10000.0
        g.centre.value = 5000.0
        g.sigma.value = 500.0
        s = Spectrum({'data' : g.function(np.arange(10000))})
        m = create_model(s)
        self.A = g.A.value
        self.centre = g.centre.value
        self.sigma = g.sigma.value
        self.model = m

    def test_fit_component(self):
        m = self.model 
        g1 = Gaussian()
        m.append(g1)
        m.fit_component(g1, signal_range=(4000,6000))
        assert_true(
                (g1.centre.value < self.centre*1.05) and 
                (g1.centre.value > self.centre*0.95))
        assert_true(
                (g1.A.value < self.A*1.05) and 
                (g1.A.value > self.A*0.95))
        assert_true(
                (g1.sigma.value < self.sigma*1.05) and 
                (g1.sigma.value > self.sigma*0.95))
