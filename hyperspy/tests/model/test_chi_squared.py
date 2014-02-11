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


import numpy as np

from nose.tools import assert_true
from hyperspy._signals.spectrum import Spectrum
from hyperspy.hspy import create_model 
from hyperspy.components import Gaussian

class TestChiSquared:
    def setUp(self):
        s = Spectrum(np.array([1.0, 2, 4, 7, 12, 7, 4, 2, 1]))
        m = create_model(s)
        self.model = m
        self.red_chisq = 1.55793245
        self.chisq = 7.78966223

    def test_chisq(self):
        m = self.model
        g = Gaussian()
        m.append(g)
        m.fit()
        assert_true(np.allclose(m.chisq(), self.chisq))

    def test_dof(self):
        m = self.model
        g = Gaussian()
        g1 = Gaussian()
        m.extend((g, g1))
        g1.set_parameters_not_free('A')
        m.fit()
        assert_true(np.equal(m.dof(), 5))

    def test_red_chisq(self):
        m = self.model
        g = Gaussian()
        m.append(g)
        m.fit()
        assert_true(np.allclose(m.red_chisq(), self.red_chisq))
