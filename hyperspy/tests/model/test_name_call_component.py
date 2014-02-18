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

from nose.tools import assert_true, assert_equal, assert_not_equal, raises
from hyperspy._signals.spectrum import Spectrum
from hyperspy.hspy import create_model
from hyperspy.components import Gaussian


class TestSetParameterInModel:

    def setUp(self):
        g1 = Gaussian()
        g2 = Gaussian()
        g3 = Gaussian()
        s = Spectrum(np.arange(1000).reshape(10, 10, 10))
        m = create_model(s)
        m.append(g1)
        m.append(g2)
        m.append(g3)
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.model = m

    def test_call_component(self):
        m = self.model
        g1 = self.g1
        g3 = self.g3
        g3.name = "gaussian3"

        temp_gaussian_1 = m[0]
        temp_gaussian_2 = m["gaussian3"]

        assert_true(temp_gaussian_1 is g1)
        assert_true(temp_gaussian_2 is g3)

    def test_component_name_when_append(self):
        m = self.model
        g1 = self.g1
        g2 = self.g2
        g3 = self.g3
        assert_true(m['Gaussian'] is g1)
        assert_true(m['Gaussian_0'] is g2)
        assert_true(m['Gaussian_1'] is g3)

    @raises(ValueError)
    def test_several_component_with_same_name(self):
        m = self.model
        m[0]._name = "Gaussian"
        m[1]._name = "Gaussian"
        m[2]._name = "Gaussian"
        m['Gaussian']

    @raises(ValueError)
    def test_no_component_with_that_name(self):
        m = self.model
        m['Voigt']

