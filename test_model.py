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


import os

import numpy as np

import nose.tools
from hyperspy._signals.spectrum import Spectrum
from hyperspy.hspy import create_model
from hyperspy.components import Gaussian


class TestModel:

    def setUp(self):
        s = Spectrum(np.empty(1))
        m = create_model(s)
        self.model = m

    def test_access_component_by_name(self):
        m = self.model
        g1 = Gaussian()
        g2 = Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nose.tools.assert_is(m["test"], g2)

    def test_access_component_by_index(self):
        m = self.model
        g1 = Gaussian()
        g2 = Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nose.tools.assert_is(m[1], g2)

    def test_component_name_when_append(self):
        m = self.model
        gs = [Gaussian(), Gaussian(), Gaussian()]
        m.extend(gs)
        nose.tools.assert_is(m['Gaussian'], gs[0])
        nose.tools.assert_is(m['Gaussian_0'], gs[1])
        nose.tools.assert_is(m['Gaussian_1'], gs[2])

    @nose.tools.raises(ValueError)
    def test_several_component_with_same_name(self):
        m = self.model
        gs = [Gaussian(), Gaussian(), Gaussian()]
        m.extend(gs)
        m[0]._name = "Gaussian"
        m[1]._name = "Gaussian"
        m[2]._name = "Gaussian"
        m['Gaussian']

    @nose.tools.raises(ValueError)
    def test_no_component_with_that_name(self):
        m = self.model
        m['Voigt']

    @nose.tools.raises(ValueError)
    def test_component_already_in_model(self):
        m = self.model
        g1 = Gaussian()
        m.extend((g1, g1))

    def test_remove_component(self):
        m = self.model
        g1 = Gaussian()
        m.append(g1)
        m.remove(g1)
        nose.tools.assert_equal(len(m), 0)

    def test_remove_component_by_index(self):
        m = self.model
        g1 = Gaussian()
        m.append(g1)
        m.remove(0)
        nose.tools.assert_equal(len(m), 0)

    def test_remove_component_by_name(self):
        m = self.model
        g1 = Gaussian()
        m.append(g1)
        m.remove(g1.name)
        nose.tools.assert_equal(len(m), 0)

    def test_get_component_by_name(self):
        m = self.model
        g1 = Gaussian()
        g2 = Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nose.tools.assert_is(m._get_component("test"), g2)

    def test_get_component_by_index(self):
        m = self.model
        g1 = Gaussian()
        g2 = Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nose.tools.assert_is(m._get_component(1), g2)

    def test_get_component_by_component(self):
        m = self.model
        g1 = Gaussian()
        g2 = Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        nose.tools.assert_is(m._get_component(g2), g2)

    @nose.tools.raises(ValueError)
    def test_get_component_wrong(self):
        m = self.model
        g1 = Gaussian()
        g2 = Gaussian()
        g2.name = "test"
        m.extend((g1, g2))
        m._get_component(1.2)
