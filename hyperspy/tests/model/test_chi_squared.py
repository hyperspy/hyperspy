# Copyright 2007-2016 The HyperSpy developers
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

from hyperspy._signals.signal1d import Signal1D
from hyperspy.components1d import Gaussian


class TestChiSquared:

    def setup_method(self, method):
        s = Signal1D(np.array([1.0, 2, 4, 7, 12, 7, 4, 2, 1]))
        m = s.create_model()
        self.model = m
        self.A = 38.022476979172588
        self.sigma = 1.4764966133859543
        self.centre = 4.0000000002462945

    def test_chisq_with_fit(self):
        m = self.model
        g = Gaussian()
        m.append(g)
        m.fit()
        assert np.allclose(m.chisq(), 7.78966223)

    def test_dof_with_fit(self):
        m = self.model
        g = Gaussian()
        g1 = Gaussian()
        m.extend((g, g1))
        g1.set_parameters_not_free('A')
        m.fit()
        assert np.equal(m.dof(), 5)

    def test_red_chisq_with_fit(self):
        m = self.model
        g = Gaussian()
        m.append(g)
        m.fit()
        assert np.allclose(m.red_chisq(), 1.55793245)

    def test_chisq(self):
        m = self.model
        g = Gaussian()
        g.A.value = self.A
        g.sigma.value = self.sigma
        g.centre.value = self.centre
        m.append(g)
        m._calculate_chisq()
        assert np.allclose(m.chisq(), 7.78966223)

    def test_dof_with_p0(self):
        m = self.model
        g = Gaussian()
        g1 = Gaussian()
        m.extend((g, g1))
        g1.set_parameters_not_free('A')
        m._set_p0()
        m._set_current_degrees_of_freedom()
        assert np.equal(m.dof(), 5)

    def test_red_chisq(self):
        m = self.model
        g = Gaussian()
        g.A.value = self.A
        g.sigma.value = self.sigma
        g.centre.value = self.centre
        m.append(g)
        m._set_p0()
        m._set_current_degrees_of_freedom()
        m._calculate_chisq()
        assert np.allclose(m.red_chisq(), 1.55793245)

    def test_chisq_in_range(self):
        m = self.model
        g = Gaussian()
        m.append(g)
        m.set_signal_range(1, 7)
        m.fit()
        assert np.allclose(m.red_chisq(), 2.87544335)

    def test_chisq_with_inactive_components(self):
        m = self.model
        ga = Gaussian()
        gin = Gaussian()
        m.append(ga)
        m.append(gin)
        gin.active = False
        m.fit()
        assert np.allclose(m.chisq(), 7.78966223)

    def test_dof_with_inactive_components(self):
        m = self.model
        ga = Gaussian()
        gin = Gaussian()
        m.append(ga)
        m.append(gin)
        gin.active = False
        m.fit()
        assert np.equal(m.dof(), 3)
