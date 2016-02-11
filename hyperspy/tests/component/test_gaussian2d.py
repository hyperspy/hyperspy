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


import nose.tools as nt
import numpy as np
from hyperspy.components import Gaussian2D, SymmetricGaussian2D
from hyperspy.signals import Image

sqrt2pi = np.sqrt(2 * np.pi)
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))


class TestSymmetricGaussian2D:

    def setUp(self):
        g = SymmetricGaussian2D()
        g.centre_x.value = 50
        g.centre_y.value = 100
        g.sigma.value = 20
        g.A.value = 40
        self.g = g
        self.s = Image(g.function(*np.mgrid[0:150, 0:150]))

    def test_fitting(self):
        m = self.s.create_model()
        g = SymmetricGaussian2D()
        # Need to set some sensible initial values
        g.centre_x.value = 40
        g.centre_y.value = 120
        g.A.value = 20
        g.sigma.value = 10
        m.append(g)
        m.fit()
        model_data = m.as_signal().data
        nt.assert_true((self.s.data == model_data).all)


class TestGaussian2D:

    def setUp(self):
        g = Gaussian2D()
        g.centre_x.value = 60
        g.centre_y.value = 80
        g.sigma_x.value = 10
        g.sigma_y.value = 20
        g.rotation.value = np.pi/4
        g.A.value = 40
        self.g = g
        self.s = Image(g.function(*np.mgrid[0:150, 0:150]))

    def test_fitting(self):
        m = self.s.create_model()
        g = Gaussian2D()
        # Need to set some sensible initial values
        g.centre_x.value = 40
        g.centre_y.value = 120
        g.A.value = 20
        g.sigma_x.value = 10
        g.sigma_y.value = 10
        m.append(g)
        m.fit()
        model_data = m.as_signal().data
        nt.assert_true((self.s.data == model_data).all)
