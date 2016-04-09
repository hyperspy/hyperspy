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
from hyperspy.io import load
import os

sqrt2pi = np.sqrt(2 * np.pi)
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))
my_path = os.path.dirname(__file__)


class TestSymmetricGaussian2DFitting:

    """Test using a 2-D Gaussian generated with
    Gaussian2D(
        A=1000., sigma=2.)"""
    def setUp(self):
        self.s = load(
            my_path +
            "/test_gaussian2d/test_symmetricgaussian2d.hdf5")

    def test_fitting(self):
        m = self.s.create_model()
        g = SymmetricGaussian2D()
        m.append(g)
        m.fit()
        model_data = m.as_signal().data
        s_model = m.as_signal()
        residual = (s_model-self.s).sum()
        nt.assert_true(residual < 1)


class TestSymmetricGaussian2DValues:

    def setUp(self):
        g = hs.model.components.SymmetricGaussian2D(
            centre_x=-5.,
            centre_y=-5.,
            sigma_x=1.,
            sigma_y=2.)
        x = np.arange(-10, 10, 0.01)
        y = np.arange(-10, 10, 0.01)
        X, Y = np.meshgrid(x, y)
        gt = g.function(X, Y)
        self.g = g
        self.gt = gt

    def test_values(self):
        gt = self.gt
        g = self.g
        nt.assert_almost_equal(g.fwhm_x, 2.35482004503)
        nt.assert_almost_equal(g.fwhm_y, 4.70964009006)
        nt.assert_almost_equal(gt.max(), 0.0795774715459)
        nt.assert_almost_equal(gt.argmax(axis=0)[0], 500)
        nt.assert_almost_equal(gt.argmax(axis=1)[0], 500)


class TestGaussian2D:

    """Test using a 2-D Gaussian generated with
    Gaussian2D(
        A=1000., sigma_x=1., sigma_y=2., rotation=np.pi/4)"""
    def setUp(self):
        self.s = load(
            my_path +
            "/test_gaussian2d/test_gaussian2d.hdf5")

    def test_fitting(self):
        m = self.s.create_model()
        g = Gaussian2D()
        # Need to set some sensible initial values
        m.append(g)
        m.fit()
        s_model = m.as_signal()
        residual = (s_model-self.s).sum()
        nt.assert_true(residual < 1)
