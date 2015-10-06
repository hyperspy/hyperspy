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


class TestModelFolding:

    def setUp(self):
        g = Gaussian()
        s = Spectrum(np.random.random((10, 20, 30)))
        m = create_model(s)
        m.append(g)
        self.model = m

    def test_unfold(self):
        m = self.model
        m.unfold()
        shape = m.spectrum.axes_manager.navigation_shape[::-1]
        assert_true(m.axes_manager.navigation_shape == shape[::-1])
        assert_true(m[0].A.map.shape == shape)
        assert_true(m[0].centre.map.shape == shape)
        assert_true(m[0].sigma.map.shape == shape)
        assert_true(m[0]._axes_manager.navigation_shape[::-1] == shape)
        assert_true(m.chisq.data.shape == shape)
        assert_true(m.dof.data.shape == shape)

    def test_fold(self):
        m = self.model
        shape = m.spectrum.axes_manager.navigation_shape[::-1]
        m.unfold()
        m.fold()
        assert_true(m.axes_manager.navigation_shape[::-1] == shape)
        assert_true(m[0].A.map.shape == shape)
        assert_true(m[0].centre.map.shape == shape)
        assert_true(m[0].sigma.map.shape == shape)
        assert_true(m[0]._axes_manager.navigation_shape[::-1] == shape)
        assert_true(m.chisq.data.shape == shape)
        assert_true(m.dof.data.shape == shape)
