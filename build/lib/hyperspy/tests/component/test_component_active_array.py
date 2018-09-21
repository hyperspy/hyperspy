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
from hyperspy.components1d import Gaussian
from hyperspy.signals import Signal1D
from hyperspy.misc.utils import stash_active_state


class TestParametersAsSignals:

    def setup_method(self, method):
        self.gaussian = Gaussian()
        self.gaussian._axes_manager = Signal1D(
            np.zeros((3, 3, 1))).axes_manager

    def test_always_active(self):
        g = self.gaussian
        g.active_is_multidimensional = False
        g._create_arrays()
        np.testing.assert_array_equal(g.A.as_signal('values').data,
                                      np.zeros((3, 3)))

    def test_some_inactive(self):
        g = self.gaussian
        g.active_is_multidimensional = True
        g._create_arrays()
        g._active_array[2, 0] = False
        g._active_array[0, 0] = False
        assert np.isnan(g.A.as_signal('values').data[[0, 2], [0]]).all()

    def test_stash_array(self):
        g = self.gaussian
        g.active_is_multidimensional = True
        g._create_arrays()
        g._active_array[2, 0] = False
        g._active_array[0, 0] = False
        with stash_active_state([g]):
            g.active_is_multidimensional = False
            assert not g._active_is_multidimensional
            np.testing.assert_array_equal(g.A.as_signal('values').data,
                                          np.zeros((3, 3)))
            assert g._active_array is None
        assert g._active_is_multidimensional
        np.testing.assert_allclose(
            g._active_array, np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]],
                                      dtype=bool))
        assert np.isnan(g.A.as_signal('values').data[[0, 2], [0]]).all()
