# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np

import hyperspy.api as hs


def test_function():
    g = hs.model.components1D.HeavisideStep()
    g.A.value = 3
    g.n.value = 2
    assert g.function(1) == 0.0
    assert g.function(2) == 1.5
    assert g.function(3) == 3.0


class TestHeavisideStep:
    def setup_method(self, method):
        self.c = hs.model.components1D.HeavisideStep()

    def test_integer_values(self):
        c = self.c
        np.testing.assert_array_almost_equal(
            c.function(np.array([-1, 0, 2])), np.array([0, 0.5, 1])
        )

    def test_float_values(self):
        c = self.c
        np.testing.assert_array_almost_equal(
            c.function(np.array([-0.5, 0.5, 2])), np.array([0, 1, 1])
        )

    def test_not_sorted(self):
        c = self.c
        np.testing.assert_array_almost_equal(
            c.function(np.array([3, -0.1, 0])), np.array([1, 0, 0.5])
        )

    # def test_gradients(self):
    #     c = self.c
    #     np.testing.assert_array_almost_equal(
    #         c.A.grad(np.array([3, -0.1, 0])), np.array([1, 0, 0.5])
    #     )
    #     np.testing.assert_array_almost_equal(
    #         c.n.grad(np.array([3, -0.1, 0])), np.array([1, 1, 1])
    #     )
