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

from hyperspy.components1d import Expression, Gaussian
from hyperspy._signals.signal1d import Signal1D
import numpy as np
class TestLinearFitting:
    def setup_method(self, method):
        self.s = Signal1D(np.zeros((4,5, 20)))
        self.m = self.s.create_model()

    def test_constant_from_expression(self):
        expression = "a * x + b"
        g = Expression(
            expression, 
            name="test_constant",
            a = 20.0,
            b = 4.0)
        self.m.append(g)
        g.b.free = False
        assert g.get_constant_term() == 4.0
        g.b.assign_current_value_to_all()
        assert np.all(g.get_constant_term(multi=True) == 4*np.ones(self.s.axes_manager._navigation_shape_in_array))

    def test_constant_from_expression2(self):
        expression = "A * exp(-(x-centre)**2/(2*sigma**2))"
        h = Expression(
            expression, 
            name="test_constant2",
            A = 20.0,
            centre = 4.0,
            sigma = 1.0)
        self.m.append(h)
        assert h.get_constant_term() == 0
        h.centre.free = False
        h.sigma.free = False
        assert h.get_constant_term() == 0