# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

from hyperspy._signals.signal1d import Signal1D
from hyperspy.components1d import Expression


class TestLinearFitting:
    def setup_method(self, method):
        self.s = Signal1D(np.zeros((4, 5, 20)))
        self.m = self.s.create_model()

    def test_constant_term_without_model(self):
        expression = "a * x + b"
        g = Expression(expression, name="test_constant", a=20.0, b=4.0)
        assert g._constant_term == 0
        self.m.append(g)
        assert g._constant_term == 0

    def test_constant_from_expression(self):
        expression = "a * x + b"
        g = Expression(expression, name="test_constant", a=20.0, b=4.0)
        g.b.free = False
        assert g._constant_term == 4.0
        self.m.append(g)
        assert g._constant_term == 4.0

    def test_constant_from_expression2(self):
        expression = "A * exp(-(x-centre)**2/(2*sigma**2))"
        h = Expression(expression, name="test_constant2", A=20.0, centre=4.0, sigma=1.0)
        self.m.append(h)

        assert h._constant_term == 0
        h.centre.free = False
        h.sigma.free = False
        assert h._constant_term == 0
