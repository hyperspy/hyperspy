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
from hyperspy.datasets import example_signals

def test_independent_from_expression():
    expression = "a * x + b"
    g = Expression(
        expression, 
        name="test_constant",
        a = 20.0,
        b = 4.0)
    assert g.independent_term == 4.0
    g.b.free = False

def test_independent_from_expression2():
    expression = "A * exp(-(x-centre)**2/(2*sigma**2))"
    g = Expression(
        expression, 
        name="test_constant2",
        A = 20.0,
        centre = 4.0,
        sigma = 1.0)
    assert g.independent_term == 0

def test_independent_from_expression3():
    expression = "a * b * x + 10**(c/d) + e"
    g = Expression(
        expression, 
        name="test_constant",
        a = 20.0,
        b = 4.0,
        c = 10.0,
        d = 2.0,
        e = 20.0)
    assert g.independent_term == 10**5 + 20