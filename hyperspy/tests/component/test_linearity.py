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

def test_sympy_linear_expression():
    expression = "height * exp(-(x - centre) ** 2 * 4 * log(2)/ fwhm ** 2)"
    g = Expression(expression, name="Test_function")
    assert g.height._is_linear
    assert not g.centre._is_linear
    assert not g.fwhm._is_linear

def test_sympy_linear_expression2():
    expression = "a * x + b"
    g = Expression(expression, name="Test_function2")
    assert g.a._is_linear
    assert not g.b._is_linear

def test_sympy_linear_expression3():
    expression = "a * b * x + c"
    g = Expression(expression, name="Test_function3")
    assert g.a._is_linear
    assert g.b._is_linear
    assert not g.c._is_linear
    assert not g.is_linear

    g.a.free = False
    g.c.free = False
    assert g.is_linear

    g.b.free = False
    g.c.free = True
    assert not g.is_linear

    g.c.free = False
    assert g.is_linear

def test_sympy_linear_expression4():
    expression = "x+c"
    g = Expression(expression, name="Test_function3")
    assert not g.c._is_linear
    assert not g.is_linear

    
def test_gaussian_linear():
    g = Gaussian()
    assert g.A._is_linear
    assert not g.centre._is_linear
    assert not g.sigma._is_linear
    assert not g.is_linear

    g.sigma.free = False
    g.centre.free = False
    assert g.is_linear