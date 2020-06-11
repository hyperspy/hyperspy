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

import numpy as np
from hyperspy.components1d import Expression, Gaussian
from hyperspy._signals.signal1d import Signal1D
from hyperspy._components.expression import check_parameter_linearity

class TestModelLinearity:

    def setup_method(self, method):
        np.random.seed(1)
        s = Signal1D(
            np.random.normal(
                scale=2,
                size=10000)).get_histogram()
        self.g = Gaussian()
        m = s.create_model()
        m.append(self.g)
        self.m = m

    def test_model_is_not_linear(self):
        """
        Model is not currently linear as Gaussian sigma and centre parameters
        are free
        """
        assert self.m.nonlinear_parameters
        assert self.m.linear_parameters

    def test_model_is_linear(self):
        self.g.sigma.free = False
        self.g.centre.free = False
        assert len(self.m.nonlinear_parameters) == 2
        assert len(self.m.linear_parameters) == 1

    def test_model_parameters_inactive(self):
        self.g.active = False
        assert not self.m.nonlinear_parameters
        assert not self.m.linear_parameters

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
    assert g.b._is_linear

def test_gaussian_linear():
    g = Gaussian()
    assert g.A._is_linear
    assert not g.centre._is_linear
    assert not g.sigma._is_linear

def test_parameter_linearity():
    expr = "a*x**2 + b*x + c"
    assert check_parameter_linearity(expr, 'a')
    assert check_parameter_linearity(expr, 'b')
    assert check_parameter_linearity(expr, 'c')
    
    expr = "a*sin(b*x)"
    assert check_parameter_linearity(expr, 'a')
    assert not check_parameter_linearity(expr, 'b')

    expr = "a*exp(-b*x)"
    assert check_parameter_linearity(expr, 'a')
    assert not check_parameter_linearity(expr, 'b')