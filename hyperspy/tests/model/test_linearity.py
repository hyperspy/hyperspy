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
import pytest

from hyperspy._components.expression import _check_parameter_linearity
from hyperspy._signals.signal1d import Signal1D
from hyperspy.components1d import Expression, Gaussian


class TestModelLinearity:
    def setup_method(self, method):
        np.random.seed(1)
        s = Signal1D(np.random.normal(scale=2, size=10000)).get_histogram()
        self.g = Gaussian()
        m = s.create_model()
        m.append(self.g)
        self.m = m

    def test_model_is_not_linear(self):
        """
        Model is not currently linear as Gaussian sigma and centre parameters
        are free
        """
        nonlinear_parameters = [
            p for c in self.m for p in c.parameters if not p._linear
        ]
        assert len(nonlinear_parameters) > 0

    def test_model_linear(self):
        self.g.sigma.free = False
        self.g.centre.free = False
        nonlinear_parameters = [
            p for c in self.m for p in c.parameters if not p._linear
        ]
        assert len(nonlinear_parameters) == 2
        _list = [p for p in nonlinear_parameters if p in self.m._free_parameters]
        assert len(_list) == 0

    def test_model_parameters_inactive(self):
        self.g.active = False
        nonlinear_parameters = [
            p for c in self.m for p in c.parameters if not p._linear
        ]
        assert len(nonlinear_parameters) == 2
        _list = [p for p in nonlinear_parameters if p in self.m._free_parameters]
        assert len(_list) == 0

    def test_model_parameters_set_inactive(self):
        self.m.set_component_active_value(False, [self.g])
        nonlinear_parameters = [
            p for c in self.m for p in c.parameters if not p._linear
        ]
        assert len(nonlinear_parameters) == 2
        _list = [p for p in nonlinear_parameters if p in self.m._free_parameters]
        assert len(_list) == 0


def test_sympy_linear_expression():
    expression = "height * exp(-(x - centre) ** 2 * 4 * log(2)/ fwhm ** 2)"
    g = Expression(expression, name="Test_function")
    assert g.height._linear
    assert not g.centre._linear
    assert not g.fwhm._linear


def test_sympy_linear_expression2():
    expression = "a * x + b"
    g = Expression(expression, name="Test_function2")
    assert g.a._linear
    assert g.b._linear


def test_gaussian_linear():
    g = Gaussian()
    assert g.A._linear
    assert not g.centre._linear
    assert not g.sigma._linear


def test_parameter_linearity():
    expr = "a*x**2 + b*x + c"
    assert _check_parameter_linearity(expr, "a")
    assert _check_parameter_linearity(expr, "b")
    assert _check_parameter_linearity(expr, "c")

    expr = "a*sin(b*x)"
    assert _check_parameter_linearity(expr, "a")
    assert not _check_parameter_linearity(expr, "b")

    expr = "a*exp(-b*x)"
    assert _check_parameter_linearity(expr, "a")
    assert not _check_parameter_linearity(expr, "b")

    expr = "where(x > 10, a*sin(b*x), 0)"
    with pytest.warns(UserWarning):
        _check_parameter_linearity(expr, "a")
    with pytest.warns(UserWarning):
        _check_parameter_linearity(expr, "b")
