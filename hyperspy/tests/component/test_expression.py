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

import decimal
from fractions import Fraction

import numpy as np
import pytest
import sympy

import hyperspy.api as hs


class TestExpression:
    def setup_method(self, method):
        self.g = hs.model.components1D.Expression(
            expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2)",
            name="Gaussian",
            position="x0",
            height=1,
            fwhm=1,
            x0=0,
            module="numpy",
        )

    def test_name(self):
        assert self.g.name == "Gaussian"

    def test_position(self):
        assert self.g._position is self.g.x0

    def test_f(self):
        assert self.g.function(0) == 1

    def test_grad_height(self):
        np.testing.assert_allclose(self.g.grad_height(2), 1.5258789062500007e-05)

    def test_grad_x0(self):
        np.testing.assert_allclose(self.g.grad_x0(2), 0.00016922538587889289)

    def test_grad_fwhm(self):
        np.testing.assert_allclose(self.g.grad_fwhm(2), 0.00033845077175778578)

    def test_function_nd(self):
        with pytest.raises(RuntimeError):
            self.g.function_nd(0)
        parameters_values = [p.value for p in self.g.parameters]
        assert self.g.function_nd(0, parameters_values=parameters_values) == 1


def test_expression_symbols():
    with pytest.raises(ValueError):
        hs.model.components1D.Expression(expression="10.0", name="offset")
    with pytest.raises(ValueError):
        hs.model.components1D.Expression(expression="10", name="offset")
    with pytest.raises(ValueError):
        hs.model.components1D.Expression(expression="10*offset", name="Offset")


def test_expression_substitution():
    expr = "A / B; A = x+2; B = x-c"
    comp = hs.model.components1D.Expression(expr, name="testcomp", autodoc=True, c=2)
    assert "".join(p.name for p in comp.parameters) == "c"
    assert comp.function(1) == -3


def test_separate_pseudocomponents():
    A = hs.model.components1D.Expression("a*b*x+c**2*x", "test")
    free, fixed = A._separate_pseudocomponents()
    assert list(free.keys()) == ["a", "b", "c"]
    assert list(fixed.keys()) == ["function", "parameters"]

    A.a.free = False
    A.b.free = False

    free, fixed = A._separate_pseudocomponents()
    assert list(free.keys()) == ["c"]


def test_separate_pseudocomponents_expression_rename_parameters():
    lorentzian = hs.model.components1D.Lorentzian()
    free, fixed = lorentzian._separate_pseudocomponents()
    assert list(free.keys()) == ["A", "centre", "gamma"]
    assert list(fixed.keys()) == ["function", "parameters"]

    lorentzian.centre.free = False
    lorentzian.gamma.free = False

    free, fixed = lorentzian._separate_pseudocomponents()
    assert list(free.keys()) == ["A"]


def test_linear_rename_parameters():
    # with the lorentzian component, the gamma component is rename
    lorentzian = hs.model.components1D.Lorentzian()
    assert lorentzian.A._linear
    assert not lorentzian.gamma._linear
    assert not lorentzian.centre._linear

    g = hs.model.components1D.Expression(
        expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2)",
        name="Gaussian",
        rename_pars={"height": "O"},
    )
    assert not hasattr(g, "height")
    assert g.O._linear
    assert not g.fwhm._linear
    assert not g.x0._linear


def test_constant_term_rename_parameters():
    g = hs.model.components1D.Expression(
        expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2) + 10",
        name="Gaussian",
        fwhm=1.0,
    )
    assert g._constant_term == 10.0

    g = hs.model.components1D.Expression(
        expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2) + 10",
        name="Gaussian",
        rename_pars={"height": "O"},
        fwhm=1.0,
    )
    assert g._constant_term == 10.0

    g = hs.model.components1D.Expression(
        expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2) + 10",
        name="Gaussian",
        rename_pars={"x0": "O"},
        fwhm=1.0,
    )
    assert g._constant_term == 10.0


REJECTED_KWARGS = [
    True,
    np.bool_(True),
    float("nan"),
    float("inf"),
    float("-inf"),
    1 + 2j,
    np.complex64(1 + 2j),
    b"x",
    None,
    [1, 2],
    ((1,), (2,)),
    2**1100,
    Fraction(2**1100, 3),
    np.float64("nan"),
    decimal.Decimal("NaN"),
    np.array(True),
    np.array(1 + 2j),
    np.array([1.5, 2.5]),
    sympy.I,
    sympy.Symbol("t"),
]


@pytest.mark.parametrize("value", REJECTED_KWARGS)
def test_expression_kwarg_rejected(value):
    with pytest.raises(ValueError, match="must be a finite real number"):
        hs.model.components1D.Expression("a*x", name="t", a=value)


ACCEPTED_KWARGS = [
    Fraction(1, 2),
    np.int8(3),
    np.uint64(2**63),
    np.float32(1.25),
    np.longdouble(1.25),
    np.array(1.25),
    np.asarray(3),
    2**64,
    decimal.Decimal("1.25"),
    sympy.pi,
]


@pytest.mark.parametrize("value", ACCEPTED_KWARGS)
def test_expression_kwarg_accepted(value):
    comp = hs.model.components1D.Expression("a*x", name="t", a=value)
    assert isinstance(comp.a.value, float)
    np.testing.assert_allclose(comp.a.value, float(value))


def test_expression_zero_dimensional_array_kwarg():
    # np.asarray-wrapped scalars reached ``Parameter.value`` before constructor
    # validation existed and must keep working; the setter still accepts them.
    comp = hs.model.components1D.Expression("a*x", name="t", a=np.array(2.5))
    assert isinstance(comp.a.value, float)
    np.testing.assert_allclose(comp.a.value, 2.5)


def test_expression_shipped_component_decimal_kwarg():
    comp = hs.model.components1D.Gaussian(A=decimal.Decimal("1.25"))
    assert isinstance(comp.A.value, float)
    np.testing.assert_allclose(comp.A.value, 1.25)
    result = float(comp.function(0.3))
    assert np.isfinite(result)
