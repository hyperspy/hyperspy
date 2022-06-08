# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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
            module="numpy")

    def test_name(self):
        assert self.g.name == "Gaussian"

    def test_position(self):
        assert self.g._position is self.g.x0

    def test_f(self):
        assert self.g.function(0) == 1

    def test_grad_height(self):
        np.testing.assert_allclose(
            self.g.grad_height(2),
            1.5258789062500007e-05)

    def test_grad_x0(self):
        np.testing.assert_allclose(
            self.g.grad_x0(2),
            0.00016922538587889289)

    def test_grad_fwhm(self):
        np.testing.assert_allclose(
            self.g.grad_fwhm(2),
            0.00033845077175778578)

    def test_function_nd(self):
        assert self.g.function_nd(0) == 1


def test_expression_symbols():
    with pytest.raises(ValueError):
        hs.model.components1D.Expression(expression="10.0", name="offset")
    with pytest.raises(ValueError):
        hs.model.components1D.Expression(expression="10", name="offset")
    with pytest.raises(ValueError):
        hs.model.components1D.Expression(expression="10*offset", name="Offset")


def test_expression_substitution():
    expr = 'A / B; A = x+2; B = x-c'
    comp = hs.model.components1D.Expression(expr, name='testcomp',
                                            autodoc=True,
                                            c=2)
    assert ''.join(p.name for p in comp.parameters) == 'c'
    assert comp.function(1) == -3


def test_separate_pseudocomponents():
    A = hs.model.components1D.Expression("a*b*x+c**2*x", "test")
    free, fixed = A._separate_pseudocomponents()
    assert list(free.keys()) == ['a', 'b', 'c']
    assert list(fixed.keys()) == ['function', 'parameters']

    A.a.free = False
    A.b.free = False

    free, fixed = A._separate_pseudocomponents()
    assert list(free.keys()) == ['c']


def test_separate_pseudocomponents_expression_rename_parameters():
    l = hs.model.components1D.Lorentzian()
    free, fixed = l._separate_pseudocomponents()
    assert list(free.keys()) == ['A', 'centre', 'gamma']
    assert list(fixed.keys()) == ['function', 'parameters']

    l.centre.free = False
    l.gamma.free = False

    free, fixed = l._separate_pseudocomponents()
    assert list(free.keys()) == ['A']
    

def test_linear_rename_parameters():
    # with the lorentzian component, the gamma component is rename
    l = hs.model.components1D.Lorentzian()
    assert l.A._linear
    assert not l.gamma._linear
    assert not l.centre._linear

    g = hs.model.components1D.Expression(
            expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2)",
            name="Gaussian",
            rename_pars={'height':'O'}
            )
    assert not hasattr(g, 'height')
    assert g.O._linear
    assert not g.fwhm._linear
    assert not g.x0._linear
    

def test_constant_term_rename_parameters():
    g = hs.model.components1D.Expression(
            expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2) + 10",
            name="Gaussian",
            fwhm=1.0,
            )
    assert g._constant_term == 10.

    g = hs.model.components1D.Expression(
            expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2) + 10",
            name="Gaussian",
            rename_pars={'height':'O'},
            fwhm=1.0,
            )
    assert g._constant_term == 10.

    g = hs.model.components1D.Expression(
            expression="height * exp(-(x - x0) ** 2 * 4 * log(2)/ fwhm ** 2) + 10",
            name="Gaussian",
            rename_pars={'x0':'O'},
            fwhm=1.0,
            )
    assert g._constant_term == 10.