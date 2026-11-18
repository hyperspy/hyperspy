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
import math
from fractions import Fraction

import numpy as np
import pytest
import sympy

from hyperspy.misc._validation import validate_finite_real


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
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
        np.float64("inf"),
        decimal.Decimal("NaN"),
        sympy.I,
        sympy.Symbol("t"),
    ],
)
def test_rejected(value):
    with pytest.raises(ValueError, match="must be a finite real number"):
        validate_finite_real(value, "t")


@pytest.mark.parametrize(
    "value",
    [
        Fraction(1, 2),
        np.int8(3),
        np.uint64(2**63),
        np.float32(1.25),
        np.longdouble(1.25),
        sympy.pi,
    ],
)
def test_accepted(value):
    result = validate_finite_real(value, "t")
    assert isinstance(result, float)
    np.testing.assert_allclose(result, float(value))


def test_overflow_raises_valueerror():
    with pytest.raises(ValueError, match="must be a finite real number"):
        validate_finite_real(2**1100, "t")


def test_nan_rejected():
    with pytest.raises(ValueError, match="must be a finite real number"):
        validate_finite_real(float("nan"), "t")


def test_inf_rejected():
    for v in (float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be a finite real number"):
            validate_finite_real(v, "t")


def test_error_message_contains_name():
    with pytest.raises(ValueError, match="myparam"):
        validate_finite_real([1, 2], "myparam")


def test_valid_float_roundtrip():
    assert validate_finite_real(math.pi, "t") == pytest.approx(math.pi)
