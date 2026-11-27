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

"""Validation helpers for numeric values passed at public boundaries."""

import decimal
import math
import numbers

import numpy as np


def _safe_repr(value):
    # ``repr`` itself can fail: Python 3.11+ refuses to stringify integers
    # above ``sys.get_int_max_str_digits()``, and a user object may raise from
    # ``__repr__``. Never let building the diagnostic replace the validation
    # error being reported.
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__} value not representable>"


def _error_message(name, value):
    return (
        f"The '{name}' parameter must be a finite real number, got "
        f"{_safe_repr(value)} of type {type(value).__name__}."
    )


def validate_finite_real(value, name):
    """
    Validate that ``value`` is a finite real number and return it as ``float``.

    Accepts instances of ``numbers.Rational``, ``numbers.Real``, ``decimal.Decimal``
    and numpy integer/floating scalars (excluding ``bool`` and ``np.bool_``), plus
    finite SymPy number-like expressions (``Expr`` with ``is_number`` truthy, e.g.
    ``sympy.pi``). Zero-dimensional real numeric arrays (e.g. ``np.asarray(2.5)``)
    are accepted as the scalar they wrap. Everything else is rejected with a
    ``ValueError`` naming the parameter.

    This module must not import from ``hyperspy.signals`` or ``hyperspy.misc.utils``
    (import-cycle hazard: ``hyperspy.misc.utils`` is a deprecation shim that pulls
    in ``hyperspy.signals``).

    Parameters
    ----------
    value :
        The value to validate.
    name : str
        Parameter name used in the error message.

    Returns
    -------
    float
        The converted value.

    """
    # ``bool`` is a subclass of ``int`` (and ``numbers.Real``), so it must be
    # rejected explicitly and before the generic ``numbers`` checks below.
    # ``np.bool_`` is not a ``numbers.Real`` and is rejected via ``np.integer`` /
    # ``np.floating`` not covering it.
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_error_message(name, value))
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # 0-d numeric arrays reach these boundaries from ``np.asarray`` and from
        # values loaded out of NumPy files; they wrap a scalar and were accepted
        # through the ``CFloat`` coercion that predates constructor validation.
        # ``np.bool_`` is not a ``np.number`` subtype, so boolean arrays are
        # rejected here too, as are complex and non-numeric dtypes.
        if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
            value.dtype, np.complexfloating
        ):
            raise ValueError(_error_message(name, value))
    elif not isinstance(
        value,
        (
            numbers.Rational,
            numbers.Real,
            decimal.Decimal,
            np.integer,
            np.floating,
        ),
    ):
        # Deferred import on purpose: sympy is an optional dependency for this
        # boundary and eager import would slow down every HyperSpy import.
        import sympy

        if not (isinstance(value, sympy.core.expr.Expr) and value.is_number):
            raise ValueError(_error_message(name, value))

    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError):
        raise ValueError(_error_message(name, value)) from None
    if not math.isfinite(result):
        raise ValueError(_error_message(name, value))
    return result
