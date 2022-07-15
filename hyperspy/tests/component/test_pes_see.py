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

import hyperspy.api as hs
from hyperspy.components1d import SEE
from hyperspy.misc.test_utils import ignore_warning


def test_see():
    see = SEE(A=10, Phi=1.5, B=0.5)
    x = np.linspace(-5, 15, 10)
    np.testing.assert_allclose(
        see.function(x),
        np.array([0.0, 0.0, 0.0, 8.4375, 0.342983001, 0.0675685032,
                  0.0236279967, 0.010861538, 0.005860978, 0.003514161])
        )
    np.testing.assert_allclose(see.function(x), see.function_nd(x))

    _ = SEE(A=10, Phi=1.5, B=0.5, sigma=0)


def test_see_fit():
    # Component parameter values
    A = 10
    Phi = 2.5
    B = 0.5

    offset, scale, size = 0, 0.1, 100
    x = np.linspace(offset, scale*size, size)
    s = hs.signals.Signal1D(SEE(A=A, Phi=Phi, B=B).function(x))
    axis = s.axes_manager[0]
    axis.offset, axis.scale = offset, scale
    s.add_gaussian_noise(0.1, random_state=1)
    m = s.create_model()
    see = SEE(A=1, Phi=1.5, B=0.5)
    m.append(see)
    with ignore_warning(message="divide by zero",
                        category=RuntimeWarning):
        m.fit(grad='analytical')
    np.testing.assert_allclose(see.A.value, A, rtol=0.1)
    np.testing.assert_allclose(see.Phi.value, Phi, rtol=0.1)
    np.testing.assert_allclose(see.B.value, B, rtol=0.1)


def test_see_function_nd():
    A, Phi, B = 10, 1.5, 0.5
    see = SEE(A=A, Phi=Phi, B=B)
    x = np.linspace(-5, 15, 10)
    s = hs.signals.Signal1D(np.array([x]*2))

    # Manually set to test function_nd
    see._axes_manager = s.axes_manager
    see._create_arrays()
    see.A.map['values'] = [A] * 2
    see.Phi.map['values'] = [Phi] * 2
    see.B.map['values'] = [B] * 2

    values = see.function_nd(x)
    assert values.shape == (2, 10)
    for v in values:
        np.testing.assert_allclose(v, see.function(x))
