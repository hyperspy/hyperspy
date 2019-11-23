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
from numpy.testing import assert_allclose
import pytest
from distutils.version import LooseVersion
import sympy

from hyperspy.components1d import SkewNormal
from hyperspy.signals import Signal1D


pytestmark = pytest.mark.skipif(LooseVersion(sympy.__version__) <
                                LooseVersion("1.3"),
                                reason="This test requires SymPy >= 1.3")


def test_function():
    g = SkewNormal()
    g.A.value = 2.5
    g.x0.value = 0
    g.scale.value = 1
    g.shape.value = 0
    assert_allclose(g.function(0), 1, rtol=3e-3)
    assert_allclose(g.function(6), 1.52e-8, rtol=1e-3)
    g.A.value = 5
    g.x0.value = 4
    g.scale.value = 3
    g.shape.value = 2
    assert_allclose(g.function(0), 6.28e-3, rtol=1e-3)
    assert_allclose(g.function(g.mean), 2.855, rtol=1e-3)


def test_fit(A=1, x0=0, shape=1, scale=1, noise=0.01):
    """
    Creates a simulated noisy skew normal distribution based on the input
    parameters and fits a skew normal component to this data.
    """
    # create skew normal signal and add noise
    g = SkewNormal(A=A, x0=x0, scale=scale, shape=shape)
    x = np.arange(x0 - scale * 3, x0 + scale * 3, step=0.01 * scale)
    s = Signal1D(g.function(x))
    s.axes_manager.signal_axes[0].axis = x
    s.add_gaussian_noise(std=noise * A)
    # fit skew normal component to signal
    g2 = SkewNormal()
    m = s.create_model()
    m.append(g2)
    g2.x0.bmin = x0 - scale * 3  # prevent parameters to run away
    g2.x0.bmax = x0 + scale * 3
    g2.x0.bounded = True
    m.fit(bounded=True)
    m.print_current_values()  # print out parameter values
    m.plot()  # plot fit
    return m


def test_util_mean_get():
    g1 = SkewNormal()
    g1.shape.value = 0
    g1.scale.value = 1
    g1.x0.value = 1
    assert_allclose(g1.mean, 1.0)


def test_util_variance_get():
    g1 = SkewNormal()
    g1.shape.value = 0
    g1.scale.value = 2
    g1.x0.value = 1
    assert_allclose(g1.variance, 4.0)


def test_util_skewness_get():
    g1 = SkewNormal()
    g1.shape.value = 30
    g1.scale.value = 1
    g1.x0.value = 1
    assert_allclose(g1.skewness, 0.99, rtol=1e-3)


def test_util_mode_get():
    g1 = SkewNormal()
    g1.shape.value = 0
    g1.scale.value = 1
    g1.x0.value = 1
    assert_allclose(g1.mode, 1.0)
