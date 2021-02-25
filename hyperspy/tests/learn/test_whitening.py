# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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
import pytest

from hyperspy.learn.whitening import whiten_data


@pytest.mark.parametrize("method", ["PCA", "ZCA"])
@pytest.mark.parametrize("centre", [True, False])
def test_whiten(method, centre):
    rng = np.random.RandomState(123)
    m, n = 500, 4

    X = rng.randn(m, n)

    Y, W = whiten_data(X, centre=centre, method=method)
    cov = Y.T @ Y
    cov *= n / np.trace(cov)

    # Y.T @ Y should be approximately an identity matrix
    np.testing.assert_allclose(cov, np.eye(n), atol=1e-6)


def test_whiten_error():
    rng = np.random.RandomState(123)
    m, n = 500, 4

    X = rng.randn(m, n)

    with pytest.raises(ValueError, match="method must be one of"):
        Y, W = whiten_data(X, method="uniform")
