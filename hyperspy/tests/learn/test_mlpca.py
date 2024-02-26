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

from hyperspy.learn.mlpca import mlpca
from hyperspy.signals import Signal1D


@pytest.mark.parametrize("tol", [1e-9, 1e-6])
@pytest.mark.parametrize("max_iter", [100, 500])
def test_mlpca(tol, max_iter):
    # Define shape etc.
    m = 100  # Dimensionality
    n = 101  # Number of samples
    r = 3

    rng = np.random.RandomState(101)
    U = rng.uniform(0, 1, size=(m, r))
    V = rng.uniform(0, 10, size=(n, r))
    varX = U @ V.T
    X = rng.poisson(varX)
    rank = r

    # Test tolerance
    tol = 300

    U, S, V, Sobj = mlpca(X, varX, output_dimension=rank, tol=tol, max_iter=max_iter)
    X = U @ np.diag(S) @ V.T

    # Check the low-rank component MSE
    normX = np.linalg.norm(X - X)
    assert normX < tol

    # Check singular values
    S_norm = S / np.sum(S)
    np.testing.assert_allclose(S_norm[:rank].sum(), 1.0)


def test_signal():
    # Define shape etc.
    m = 100  # Dimensionality
    n = 101  # Number of samples
    r = 3

    rng = np.random.RandomState(101)
    U = rng.uniform(0, 1, size=(m, r))
    V = rng.uniform(0, 10, size=(n, r))
    varX = U @ V.T
    X = rng.poisson(varX).astype(float)

    # Test tolerance
    tol = 300

    x = X.copy().reshape(10, 10, 101)
    s = Signal1D(x)
    s.decomposition(algorithm="MLPCA", output_dimension=r)

    # Check singular values
    v = s.get_explained_variance_ratio().data
    np.testing.assert_allclose(v[:r].sum(), 1.0)

    # Check the low-rank component MSE
    Y = s.get_decomposition_model(r).data
    normX = np.linalg.norm(Y.reshape(m, n) - X)
    assert normX < tol
