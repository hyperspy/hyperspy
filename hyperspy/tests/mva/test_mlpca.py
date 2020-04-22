# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed
from hyperspy.learn.mlpca import mlpca


class TestMLPCA:
    def setup_method(self, method):
        # Define shape etc.
        m = 100  # Dimensionality
        n = 101  # Number of samples
        r = 3

        rng = np.random.RandomState(101)
        U = rng.uniform(0, 1, size=(m, r))
        V = rng.uniform(0, 10, size=(n, r))
        varX = U @ V.T
        X = rng.poisson(varX)
        self.m = m
        self.n = n
        self.rank = r
        self.X = X
        self.varX = varX

        # Test tolerance
        self.tol = 270

    @pytest.mark.parametrize("tol", [1e-9, 1e-6])
    @pytest.mark.parametrize("max_iter", [100, 500])
    def test_mlpca(self, tol, max_iter):
        U, S, V, Sobj = mlpca(
            self.X, self.varX, output_dimension=self.rank, tol=tol, max_iter=max_iter
        )
        X = U @ np.diag(S) @ V.T

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.X)
        assert normX < self.tol

        # Check singular values
        S_norm = S / np.sum(S)
        np.testing.assert_allclose(S_norm[:self.rank].sum(), 1.0)


    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize("tol", [1e-9, 1e-6])
    @pytest.mark.parametrize("max_iter", [100, 500])
    def test_mlpca_fast(self, tol, max_iter):
        U, S, V, Sobj = mlpca(
            self.X,
            self.varX,
            output_dimension=self.rank,
            tol=tol,
            max_iter=max_iter,
            fast=True,
        )
        X = U @ np.diag(S) @ V.T

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.X)
        assert normX < self.tol

        # Check singular values
        S_norm = S / np.sum(S)
        np.testing.assert_allclose(S_norm[:self.rank].sum(), 1.0)
