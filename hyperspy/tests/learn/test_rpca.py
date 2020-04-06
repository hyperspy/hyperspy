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
import scipy.linalg
import pytest

from hyperspy.signals import Signal1D
from hyperspy.learn.rpca import rpca_godec, orpca
from hyperspy.exceptions import VisibleDeprecationWarning


class TestRPCA:
    def setup_method(self, method):
        # Define shape etc.
        m = 128  # Dimensionality
        n = 256  # Number of samples
        r = 3
        s = 0.01

        # Low-rank, sparse and noise matrices
        rng = np.random.RandomState(101)
        U = scipy.linalg.orth(rng.randn(m, r))
        V = rng.randn(n, r)
        A = np.dot(U, V.T)
        E = 10 * rng.binomial(1, s, (m, n))
        G = 0.005 * rng.randn(m, n)
        X = A + E + G

        self.m = m
        self.n = n
        self.rank = r
        self.lambda1 = 0.01
        self.A = A
        self.X = X

        # Test tolerance
        self.tol = 5e-3 * (self.m * self.n)

    def test_default(self):
        X, E, G, U, S, V = rpca_godec(self.X, rank=self.rank)

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        assert normX < self.tol

    @pytest.mark.parametrize("power", [None, 1, 2])
    @pytest.mark.parametrize("maxiter", [None, 1e2, 1e3])
    @pytest.mark.parametrize("tol", [None, 1e-1, 1e-4])
    def test_tol_iter(self, power, maxiter, tol):
        X, E, G, U, S, V = rpca_godec(
            self.X, rank=self.rank, power=power, maxiter=maxiter, tol=tol
        )

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        assert normX < self.tol

    def test_regularization(self):
        X, E, G, U, S, V = rpca_godec(self.X, rank=self.rank, lambda1=self.lambda1)

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        assert normX < self.tol

    @pytest.mark.parametrize("poisson", [True, False])
    def test_signal(self, poisson):
        # Note that s1.decomposition() operates on the transpose
        # i.e. (n_samples, n_features).
        x = self.X.copy().T.reshape(16, 16, 128)

        if poisson:
            x -= x.min()
            x[x <= 0] = 1e-16

        s1 = Signal1D(x)

        X, E = s1.decomposition(
            normalize_poissonian_noise=poisson,
            algorithm="rpca",
            output_dimension=self.rank,
            return_info=True,
        )

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A.T)
        assert normX < self.tol


class TestORPCA:
    def setup_method(self, method):
        # Define shape etc.
        m = 128  # Dimensionality
        n = 256  # Number of samples
        r = 3
        s = 0.01

        # Low-rank and sparse error matrices
        rng = np.random.RandomState(101)
        U = scipy.linalg.orth(rng.randn(m, r))
        V = rng.randn(n, r)
        A = np.dot(U, V.T)
        E = 10 * rng.binomial(1, s, (m, n))
        X = A + E

        self.m = m
        self.n = n
        self.rank = r
        self.lambda1 = 1.0 / np.sqrt(n)
        self.lambda2 = 1.0 / np.sqrt(n)
        self.A = A
        self.X = X
        self.subspace_learning_rate = 1.1
        self.training_samples = 32
        self.subspace_momentum = 0.1

        # Test tolerance
        self.tol = 5e-3 * (self.m * self.n)

    def test_default(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank)

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        assert normX < self.tol

    def test_project(self):
        L, R = orpca(self.X, rank=self.rank, project=True)

        # Check shape is double (i.e. equivalent to
        # running over the data twice)
        assert L.shape == (self.m, self.rank)
        assert R.shape == (self.rank, self.n)

    def test_method_BCD(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank, method="BCD")

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        assert normX < self.tol

    @pytest.mark.parametrize("subspace_learning_rate", [None, 1.1])
    def test_method_SGD(self, subspace_learning_rate):
        X, E, U, S, V = orpca(
            self.X,
            rank=self.rank,
            method="SGD",
            subspace_learning_rate=subspace_learning_rate,
        )

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        assert normX < self.tol

    @pytest.mark.parametrize("subspace_momentum", [None, 0.1])
    def test_method_MomentumSGD(self, subspace_momentum):
        X, E, U, S, V = orpca(
            self.X,
            rank=self.rank,
            method="MomentumSGD",
            subspace_learning_rate=self.subspace_learning_rate,
            subspace_momentum=subspace_momentum,
        )

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        assert normX < self.tol

        with pytest.raises(ValueError, match=f"must be a float between 0 and 1"):
            _ = orpca(
                self.X, rank=self.rank, method="MomentumSGD", subspace_momentum=1.9
            )

    def test_init(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank, init="rand")

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        assert normX < self.tol

        with pytest.raises(ValueError, match=f"has to be a two-dimensional matrix"):
            mat = np.zeros(self.m)
            _ = orpca(self.X, rank=self.rank, init=mat)

        with pytest.raises(ValueError, match=f"has to be of shape"):
            mat = np.zeros((self.m, self.rank - 1))
            _ = orpca(self.X, rank=self.rank, init=mat)

    def test_training(self):
        X, E, U, S, V = orpca(
            self.X, rank=self.rank, init="qr", training_samples=self.training_samples
        )

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        print(normX)
        assert normX < self.tol

        with pytest.raises(ValueError, match=f"must be >="):
            _ = orpca(self.X, rank=self.rank, init="qr", training_samples=self.rank - 1)

    def test_regularization(self):
        X, E, U, S, V = orpca(
            self.X, rank=self.rank, lambda1=self.lambda1, lambda2=self.lambda2
        )

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A)
        assert normX < self.tol

    def test_exception_method(self):
        with pytest.raises(ValueError, match=f"'method' not recognised"):
            _ = orpca(self.X, rank=self.rank, method="uniform")

    def test_exception_init(self):
        with pytest.raises(ValueError, match=f"'init' not recognised"):
            _ = orpca(self.X, rank=self.rank, init="uniform")

    def test_warnings(self):
        with pytest.warns(
            VisibleDeprecationWarning,
            match=f"The argument `learning_rate` has been deprecated",
        ):
            _ = orpca(self.X, rank=self.rank, learning_rate=0.1)

        with pytest.warns(
            VisibleDeprecationWarning,
            match=f"The argument `momentum` has been deprecated",
        ):
            _ = orpca(self.X, rank=self.rank, momentum=0.1)

    @pytest.mark.parametrize("poisson", [True, False])
    def test_signal(self, poisson):
        # Note that s1.decomposition() operates on the transpose
        # i.e. (n_samples, n_features).
        x = self.X.copy().T.reshape(16, 16, 128)

        if poisson:
            x -= x.min()
            x[x <= 0] = 1e-16

        s1 = Signal1D(x)

        X, E = s1.decomposition(
            normalize_poissonian_noise=poisson,
            algorithm="orpca",
            output_dimension=self.rank,
            return_info=True,
        )

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.A.T)
        assert normX < self.tol
