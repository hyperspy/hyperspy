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
import scipy.linalg

from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.learn.rpca import orpca, rpca_godec
from hyperspy.signals import Signal1D


def compare_norms(a, b, tol=5e-3):
    assert a.shape == b.shape

    m, n = a.shape
    tol *= m * n
    n1 = np.linalg.norm(a)
    n2 = np.linalg.norm(b)

    assert np.linalg.norm((a / n1) - (b / n2)) < tol


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
        A = U @ V.T
        E = 10 * rng.binomial(1, s, (m, n))
        G = 0.005 * rng.randn(m, n)
        X = A + E + G

        self.m = m
        self.n = n
        self.rank = r
        self.A = A
        self.X = X

    def test_default(self):
        X, E, U, S, V = rpca_godec(self.X, rank=self.rank)
        compare_norms(X, self.A)

    @pytest.mark.parametrize("power", [0, 1, 2])
    @pytest.mark.parametrize("maxiter", [1e2, 1e3])
    @pytest.mark.parametrize("tol", [1e-1, 1e-4])
    def test_tol_iter(self, power, maxiter, tol):
        X, E, U, S, V = rpca_godec(
            self.X, rank=self.rank, power=power, maxiter=maxiter, tol=tol
        )
        compare_norms(X, self.A)

    def test_regularization(self):
        X, E, U, S, V = rpca_godec(self.X, rank=self.rank, lambda1=0.01)
        compare_norms(X, self.A)

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
            algorithm="RPCA",
            output_dimension=self.rank,
            return_info=True,
        )
        compare_norms(X, self.A.T)


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
        A = U @ V.T
        E = 10 * rng.binomial(1, s, (m, n))
        X = A + E

        self.m = m
        self.n = n
        self.rank = r
        self.A = A
        self.X = X
        self.U = U

    def test_default(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank, store_error=True)
        compare_norms(X, self.A)

    def test_project(self):
        L, R = orpca(self.X, rank=self.rank, project=True)

        assert L.shape == (self.m, self.rank)
        assert R.shape == (self.rank, self.n)

    def test_batch_size(self):
        L, R = orpca(self.X, rank=self.rank, batch_size=2)

        assert L.shape == (self.m, self.rank)
        assert R.shape == (self.rank, self.n)

    def test_method_BCD(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank, store_error=True, method="BCD")
        compare_norms(X, self.A)

    @pytest.mark.parametrize("subspace_learning_rate", [1.0, 1.1])
    def test_method_SGD(self, subspace_learning_rate):
        X, E, U, S, V = orpca(
            self.X,
            rank=self.rank,
            store_error=True,
            method="SGD",
            subspace_learning_rate=subspace_learning_rate,
        )
        compare_norms(X, self.A)

    @pytest.mark.parametrize("subspace_momentum", [0.5, 0.1])
    def test_method_MomentumSGD(self, subspace_momentum):
        X, E, U, S, V = orpca(
            self.X,
            rank=self.rank,
            store_error=True,
            method="MomentumSGD",
            subspace_learning_rate=1.1,
            subspace_momentum=subspace_momentum,
        )
        compare_norms(X, self.A)

        with pytest.raises(ValueError, match=f"must be a float between 0 and 1"):
            _ = orpca(
                self.X, rank=self.rank, method="MomentumSGD", subspace_momentum=1.9
            )

    def test_init_rand(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank, store_error=True, init="rand")
        compare_norms(X, self.A)

    def test_init_mat(self):
        X, E, U, S, V = orpca(self.X, rank=self.rank, store_error=True, init=self.U)
        compare_norms(X, self.A)

        with pytest.raises(ValueError, match=f"has to be a two-dimensional matrix"):
            mat = np.zeros(self.m)
            _ = orpca(self.X, rank=self.rank, init=mat)

        with pytest.raises(ValueError, match=f"has to be of shape"):
            mat = np.zeros((self.m, self.rank - 1))
            _ = orpca(self.X, rank=self.rank, init=mat)

    @pytest.mark.parametrize("rank", [3, 11])
    @pytest.mark.parametrize("training_samples", [16, 32])
    def test_training(self, rank, training_samples):
        X, E, U, S, V = orpca(
            self.X,
            rank=rank,
            store_error=True,
            init="qr",
            training_samples=training_samples,
        )
        compare_norms(X, self.A)

        with pytest.raises(ValueError, match=f"must be >="):
            _ = orpca(self.X, rank=self.rank, init="qr", training_samples=self.rank - 1)

    def test_regularization(self):
        X, E, U, S, V = orpca(
            self.X, rank=self.rank, store_error=True, lambda1=0.01, lambda2=0.02,
        )
        compare_norms(X, self.A)

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
            algorithm="ORPCA",
            output_dimension=self.rank,
            return_info=True,
        )
        compare_norms(X, self.A.T)
