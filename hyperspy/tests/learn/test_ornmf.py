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

import numpy as np
import pytest

from hyperspy.learn._ornmf import ornmf
from hyperspy.signals import Signal1D


def compare_norms(a, b, tol=5e-3):
    assert a.shape == b.shape

    m, n = a.shape
    tol *= m * n
    n1 = np.linalg.norm(a)
    n2 = np.linalg.norm(b)

    assert np.linalg.norm((a / n1) - (b / n2)) < tol


class TestRNMF:
    def setup_method(self, method):
        # Define shape etc.
        m = 128  # Dimensionality
        n = 256  # Number of samples
        r = 3
        s = 0.01

        rng = np.random.RandomState(101)
        U = rng.uniform(0, 1, (m, r))
        V = rng.uniform(0, 1, (n, r))
        X = U @ V.T
        np.divide(X, max(1.0, np.linalg.norm(X)), out=X)

        E = 100 * rng.binomial(1, s, X.shape)
        Y = X + E

        self.m = m
        self.n = n
        self.rank = r
        self.U = U
        self.V = V
        self.X = X
        self.Y = Y
        self.E = E

    @pytest.mark.parametrize("project", [True, False])
    def test_default(self, project):
        W, H = ornmf(self.X, self.rank, project=project)
        compare_norms(W @ H, self.X)

        assert W.shape == self.U.shape
        assert H.shape == self.V.T.shape

    def test_batch_size(self):
        W, H = ornmf(self.X, self.rank, batch_size=2)
        compare_norms(W @ H, self.X)

        assert W.shape == self.U.shape
        assert H.shape == self.V.T.shape

    def test_store_error(self):
        Xhat, Ehat, W, H = ornmf(self.X, self.rank, store_error=True)
        compare_norms(Xhat, self.X)

        assert Xhat.shape == self.X.shape
        assert Ehat.shape == self.E.shape

    def test_corrupted_default(self):
        W, H = ornmf(self.Y, self.rank)
        compare_norms(W @ H, self.X)

    def test_robust(self):
        W, H = ornmf(self.X, self.rank, method="RobustPGD")
        compare_norms(W @ H, self.X)

    def test_corrupted_robust(self):
        W, H = ornmf(self.Y, self.rank, method="RobustPGD")
        compare_norms(W @ H, self.X)

    def test_no_method(self):
        with pytest.raises(ValueError, match="'method' not recognised"):
            _ = ornmf(self.X, self.rank, method="uniform")

    def test_subspace_tracking(self):
        W, H = ornmf(self.X, self.rank, method="MomentumSGD")
        compare_norms(W @ H, self.X)

    @pytest.mark.parametrize("subspace_learning_rate", [1.0, 1.1])
    def test_subspace_tracking_learning_rate(self, subspace_learning_rate):
        W, H = ornmf(
            self.X,
            self.rank,
            method="MomentumSGD",
            subspace_learning_rate=subspace_learning_rate,
        )
        compare_norms(W @ H, self.X)

    @pytest.mark.parametrize("subspace_momentum", [0.5, 0.9])
    def test_subspace_tracking_momentum(self, subspace_momentum):
        W, H = ornmf(
            self.X, self.rank, method="MomentumSGD", subspace_momentum=subspace_momentum
        )
        compare_norms(W @ H, self.X)

        with pytest.raises(ValueError, match="must be a float between 0 and 1"):
            _ = ornmf(self.X, self.rank, method="MomentumSGD", subspace_momentum=1.9)

    @pytest.mark.parametrize("poisson", [True, False])
    def test_signal(self, poisson):
        # Note that s1.decomposition() operates on the transpose
        # i.e. (n_samples, n_features).
        x = self.Y.T.copy().reshape(16, 16, 128)

        if poisson:
            x -= x.min()
            x[x <= 0] = 1e-16

        s1 = Signal1D(x)

        X_out, E_out = s1.decomposition(
            normalize_poissonian_noise=poisson,
            algorithm="ORNMF",
            output_dimension=self.rank,
            return_info=True,
        )

        # Check the low-rank component MSE
        compare_norms(X_out, self.X.T)


class TestORNMFNegativeMean:
    """Regression tests for PR #3656: ORNMF hang on negative-mean data."""

    def test_setup_with_negative_mean_does_not_produce_nan(self):
        """_setup should produce finite W even when data has negative mean."""
        from hyperspy.learn._ornmf import ORNMF

        rng = np.random.default_rng(42)
        X = rng.random((13, 25)) - 15.0  # negative mean ~ -2.5
        obj = ORNMF(rank=3, random_state=1)
        obj._setup(X)
        assert np.all(np.isfinite(obj.W))
        assert obj.W.shape[1] == 3


class TestORNMFIteratorSetup:
    """Cover the iterator path in _setup for full coverage of #3611 fix."""

    def test_setup_with_iterator_uses_abs(self):
        """_setup should use abs() when X is an iterator (not ndarray)."""
        from hyperspy.learn._ornmf import ORNMF

        rng = np.random.default_rng(42)
        X = rng.random((13, 7))
        # Make mean negative so abs() matters
        X = X - 2.0
        # Pass as generator to trigger iterator path
        obj = ORNMF(rank=3, random_state=1)
        gen = (row for row in X)
        obj._setup(gen)
        assert np.all(np.isfinite(obj.W))
        assert obj.W.shape[1] == 3
