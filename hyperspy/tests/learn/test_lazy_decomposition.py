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

from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.misc.machine_learning.import_sklearn import sklearn_installed
from hyperspy.signals import Signal1D


class TestLazyDecomposition:
    def setup_method(self, method):
        # Define shape etc.
        m = 100  # Dimensionality
        n = 128  # Number of samples
        r = 3

        self.rng = np.random.RandomState(101)
        U = self.rng.randn(m, r)
        V = self.rng.randn(n, r)
        X = U @ V.T
        X = np.exp(0.1 * X / np.linalg.norm(X))

        self.m = m
        self.n = n
        self.rank = r
        self.X = X
        self.s = Signal1D(
            X.copy().reshape(int(np.sqrt(m)), int(np.sqrt(m)), n)
        ).as_lazy()

        # Test tolerance
        self.tol = 1e-2 * (self.m * self.n)

    @pytest.mark.parametrize("output_dimension", [None, 3])
    @pytest.mark.parametrize("normalize_poissonian_noise", [True, False])
    def test_svd(self, output_dimension, normalize_poissonian_noise):
        self.s.decomposition(
            output_dimension=output_dimension,
            normalize_poissonian_noise=normalize_poissonian_noise,
        )
        factors = self.s.learning_results.factors
        loadings = self.s.learning_results.loadings

        if hasattr(factors, "compute"):
            factors = factors.compute()
        if hasattr(loadings, "compute"):
            loadings = loadings.compute()

        explained_variance = self.s.learning_results.explained_variance
        X = loadings @ factors.T

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.X)
        assert normX < self.tol

        # Check singular values
        explained_variance_norm = explained_variance / np.sum(explained_variance)
        np.testing.assert_allclose(
            explained_variance_norm[: self.rank].sum(), 1.0, atol=1e-6
        )

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize("normalize_poissonian_noise", [True, False])
    def test_pca(self, normalize_poissonian_noise):
        self.s.decomposition(
            output_dimension=3,
            algorithm="PCA",
            normalize_poissonian_noise=normalize_poissonian_noise,
        )
        factors = self.s.learning_results.factors
        loadings = self.s.learning_results.loadings

        if hasattr(factors, "compute"):
            factors = factors.compute()
        if hasattr(loadings, "compute"):
            loadings = loadings.compute()

        explained_variance = self.s.learning_results.explained_variance
        X = loadings @ factors.T

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.X)
        assert normX < self.tol

        # Check singular values
        explained_variance_norm = explained_variance / np.sum(explained_variance)
        np.testing.assert_allclose(
            explained_variance_norm[: self.rank].sum(), 1.0, atol=1e-6
        )

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_pca_mask(self):
        s = self.s
        sig_mask = (s.inav[0, 0].data < 1.0).compute()

        s.decomposition(output_dimension=3,
                        algorithm="PCA",
                        signal_mask=sig_mask)
        factors = s.learning_results.factors
        loadings = s.learning_results.loadings
        _ = loadings @ factors.T

        # Check singular values
        explained_variance = s.learning_results.explained_variance
        explained_variance_norm = explained_variance / np.sum(explained_variance)
        np.testing.assert_allclose(
            explained_variance_norm[: self.rank].sum(), 1.0, atol=1e-6
        )

        nav_mask = (s.isig[0].data < 1.0).compute()

        s.decomposition(output_dimension=3,
                        algorithm="PCA",
                        navigation_mask=nav_mask)
        factors = s.learning_results.factors
        loadings = s.learning_results.loadings
        _ = loadings @ factors.T

        # Check singular values
        explained_variance = s.learning_results.explained_variance
        explained_variance_norm = explained_variance / np.sum(explained_variance)
        np.testing.assert_allclose(
            explained_variance_norm[: self.rank].sum(), 1.0, atol=1e-6
        )


    @pytest.mark.parametrize("normalize_poissonian_noise", [True, False])
    def test_orpca(self, normalize_poissonian_noise):
        self.s.decomposition(
            output_dimension=3,
            algorithm="ORPCA",
            normalize_poissonian_noise=normalize_poissonian_noise,
        )
        factors = self.s.learning_results.factors
        loadings = self.s.learning_results.loadings

        if hasattr(factors, "compute"):
            factors = factors.compute()
        if hasattr(loadings, "compute"):
            loadings = loadings.compute()

        explained_variance = self.s.learning_results.explained_variance
        X = loadings @ factors.T

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.X)
        assert normX < self.tol

        # Check singular values
        assert explained_variance is None

    @pytest.mark.parametrize("normalize_poissonian_noise", [True, False])
    def test_ornmf(self, normalize_poissonian_noise):
        self.s.decomposition(
            output_dimension=3,
            algorithm="ORNMF",
            normalize_poissonian_noise=normalize_poissonian_noise,
        )
        factors = self.s.learning_results.factors
        loadings = self.s.learning_results.loadings

        if hasattr(factors, "compute"):
            factors = factors.compute()
        if hasattr(loadings, "compute"):
            loadings = loadings.compute()

        explained_variance = self.s.learning_results.explained_variance
        X = loadings @ factors.T

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.X)
        assert normX < self.tol

        # Check singular values
        assert explained_variance is None

    def test_output_dimension_error(self):
        with pytest.raises(ValueError, match="`output_dimension` must be specified"):
            self.s.decomposition(algorithm="ORPCA")

    def test_algorithm_error(self):
        with pytest.raises(ValueError, match="'algorithm' not recognised"):
            self.s.decomposition(algorithm="random")

    def test_bounds_warning(self):
        with pytest.warns(
            VisibleDeprecationWarning, match="`bounds` keyword is deprecated"
        ):
            self.s.decomposition(bounds=True)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize("algorithm", ["ONMF"])
    def test_deprecated_algorithms_warning(self, algorithm):
        with pytest.warns(
            VisibleDeprecationWarning,
            match="`algorithm='{}'` has been deprecated".format(algorithm),
        ):
            self.s.decomposition(output_dimension=3, algorithm=algorithm)


class TestPrintInfo:
    def setup_method(self, method):
        rng = np.random.RandomState(123)
        self.s = Signal1D(rng.random_sample(size=(20, 100))).as_lazy()

    @pytest.mark.parametrize("algorithm", ["SVD", "ORPCA", "ORNMF"])
    def test_decomposition(self, algorithm, capfd):
        self.s.decomposition(algorithm=algorithm, output_dimension=3)
        captured = capfd.readouterr()
        assert "Decomposition info:" in captured.out

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    @pytest.mark.parametrize("algorithm", ["PCA"])
    def test_decomposition_sklearn(self, capfd, algorithm):
        self.s.decomposition(algorithm=algorithm, output_dimension=3)
        captured = capfd.readouterr()
        assert "Decomposition info:" in captured.out
        assert "scikit-learn estimator:" in captured.out

    @pytest.mark.parametrize("algorithm", ["SVD"])
    def test_no_print(self, algorithm, capfd):
        self.s.decomposition(algorithm=algorithm, output_dimension=2, print_info=False)
        captured = capfd.readouterr()
        assert "Decomposition info:" not in captured.out

    def test_decomposition_mask_SVD(self):
        s = self.s
        sig_mask = (s.inav[0].data < 0.5).compute()
        with pytest.raises(NotImplementedError):
            s.decomposition(algorithm="SVD", signal_mask=sig_mask)

        nav_mask = (s.isig[0].data < 0.5).compute()
        with pytest.raises(NotImplementedError):
            s.decomposition(algorithm="SVD", navigation_mask=nav_mask)

    @pytest.mark.skipif(not sklearn_installed, reason="sklearn not installed")
    def test_decomposition_mask_wrong_Shape(self):
        s = self.s
        sig_mask = (s.inav[0].data < 0.5).compute()[:-2]
        with pytest.raises(ValueError):
            s.decomposition(algorithm='PCA', signal_mask=sig_mask)

        nav_mask = (s.isig[0].data < 0.5).compute()[:-2]
        with pytest.raises(ValueError):
            s.decomposition(algorithm='PCA', navigation_mask=nav_mask)
