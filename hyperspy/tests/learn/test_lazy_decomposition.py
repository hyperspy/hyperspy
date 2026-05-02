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

import importlib

import dask.array as da
import numpy as np
import pytest

from hyperspy.signals import Signal1D

sklearn = importlib.util.find_spec("sklearn")
skip_sklearn = pytest.mark.skipif(sklearn is None, reason="sklearn not installed")

# Suppress the svd_solver-default-change DeprecationWarning in all tests except
# the dedicated deprecation warning tests.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The default svd_solver for algorithm='SVD':DeprecationWarning"
)


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

    @skip_sklearn
    @pytest.mark.parametrize("normalize_poissonian_noise", [True, False])
    def test_svd(self, normalize_poissonian_noise):
        self.s.decomposition(
            output_dimension=3,
            normalize_poissonian_noise=normalize_poissonian_noise,
        )
        factors = self.s.learning_results.factors
        loadings = self.s.learning_results.loadings

        if isinstance(factors, da.Array):
            factors = factors.compute()
        if isinstance(loadings, da.Array):
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

    @skip_sklearn
    @pytest.mark.parametrize("normalize_poissonian_noise", [True, False])
    def test_pca(self, normalize_poissonian_noise):
        self.s.decomposition(
            output_dimension=3,
            algorithm="PCA",
            normalize_poissonian_noise=normalize_poissonian_noise,
        )
        factors = self.s.learning_results.factors
        loadings = self.s.learning_results.loadings

        if isinstance(factors, da.Array):
            factors = factors.compute()
        if isinstance(loadings, da.Array):
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

    @skip_sklearn
    def test_pca_mask(self):
        s = self.s
        sig_mask = (s.inav[0, 0].data < 1.0).compute()

        s.decomposition(output_dimension=3, algorithm="PCA", signal_mask=sig_mask)
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

        s.decomposition(output_dimension=3, algorithm="PCA", navigation_mask=nav_mask)
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

        if isinstance(factors, da.Array):
            factors = factors.compute()
        if isinstance(loadings, da.Array):
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

        if isinstance(factors, da.Array):
            factors = factors.compute()
        if isinstance(loadings, da.Array):
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
        with pytest.raises(ValueError, match="`output_dimension` must be specified"):
            self.s.decomposition(algorithm="SVD", svd_solver="incremental")

    @skip_sklearn
    @pytest.mark.parametrize("centre", ["navigation", "signal"])
    def test_svd_centre(self, centre):
        self.s.decomposition(output_dimension=3, centre=centre)

        assert self.s.learning_results.centre == centre
        assert self.s.learning_results.mean is not None

    @skip_sklearn
    def test_svd_no_centering(self):
        self.s.decomposition(output_dimension=3, centre=None)

        assert self.s.learning_results.centre is None
        assert self.s.learning_results.mean is None

    @skip_sklearn
    def test_svd_centre_invalid(self):
        with pytest.raises(ValueError, match="`centre` must be"):
            self.s.decomposition(output_dimension=3, centre="invalid")

    @skip_sklearn
    def test_svd_mask(self):
        """SVD with signal mask runs without error and produces results."""
        s = self.s
        sig_mask = (s.inav[0, 0].data < 1.0).compute()
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            signal_mask=sig_mask,
        )
        assert s.learning_results.factors is not None
        assert s.learning_results.loadings is not None

    def test_algorithm_error(self):
        with pytest.raises(ValueError, match="not recognised"):
            self.s.decomposition(algorithm="random")

    def test_svd_default_solver_deprecation_warning(self):
        """algorithm='SVD' without svd_solver warns that the default will change."""
        with pytest.warns(
            DeprecationWarning,
            match="The default svd_solver for algorithm='SVD'",
        ):
            self.s.decomposition(algorithm="SVD", output_dimension=3)

    @skip_sklearn
    def test_isvd_alias_deprecation_warning(self):
        """algorithm='ISVD' (deprecated alias) raises DeprecationWarning."""
        with pytest.warns(
            DeprecationWarning,
            match="algorithm='ISVD' is deprecated",
        ):
            self.s.decomposition(algorithm="ISVD", output_dimension=3)

    def test_dasksvd_alias_deprecation_warning(self):
        """algorithm='DaskSVD' (deprecated alias) raises DeprecationWarning."""
        with pytest.warns(
            DeprecationWarning,
            match="algorithm='DaskSVD' is deprecated",
        ):
            self.s.decomposition(algorithm="DaskSVD")


class TestPrintInfo:
    def setup_method(self, method):
        rng = np.random.default_rng(123)
        self.s = Signal1D(rng.random(size=(20, 100))).as_lazy()

    @pytest.mark.parametrize(
        "algorithm,svd_solver",
        [("SVD", "incremental"), ("ORPCA", None), ("ORNMF", None)],
    )
    def test_decomposition(self, algorithm, svd_solver, capfd):
        self.s.decomposition(
            algorithm=algorithm, svd_solver=svd_solver, output_dimension=3
        )
        captured = capfd.readouterr()
        assert "Decomposition info:" in captured.out

    @skip_sklearn
    @pytest.mark.parametrize("algorithm", ["PCA"])
    def test_decomposition_sklearn(self, capfd, algorithm):
        self.s.decomposition(algorithm=algorithm, output_dimension=3)
        captured = capfd.readouterr()
        assert "Decomposition info:" in captured.out
        assert "scikit-learn estimator:" in captured.out

    @pytest.mark.parametrize("algorithm,svd_solver", [("SVD", "incremental")])
    def test_no_print(self, algorithm, svd_solver, capfd):
        self.s.decomposition(
            algorithm=algorithm,
            svd_solver=svd_solver,
            output_dimension=2,
            print_info=False,
        )
        captured = capfd.readouterr()
        assert "Decomposition info:" not in captured.out

    @skip_sklearn
    def test_decomposition_mask_SVD(self):
        """SVD masking is now supported; check shapes are correct."""
        s = self.s
        sig_mask = (s.inav[0].data < 0.5).compute()
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=2,
            signal_mask=sig_mask,
        )
        assert s.learning_results.factors is not None

        nav_mask = (s.isig[0].data < 0.5).compute()
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=2,
            navigation_mask=nav_mask,
        )
        assert s.learning_results.loadings is not None

    @skip_sklearn
    def test_decomposition_mask_wrong_Shape(self):
        s = self.s
        sig_mask = (s.inav[0].data < 0.5).compute()[:-2]
        with pytest.raises(ValueError):
            s.decomposition(algorithm="PCA", signal_mask=sig_mask)

        nav_mask = (s.isig[0].data < 0.5).compute()[:-2]
        with pytest.raises(ValueError):
            s.decomposition(algorithm="PCA", navigation_mask=nav_mask)


class TestNormalizePoissonianNoise:
    """Tests for LazySignal.normalize_poissonian_noise()."""

    def setup_method(self, method):
        rng = np.random.default_rng(42)
        # Poisson-like data: positive integers, shape (10 nav, 20 sig)
        self.data = rng.integers(1, 100, size=(10, 20)).astype(float)
        self.s = Signal1D(self.data.copy()).as_lazy()

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_scaling_no_mask(self):
        """Scaled data matches manual Keenan-Kotula formula."""
        data = self.data
        aG = data.sum(axis=1)  # sum over signal axis -> (10,)
        bH = data.sum(axis=0)  # sum over nav axis   -> (20,)
        expected = data / (np.sqrt(aG)[:, None] * np.sqrt(bH)[None, :])

        self.s.normalize_poissonian_noise()
        result = self.s.data.compute()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_root_attributes_stored(self):
        """_root_aG and _root_bH are stored as dask arrays after call."""
        s = self.s
        s.normalize_poissonian_noise()
        assert hasattr(s, "_root_aG")
        assert hasattr(s, "_root_bH")
        assert isinstance(s._root_aG, da.Array)
        assert isinstance(s._root_bH, da.Array)
        assert s._root_aG.shape == (10,)
        assert s._root_bH.shape == (20,)

    # ------------------------------------------------------------------
    # Mask support
    # ------------------------------------------------------------------

    def test_scaling_with_signal_mask(self):
        """Signal mask excludes channels from bH computation."""
        data = self.data
        sig_mask = np.zeros(20, dtype=bool)
        sig_mask[0] = True  # mask out first channel

        s = Signal1D(data.copy()).as_lazy()
        s.normalize_poissonian_noise(signal_mask=sig_mask)

        # Manual: zero the masked channel before summing
        masked = data.copy()
        masked[:, sig_mask] = 0.0
        aG = masked.sum(axis=1)
        bH = masked.sum(axis=0)
        aG = np.where(aG == 0, 1, aG)
        bH = np.where(bH == 0, 1, bH)
        expected = data / (np.sqrt(aG)[:, None] * np.sqrt(bH)[None, :])
        # Masked positions are left unscaled (original values)
        expected[:, sig_mask] = data[:, sig_mask]

        result = s.data.compute()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_scaling_with_navigation_mask(self):
        """Navigation mask excludes positions from aG computation."""
        data = self.data
        nav_mask = np.zeros(10, dtype=bool)
        nav_mask[0] = True  # mask out first nav position

        s = Signal1D(data.copy()).as_lazy()
        s.normalize_poissonian_noise(navigation_mask=nav_mask)

        masked = data.copy()
        masked[nav_mask, :] = 0.0
        aG = masked.sum(axis=1)
        bH = masked.sum(axis=0)
        aG = np.where(aG == 0, 1, aG)
        bH = np.where(bH == 0, 1, bH)
        expected = data / (np.sqrt(aG)[:, None] * np.sqrt(bH)[None, :])
        # Masked positions are left unscaled (original values)
        expected[nav_mask, :] = data[nav_mask, :]

        result = s.data.compute()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    # ------------------------------------------------------------------
    # Guard conditions
    # ------------------------------------------------------------------

    def test_negative_values_raise(self):
        """ValueError if unmasked data contains negative values."""
        data = self.data.copy()
        data[0, 0] = -1.0
        s = Signal1D(data).as_lazy()
        with pytest.raises(ValueError, match="Negative values"):
            s.normalize_poissonian_noise()

    def test_all_masked_raises(self):
        """ValueError if the entire array is masked."""
        nav_mask = np.ones(10, dtype=bool)  # mask every nav position
        with pytest.raises(ValueError, match="All the data are masked"):
            self.s.normalize_poissonian_noise(navigation_mask=nav_mask)

    # ------------------------------------------------------------------
    # Integration with decomposition()
    # ------------------------------------------------------------------

    @skip_sklearn
    def test_decomposition_centre_guard(self):
        """decomposition() raises if both normalize_poissonian_noise and centre are set."""
        with pytest.raises(ValueError, match="normalize_poissonian_noise"):
            self.s.decomposition(
                normalize_poissonian_noise=True,
                centre="navigation",
                output_dimension=2,
            )

    @skip_sklearn
    def test_decomposition_rescales_back(self):
        """factors/loadings are rescaled back to original data space after SVD."""
        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            output_dimension=2,
            normalize_poissonian_noise=True,
            print_info=False,
        )
        factors = s.learning_results.factors  # (20, 2)
        loadings = s.learning_results.loadings  # (10, 2)
        reconstruction = loadings @ factors.T  # (10, 20)
        # Loose check: reconstruction is in the original data space
        assert reconstruction.min() > -1e3
        assert reconstruction.max() < 1e5


class TestLazyDecompositionParityFixes:
    """Tests for parity with the non-lazy MVA.decomposition(svd_solver="incremental") (fixes 1-7).

    Uses a small Signal1D with 2-D navigation so that masks are non-trivial.
    Shape: nav (4, 5) = 20 positions, signal 30 channels.
    """

    def setup_method(self, method):
        rng = np.random.default_rng(7)
        self.s = Signal1D(rng.random((20, 30))).as_lazy()
        # 1-D navigation of size 20; mask out first 4 positions
        nav_arr = np.zeros(20, dtype=bool)
        nav_arr[:4] = True
        self.nav_mask = nav_arr
        # Mask out first 3 signal channels
        sig_arr = np.zeros(30, dtype=bool)
        sig_arr[:3] = True
        self.sig_mask = sig_arr

    # ------------------------------------------------------------------
    # Fix 1: poissonian_noise_normalized stored in LearningResults
    # ------------------------------------------------------------------

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_poissonian_flag_stored_true(self, algorithm, svd_solver):
        """poissonian_noise_normalized is True when normalisation was applied."""
        s = Signal1D(np.abs(self.s.data.compute()) + 1).as_lazy()
        s.decomposition(
            algorithm=algorithm,
            output_dimension=2,
            normalize_poissonian_noise=True,
            print_info=False,
        )
        assert s.learning_results.poissonian_noise_normalized is True

    @skip_sklearn
    def test_poissonian_flag_stored_false(self):
        """poissonian_noise_normalized is False when normalisation was not applied."""
        self.s.decomposition(output_dimension=2, print_info=False)

        assert self.s.learning_results.poissonian_noise_normalized is False

    # ------------------------------------------------------------------
    # Fix 2: number_significant_components stored (elbow estimate)
    # ------------------------------------------------------------------

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_number_significant_components(self, algorithm, svd_solver):
        """number_significant_components is a plain Python int after decomposition."""
        self.s.decomposition(
            algorithm=algorithm,
            svd_solver=svd_solver,
            output_dimension=5,
            print_info=False,
        )
        nsc = self.s.learning_results.number_significant_components
        assert isinstance(nsc, int)
        assert 1 <= nsc <= 5

    def test_number_significant_components_none_without_variance(self):
        """number_significant_components is None for algorithms without variance."""
        self.s.decomposition(algorithm="ORPCA", output_dimension=3, print_info=False)
        assert self.s.learning_results.number_significant_components is None

    # ------------------------------------------------------------------
    # Fix 3: navigation_mask and signal_mask stored in LearningResults
    # ------------------------------------------------------------------

    @skip_sklearn
    def test_navigation_mask_stored(self):
        """navigation_mask is stored as an array on LearningResults."""
        self.s.decomposition(
            output_dimension=2,
            navigation_mask=self.nav_mask,
            print_info=False,
        )
        t = self.s.learning_results
        assert t.navigation_mask is not None
        assert t.navigation_mask.shape == (20,)  # _navigation_shape_in_array order
        assert t.navigation_mask.sum() == self.nav_mask.sum()

    @skip_sklearn
    def test_signal_mask_stored(self):
        """signal_mask is stored as an array on LearningResults."""
        self.s.decomposition(
            output_dimension=2,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        t = self.s.learning_results
        assert t.signal_mask is not None
        assert t.signal_mask.shape == (30,)
        np.testing.assert_array_equal(t.signal_mask, self.sig_mask)

    # ------------------------------------------------------------------
    # Fix 4: NaN-fill excluded positions in factors / loadings
    # ------------------------------------------------------------------

    @skip_sklearn
    def test_nan_fill_loadings_navigation_mask(self):
        """Masked nav positions become NaN rows in loadings (no reproject)."""
        n_components = 2
        self.s.decomposition(
            output_dimension=n_components,
            navigation_mask=self.nav_mask,
            print_info=False,
        )
        loadings = self.s.learning_results.loadings
        # loadings shape should be (nav_size, n_components) = (20, 2)
        assert loadings.shape == (20, n_components)
        flat_mask = self.nav_mask.ravel()
        # Masked positions (True) should be NaN
        assert np.all(np.isnan(loadings[flat_mask, :]))
        # Unmasked positions should not be NaN
        assert not np.any(np.isnan(loadings[~flat_mask, :]))

    @skip_sklearn
    def test_nan_fill_factors_signal_mask(self):
        """Masked signal channels become NaN rows in factors (no reproject)."""
        n_components = 2
        self.s.decomposition(
            output_dimension=n_components,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        factors = self.s.learning_results.factors
        # factors shape should be (sig_size, n_components) = (30, 2)
        assert factors.shape == (30, n_components)
        # Masked channels (first 3) should be NaN
        assert np.all(np.isnan(factors[: self.sig_mask.sum(), :]))
        # Unmasked channels should not be NaN
        assert not np.any(np.isnan(factors[self.sig_mask.sum() :, :]))

    # ------------------------------------------------------------------
    # Fix 5: reproject as string enum
    # ------------------------------------------------------------------

    @skip_sklearn
    def test_reproject_invalid_raises(self):
        """Invalid reproject value raises ValueError."""
        with pytest.raises(ValueError, match="`reproject` must be"):
            self.s.decomposition(
                output_dimension=2, reproject="invalid", print_info=False
            )

    @skip_sklearn
    @pytest.mark.parametrize("algorithm", ["PCA", "ORPCA", "ORNMF"])
    def test_reproject_navigation_full_loadings(self, algorithm):
        """reproject='navigation' returns full (unmasked) loadings without NaN."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=2,
            navigation_mask=self.nav_mask,
            reproject="navigation",
            print_info=False,
        )
        loadings = self.s.learning_results.loadings
        assert loadings.shape == (20, 2)
        assert not np.any(np.isnan(loadings))

    @skip_sklearn
    def test_reproject_both_signal_fills_factors(self):
        """reproject='both' fills masked signal channels in factors (no NaN)."""
        self.s.decomposition(
            algorithm="PCA",
            output_dimension=2,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="both",
            print_info=False,
        )
        factors = self.s.learning_results.factors
        assert factors.shape[0] == 30  # full signal size
        assert not np.any(np.isnan(factors))

    @skip_sklearn
    def test_reproject_signal_fills_factors(self):
        """reproject='signal' produces factors with no NaN at masked channels."""
        self.s.decomposition(
            algorithm="PCA",
            output_dimension=2,
            signal_mask=self.sig_mask,
            reproject="signal",
            print_info=False,
        )
        factors = self.s.learning_results.factors
        assert factors.shape[0] == 30  # full signal size
        assert not np.any(np.isnan(factors))

    # ------------------------------------------------------------------
    # Fix 6: mean stored for PCA
    # ------------------------------------------------------------------

    @skip_sklearn
    def test_mean_stored_for_pca(self):
        """mean is a 1-D array of signal size after PCA."""
        self.s.decomposition(algorithm="PCA", output_dimension=2, print_info=False)
        mean = self.s.learning_results.mean
        assert mean is not None
        assert mean.shape == (30,)

    def test_mean_none_for_orpca(self):
        """mean is None after ORPCA (algorithm does not compute it)."""
        self.s.decomposition(algorithm="ORPCA", output_dimension=2, print_info=False)
        assert self.s.learning_results.mean is None

    def test_mean_none_for_ornmf(self):
        """mean is None after ORNMF (algorithm does not compute it)."""
        self.s.decomposition(algorithm="ORNMF", output_dimension=2, print_info=False)
        assert self.s.learning_results.mean is None

    # ------------------------------------------------------------------
    # Fix 7: return_info
    # ------------------------------------------------------------------

    @skip_sklearn
    def test_return_info_true_pca(self):
        """return_info=True returns the fitted sklearn IncrementalPCA object."""
        import sklearn.decomposition

        obj = self.s.decomposition(
            algorithm="PCA", output_dimension=2, return_info=True, print_info=False
        )
        assert isinstance(obj, sklearn.decomposition.IncrementalPCA)
        assert hasattr(obj, "components_")

    @skip_sklearn
    def test_return_info_false(self):
        """return_info=False (default) returns None."""
        result = self.s.decomposition(
            algorithm="PCA", output_dimension=2, return_info=False, print_info=False
        )
        assert result is None

    @skip_sklearn
    def test_return_info_svd_returns_none(self):
        """return_info=True with SVD returns None (no persistent estimator object)."""
        result = self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=2,
            return_info=True,
            print_info=False,
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive lazy mask × reproject tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_lazy_lowrank(nav=20, sig=100, rank=3, seed=11):
    """Return a lazy rank-*rank* Signal1D and its raw data array.

    The data is non-negative (abs of a Gaussian low-rank product) so that
    it is also valid input for NMF-based algorithms (ORNMF).
    """
    rng = np.random.default_rng(seed)
    U = np.abs(rng.standard_normal((nav, rank)))
    V = np.abs(rng.standard_normal((sig, rank)))
    data = U @ V.T
    return Signal1D(data.copy()).as_lazy(), data


def _nav_mask_1d(nav=20, step=4):
    m = np.zeros(nav, dtype=bool)
    m[::step] = True
    return m


def _sig_mask_1d(sig=100, step=10):
    m = np.zeros(sig, dtype=bool)
    m[::step] = True
    return m


class TestLazyDecompositionBothMasks:
    """Both navigation and signal masks applied simultaneously on lazy signals.

    All three lazy algorithms (SVD, PCA, ORPCA) are tested.  Checks:
    - Correct shape of factors/loadings (full data dimensions)
    - NaN placed only at the masked positions
    - Reconstruction quality on the unmasked region (SVD and PCA only)
    """

    def setup_method(self, method):
        self.s, self.data = _make_lazy_lowrank()
        self.nav_mask = _nav_mask_1d()
        self.sig_mask = _sig_mask_1d()

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_both_masks_nan_pattern(self, algorithm, svd_solver):
        """Nav-masked → NaN loadings rows; sig-masked → NaN factor rows."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        t = self.s.learning_results
        assert t.loadings.shape == (20, 3)
        assert t.factors.shape == (100, 3)
        assert np.all(np.isnan(t.loadings[self.nav_mask, :]))
        assert not np.any(np.isnan(t.loadings[~self.nav_mask, :]))
        assert np.all(np.isnan(t.factors[self.sig_mask, :]))
        assert not np.any(np.isnan(t.factors[~self.sig_mask, :]))

    @pytest.mark.parametrize("algorithm", ["ORPCA", "ORNMF"])
    def test_both_masks_nan_pattern_online(self, algorithm):
        """Online algorithms (ORPCA/ORNMF) also produce correct NaN patterns."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        t = self.s.learning_results
        assert np.all(np.isnan(t.loadings[self.nav_mask, :]))
        assert not np.any(np.isnan(t.loadings[~self.nav_mask, :]))
        assert np.all(np.isnan(t.factors[self.sig_mask, :]))
        assert not np.any(np.isnan(t.factors[~self.sig_mask, :]))

    @skip_sklearn
    def test_both_masks_reconstruction_quality(self):
        """Unmasked region reconstructed near-exactly by SVD for a rank-3 signal."""
        # Only SVD gives an exact rank-k factorisation; IncrementalPCA is
        # approximate and does not guarantee 1e-10 accuracy.
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        kept_nav = ~self.nav_mask
        kept_sig = ~self.sig_mask
        f = self.s.learning_results.factors[kept_sig, :]
        l_ = self.s.learning_results.loadings[kept_nav, :]
        rms = np.sqrt(np.mean((l_ @ f.T - self.data[kept_nav][:, kept_sig]) ** 2))
        assert rms < 1e-10

    @skip_sklearn
    def test_masks_stored_on_learning_results(self):
        """Both masks are stored in LearningResults after decomposition."""
        self.s.decomposition(
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        t = self.s.learning_results
        assert t.navigation_mask is not None
        assert t.signal_mask is not None
        assert t.navigation_mask.shape == (20,)
        assert t.signal_mask.shape == (100,)


class TestLazyDecompositionReprojectionNumerical:
    """Numerical tests for the reproject parameter on lazy signals.

    Verifies that reprojection correctly fills masked positions and
    that the reconstructed data is accurate for low-rank signals.
    """

    def setup_method(self, method):
        self.s, self.data = _make_lazy_lowrank()
        self.nav_mask = _nav_mask_1d()
        self.sig_mask = _sig_mask_1d()

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver",
        [("SVD", "incremental"), ("PCA", None), ("ORPCA", None), ("ORNMF", None)],
    )
    def test_reproject_navigation_no_nan(self, algorithm, svd_solver):
        """reproject='navigation' → full loadings, no NaN, correct shape."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="navigation",
            print_info=False,
        )
        loadings = self.s.learning_results.loadings
        assert loadings.shape == (20, 3)
        assert not np.any(np.isnan(loadings))

    def test_reproject_navigation_reconstruction(self):
        """Reprojected loadings × factors reconstruct the full data (rank-3).

        Only SVD gives exact reconstruction; PCA (incremental) is approximate.
        """
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="navigation",
            print_info=False,
        )
        t = self.s.learning_results
        recon = t.loadings @ t.factors.T
        rms = np.sqrt(np.mean((recon - self.data) ** 2))
        assert rms < 1e-10

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_reproject_navigation_unmasked_rows_unchanged(self, algorithm, svd_solver):
        """reproject='navigation' does not alter the unmasked rows of loadings."""
        # Baseline: no reproject, unmasked positions only
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            print_info=False,
        )
        baseline_loadings = self.s.learning_results.loadings[~self.nav_mask, :].copy()

        # With reproject: should give the same values at unmasked positions
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="navigation",
            print_info=False,
        )
        reproj_loadings = self.s.learning_results.loadings[~self.nav_mask, :]
        np.testing.assert_allclose(baseline_loadings, reproj_loadings, atol=1e-10)

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver",
        [("SVD", "incremental"), ("PCA", None), ("ORPCA", None), ("ORNMF", None)],
    )
    def test_reproject_both_nav_loadings_filled(self, algorithm, svd_solver):
        """reproject='both' fills nav-masked positions (signal reproject warns)."""
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.s.decomposition(
                algorithm=algorithm,
                output_dimension=3,
                navigation_mask=self.nav_mask,
                signal_mask=self.sig_mask,
                reproject="both",
                print_info=False,
            )
        loadings = self.s.learning_results.loadings
        assert loadings.shape == (20, 3)
        assert not np.any(np.isnan(loadings))

    def test_reproject_navigation_with_both_masks_reconstruction(self):
        """With both masks + reproject='navigation', full data reconstructed.

        Only SVD gives exact reconstruction; PCA (incremental) is approximate.
        """
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="navigation",
            print_info=False,
        )
        t = self.s.learning_results
        # Factors still have NaN at masked signal channels; only check
        # that the unmasked-signal reconstruction is near-exact.
        kept_sig = ~self.sig_mask
        f = t.factors[kept_sig, :]
        recon = t.loadings @ f.T
        rms = np.sqrt(np.mean((recon - self.data[:, kept_sig]) ** 2))
        assert rms < 1e-10

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_reproject_signal_fills_factors(self, algorithm, svd_solver):
        """reproject='signal' → factors fully filled (no NaN), loadings still
        have NaN at nav-masked positions."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="signal",
            print_info=False,
        )
        t = self.s.learning_results
        # Factors must cover the full signal (sig_size rows, no NaN)
        assert t.factors.shape[0] == self.data.shape[1]
        assert not np.any(np.isnan(t.factors))
        # Loadings must still have NaN at nav-masked positions
        assert t.loadings.shape[0] == self.data.shape[0]
        assert np.any(np.isnan(t.loadings[self.nav_mask, :]))

    def test_reproject_signal_reconstruction(self):
        """reproject='signal' SVD gives exact reconstruction at unmasked nav
        positions over the full signal."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="signal",
            print_info=False,
        )
        t = self.s.learning_results
        kept_nav = ~self.nav_mask
        # Loadings at unmasked nav rows × full factors must reconstruct data
        recon = t.loadings[kept_nav, :] @ t.factors.T
        rms = np.sqrt(np.mean((recon - self.data[kept_nav]) ** 2))
        assert rms < 1e-10, f"reproject='signal' RMS {rms:.2e} too large"

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_reproject_signal_unmasked_channels_unchanged(self, algorithm, svd_solver):
        """reproject='signal' does not alter the unmasked channel rows of
        factors (compared to no-reproject baseline)."""
        kw = dict(
            algorithm=algorithm,
            output_dimension=3,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        # Baseline: no reproject
        self.s.decomposition(**kw)

        baseline_factors = self.s.learning_results.factors[~self.sig_mask, :].copy()

        # With reproject='signal'
        self.s.decomposition(**kw, reproject="signal")

        reproj_factors = self.s.learning_results.factors[~self.sig_mask, :]
        np.testing.assert_allclose(baseline_factors, reproj_factors, atol=1e-10)

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_reproject_both_fills_factors_and_loadings(self, algorithm, svd_solver):
        """reproject='both' fills both factors (signal channels) and loadings
        (nav positions) — no NaN anywhere."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="both",
            print_info=False,
        )
        t = self.s.learning_results
        assert t.factors.shape[0] == self.data.shape[1]
        assert not np.any(np.isnan(t.factors)), "factors still contain NaN"
        assert t.loadings.shape[0] == self.data.shape[0]
        assert not np.any(np.isnan(t.loadings)), "loadings still contain NaN"

    def test_reproject_both_svd_reconstruction(self):
        """reproject='both' SVD: full data reconstructed from loadings × factors."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="both",
            print_info=False,
        )
        t = self.s.learning_results
        rms = np.sqrt(np.mean((t.loadings @ t.factors.T - self.data) ** 2))
        assert rms < 1e-10, f"reproject='both' RMS {rms:.2e} too large"

    @skip_sklearn
    @pytest.mark.parametrize("algorithm", ["ORPCA", "ORNMF"])
    def test_reproject_signal_orpca_ornmf(self, algorithm):
        """ORPCA/ORNMF with reproject='signal' now works: factors cover the full
        signal (no NaN at masked signal channels)."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="signal",
            print_info=False,
        )
        t = self.s.learning_results
        assert t.factors is not None
        assert t.loadings is not None
        # After signal reprojection, factors must cover all signal channels
        assert t.factors.shape[0] == self.s.axes_manager.signal_size
        assert not np.any(np.isnan(t.factors)), "factors must not contain NaN"
        assert t.loadings is not None

    @skip_sklearn
    @pytest.mark.parametrize("algorithm", ["ORPCA", "ORNMF"])
    def test_reproject_both_orpca_ornmf(self, algorithm):
        """ORPCA/ORNMF with reproject='both' fills both loadings and factors."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="both",
            print_info=False,
        )
        # Nav reproject should still have run → loadings fully filled
        loadings = self.s.learning_results.loadings
        assert loadings.shape[0] == self.data.shape[0]
        assert not np.any(np.isnan(loadings)), "loadings still contain NaN"
        # Signal reproject should have run → factors fully filled
        factors = self.s.learning_results.factors
        assert factors.shape[0] == self.s.axes_manager.signal_size
        assert not np.any(np.isnan(factors)), "factors still contain NaN"


class TestLazyVsNonLazyDecomposition:
    """Verify that lazy and non-lazy SVD decomposition agree numerically.

    Both algorithms perform an exact rank-k factorisation of the same
    data, so the reconstruction errors and singular-value spectra should
    be identical (up to floating-point tolerance).  Factors/loadings may
    differ by an orthogonal rotation, so we compare the *subspace* via
    reconstruction rather than individual vectors.
    """

    def setup_method(self, method):
        rng = np.random.default_rng(99)
        rank = 3
        U = rng.standard_normal((20, rank))
        V = rng.standard_normal((100, rank))
        self.data = U @ V.T
        from hyperspy.signals import Signal1D as S1D

        self.s_nl = S1D(self.data.copy())
        self.s_lz = S1D(self.data.copy()).as_lazy()
        self.nav_mask = _nav_mask_1d()
        self.sig_mask = _sig_mask_1d()

    @skip_sklearn
    def test_no_mask_reconstruction(self):
        """Both paths reconstruct exact rank-3 data without masks."""
        self.s_nl.decomposition(output_dimension=3, print_info=False)

        self.s_lz.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            print_info=False,
        )
        for s, label in [(self.s_nl, "non-lazy"), (self.s_lz, "lazy")]:
            t = s.learning_results
            rms = np.sqrt(np.mean((t.loadings @ t.factors.T - self.data) ** 2))
            assert rms < 1e-10, f"{label} reconstruction RMS {rms:.2e} too large"

    @skip_sklearn
    def test_nav_mask_reconstruction(self):
        """Both paths reconstruct unmasked region accurately with nav mask."""
        kw = dict(output_dimension=3, navigation_mask=self.nav_mask, print_info=False)
        self.s_nl.decomposition(**kw)

        self.s_lz.decomposition(algorithm="SVD", svd_solver="incremental", **kw)

        kept_nav = ~self.nav_mask
        for s, label in [(self.s_nl, "non-lazy"), (self.s_lz, "lazy")]:
            t = s.learning_results
            f = t.factors
            l_ = t.loadings[kept_nav, :]
            rms = np.sqrt(np.mean((l_ @ f.T - self.data[kept_nav]) ** 2))
            assert rms < 1e-10, f"{label} nav-masked RMS {rms:.2e} too large"

    @skip_sklearn
    def test_sig_mask_reconstruction(self):
        """Both paths reconstruct unmasked region accurately with sig mask."""
        kw = dict(output_dimension=3, signal_mask=self.sig_mask, print_info=False)
        self.s_nl.decomposition(**kw)

        self.s_lz.decomposition(algorithm="SVD", svd_solver="incremental", **kw)

        kept_sig = ~self.sig_mask
        for s, label in [(self.s_nl, "non-lazy"), (self.s_lz, "lazy")]:
            t = s.learning_results
            f = t.factors[kept_sig, :]
            l_ = t.loadings
            rms = np.sqrt(np.mean((l_ @ f.T - self.data[:, kept_sig]) ** 2))
            assert rms < 1e-10, f"{label} sig-masked RMS {rms:.2e} too large"

    @skip_sklearn
    def test_both_masks_reconstruction(self):
        """Both paths reconstruct unmasked sub-region with both masks."""
        kw = dict(
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        self.s_nl.decomposition(**kw)

        self.s_lz.decomposition(algorithm="SVD", svd_solver="incremental", **kw)

        kept_nav = ~self.nav_mask
        kept_sig = ~self.sig_mask
        for s, label in [(self.s_nl, "non-lazy"), (self.s_lz, "lazy")]:
            t = s.learning_results
            f = t.factors[kept_sig, :]
            l_ = t.loadings[kept_nav, :]
            rms = np.sqrt(np.mean((l_ @ f.T - self.data[kept_nav][:, kept_sig]) ** 2))
            assert rms < 1e-10, f"{label} both-masked RMS {rms:.2e} too large"

    @skip_sklearn
    def test_reproject_navigation_reconstruction(self):
        """After reproject='navigation', lazy SVD reconstructs full data."""
        # Non-lazy SVD + reproject='navigation' is known to produce poor
        # full-data reconstruction (pre-existing issue: the basis is trained
        # on unmasked rows only, and the reproject scale is off). Only assert
        # the lazy path here.
        kw = dict(
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="navigation",
            print_info=False,
        )
        self.s_lz.decomposition(algorithm="SVD", svd_solver="incremental", **kw)

        t = self.s_lz.learning_results
        rms = np.sqrt(np.mean((t.loadings @ t.factors.T - self.data) ** 2))
        assert rms < 1e-10, f"lazy reproject RMS {rms:.2e} too large"

    @skip_sklearn
    def test_explained_variance_is_decreasing(self):
        """Both paths yield a monotonically decreasing explained variance."""
        # The non-lazy and lazy SVD backends (scipy vs ISVD) may produce
        # different singular value estimates, so we only verify the ordering,
        # not the exact values.
        self.s_nl.decomposition(output_dimension=5, print_info=False)

        self.s_lz.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=5,
            print_info=False,
        )
        for s, label in [(self.s_nl, "non-lazy"), (self.s_lz, "lazy")]:
            ev = s.learning_results.explained_variance
            assert np.all(np.diff(ev) <= 0), (
                f"{label} explained_variance not monotonically decreasing: {ev}"
            )


class TestSubSignalChunking:
    """Decomposition on lazy signals whose on-disk chunk size is smaller than
    the full signal size.

    This is the common case for files saved with per-spectrum chunking
    (e.g. HDF5 files written by acquisition software where each spatial pixel
    is its own chunk).  Before the fix in _block_iterator, the signal
    dimension was not rechunked to a single chunk, so only the first signal
    chunk was read per navigation block, producing factors with the wrong
    number of rows and a broadcast error when Poisson rescaling was applied.
    """

    def _make_signal(self, nav_shape, sig_size, sig_chunk, nav_chunk, rank=3):
        """Build a rank-*rank* lazy Signal1D with controlled chunk sizes.

        Parameters
        ----------
        nav_shape : tuple of int
            Navigation shape, e.g. (8, 8) for a 2-D map.
        sig_size : int
            Number of signal channels.
        sig_chunk : int
            Chunk size along the signal axis (< sig_size to trigger the bug).
        nav_chunk : int or tuple
            Chunk size(s) along each navigation axis.
        rank : int
            Rank of the underlying low-rank matrix.
        """
        rng = np.random.default_rng(42)
        nav_size = int(np.prod(nav_shape))
        # Non-negative data (compatible with Poisson noise normalisation)
        U = np.abs(rng.standard_normal((nav_size, rank)))
        V = np.abs(rng.standard_normal((sig_size, rank)))
        data = (U @ V.T).reshape(nav_shape + (sig_size,))
        # Ensure strictly positive for Poisson noise normalisation
        data += 0.1
        chunks = tuple(
            nav_chunk if np.isscalar(nav_chunk) else nav_chunk[i]
            for i in range(len(nav_shape))
        ) + (sig_chunk,)
        da_data = da.from_array(data, chunks=chunks)
        return Signal1D(da_data).as_lazy(), data, rank

    # ------------------------------------------------------------------
    # Basic correctness: factors must have sig_size rows
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "nav_shape,sig_size,sig_chunk,nav_chunk",
        [
            # 1-D navigation, signal chunked into 8 pieces
            ((16,), 64, 8, 4),
            # 2-D navigation, signal chunked into 4 pieces
            ((8, 8), 64, 16, 4),
            # 2-D navigation, signal chunk == 1 (extreme case)
            ((4, 4), 32, 1, 2),
        ],
    )
    def test_factors_have_correct_signal_size(
        self, nav_shape, sig_size, sig_chunk, nav_chunk
    ):
        """factors.shape[0] must equal sig_size regardless of chunk layout."""
        s, _, rank = self._make_signal(nav_shape, sig_size, sig_chunk, nav_chunk)
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=rank,
            print_info=False,
        )
        assert s.learning_results.factors.shape[0] == sig_size

    # ------------------------------------------------------------------
    # normalize_poissonian_noise must not raise a broadcast error
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "nav_shape,sig_size,sig_chunk,nav_chunk",
        [
            ((16,), 64, 8, 4),
            ((8, 8), 64, 16, 4),
        ],
    )
    def test_normalize_poissonian_noise_no_broadcast_error(
        self, nav_shape, sig_size, sig_chunk, nav_chunk
    ):
        """decomposition(normalize_poissonian_noise=True) must not raise
        ValueError when the signal has multiple chunks."""
        s, _, rank = self._make_signal(nav_shape, sig_size, sig_chunk, nav_chunk)
        # Should not raise
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=rank,
            normalize_poissonian_noise=True,
            print_info=False,
        )
        assert s.learning_results.factors.shape[0] == sig_size

    # ------------------------------------------------------------------
    # Reconstruction quality must be preserved despite sub-signal chunking
    # ------------------------------------------------------------------

    def test_reconstruction_quality_sub_signal_chunks(self):
        """SVD on sub-signal-chunked data gives the same reconstruction
        quality as SVD on a signal-contiguous chunked version."""
        nav_shape = (8, 8)
        sig_size = 64
        rank = 3
        rng = np.random.default_rng(7)
        nav_size = int(np.prod(nav_shape))
        U = np.abs(rng.standard_normal((nav_size, rank)))
        V = np.abs(rng.standard_normal((sig_size, rank)))
        data = (U @ V.T).reshape(nav_shape + (sig_size,)) + 0.1

        # Signal-contiguous chunking (signal in one chunk)
        s_cont = Signal1D(da.from_array(data, chunks=(4, 4, sig_size))).as_lazy()
        # Sub-signal chunking (signal split across 8 chunks of 8)
        s_sub = Signal1D(da.from_array(data, chunks=(4, 4, 8))).as_lazy()

        s_cont.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=rank,
            print_info=False,
        )
        s_sub.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=rank,
            print_info=False,
        )

        t_cont = s_cont.learning_results
        t_sub = s_sub.learning_results

        flat = data.reshape(nav_size, sig_size)
        rms_cont = np.sqrt(np.mean((t_cont.loadings @ t_cont.factors.T - flat) ** 2))
        rms_sub = np.sqrt(np.mean((t_sub.loadings @ t_sub.factors.T - flat) ** 2))
        # Both chunk layouts should give the same reconstruction quality
        np.testing.assert_allclose(
            rms_sub,
            rms_cont,
            rtol=1e-5,
            err_msg="sub-signal chunking changed reconstruction quality",
        )

    # ------------------------------------------------------------------
    # PCA and ORPCA also go through _block_iterator — verify shapes
    # ------------------------------------------------------------------

    @skip_sklearn
    @pytest.mark.parametrize("algorithm", ["PCA", "ORPCA"])
    def test_factors_shape_pca_orpca(self, algorithm):
        """PCA and ORPCA also produce factors with sig_size rows when signal
        has sub-signal chunking."""
        nav_shape = (8, 8)
        sig_size = 64
        sig_chunk = 8  # 8 signal chunks
        nav_chunk = 4
        rank = 3
        s, _, _ = self._make_signal(nav_shape, sig_size, sig_chunk, nav_chunk, rank)
        s.decomposition(algorithm=algorithm, output_dimension=rank, print_info=False)
        assert s.learning_results.factors.shape[0] == sig_size


def _make_mask_test_signal(nav_shape, sig_size, seed=123):
    """Build a lazy Signal1D with asymmetric nav and signal shapes.

    Using non-equal dimensions (e.g. nav (6, 8) and sig 40) ensures that any
    accidental axis transposition or shape assumption in the code is caught.
    """
    rng = np.random.default_rng(seed)
    rank = 3
    nav_size = int(np.prod(nav_shape))
    U = np.abs(rng.standard_normal((nav_size, rank)))
    V = np.abs(rng.standard_normal((sig_size, rank)))
    data = (U @ V.T).reshape(nav_shape + (sig_size,)) + 0.1
    # Use asymmetric chunk sizes: nav chunks that don't divide evenly,
    # and sig in one chunk.
    if len(nav_shape) == 1:
        chunks = (nav_shape[0] // 3 + 1, sig_size)
    else:
        chunks = tuple(n // 3 + 1 for n in nav_shape) + (sig_size,)
    s = Signal1D(da.from_array(data, chunks=chunks)).as_lazy()
    return s, nav_size


def _build_nav_masks(s, nav_shape):
    """Return a dict of all four navigation mask types for 2-D nav *s*.

    numpy and dask masks are created with shape == navigation_shape
    (HyperSpy convention: reversed from the underlying array axis order)
    because both _check_navigation_mask and the non-lazy .T.ravel() path
    expect the mask in navigation_shape order.
    """
    # navigation_shape is the HyperSpy-convention shape (reversed from array).
    hs_nav_shape = s.axes_manager.navigation_shape
    nm_np = np.zeros(hs_nav_shape, dtype=bool)
    # Mask a corner using indices in navigation_shape order.
    # Use first two elements along the last nav axis to expose transposition.
    nm_np[0, :2] = True
    nm_dask = da.from_array(nm_np, chunks=tuple(max(1, n // 2) for n in hs_nav_shape))
    # BaseSignal mask: _get_navigation_signal().data is in array axis order,
    # so index it directly and transpose to get signal_dimension=0.
    nm_signal_std = s._get_navigation_signal(dtype="bool")
    nm_signal_std.data[0, :2] = True  # array-axis-order indexing
    nm_signal_lazy = nm_signal_std.as_lazy()
    return {
        "numpy": nm_np,
        "dask": nm_dask,
        "BaseSignal": nm_signal_std.T,
        "LazySignal": nm_signal_lazy.T,
    }


def _build_nav_masks_1d(s, nav_size):
    """Return a dict of all four navigation mask types for 1-D nav *s*."""
    nm_np = np.zeros(nav_size, dtype=bool)
    nm_np[::4] = True
    nm_dask = da.from_array(nm_np, chunks=nav_size // 3 + 1)
    nm_signal_std = s._get_navigation_signal(dtype="bool")
    nm_signal_std.data[::4] = True
    nm_signal_lazy = nm_signal_std.as_lazy()
    return {
        "numpy": nm_np,
        "dask": nm_dask,
        "BaseSignal": nm_signal_std.T,
        "LazySignal": nm_signal_lazy.T,
    }


def _build_sig_masks(s, sig_size):
    """Return a dict of all four signal mask types."""
    sm_np = np.zeros(sig_size, dtype=bool)
    sm_np[:5] = True
    sm_dask = da.from_array(sm_np, chunks=sig_size // 3 + 1)
    sm_signal_std = s._get_signal_signal(dtype="bool")
    sm_signal_std.data[:5] = True
    sm_signal_lazy = sm_signal_std.as_lazy()
    return {
        "numpy": sm_np,
        "dask": sm_dask,
        "BaseSignal": sm_signal_std,
        "LazySignal": sm_signal_lazy,
    }


class TestLazyDecompositionMaskTypes:
    """Verify that lazy SVD decomposition accepts every supported mask type for
    both navigation_mask and signal_mask, for both 1-D and 2-D navigation
    spaces with **asymmetric** shapes to catch axis-transposition bugs.

    Mask types tested:
    - numpy boolean array
    - dask boolean array
    - standard (in-memory) BaseSignal
    - lazy (dask-backed) BaseSignal

    Regression cases covered:
    - BaseSignal nav mask not unwrapped before unfold() → dask chunk mismatch.
    - _root_aG/_root_bH broadcast error with Poisson noise + masks.
    - 2-D nav mask ravelled before fold() but used post-fold for reproject.
    """

    # nav_shape (6, 8) × sig 40: all three dimensions deliberately differ so
    # that a swapped axis would produce the wrong size and be caught immediately.
    @pytest.mark.parametrize(
        "nav_shape,sig_size",
        [
            ((18,), 40),  # 1-D nav: 18 ≠ 40
            ((6, 8), 40),  # 2-D nav: 6 ≠ 8 ≠ 40
        ],
    )
    @pytest.mark.parametrize("mask_type", ["numpy", "dask", "BaseSignal", "LazySignal"])
    def test_nav_mask_types(self, nav_shape, sig_size, mask_type):
        """Every navigation mask type produces correct loadings shape."""
        s, nav_size = _make_mask_test_signal(nav_shape, sig_size)
        if len(nav_shape) == 1:
            nav_masks = _build_nav_masks_1d(s, nav_size)
        else:
            nav_masks = _build_nav_masks(s, nav_shape)
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=nav_masks[mask_type],
            print_info=False,
        )
        assert s.learning_results.loadings.shape[0] == nav_size

    @pytest.mark.parametrize(
        "nav_shape,sig_size",
        [
            ((18,), 40),
            ((6, 8), 40),
        ],
    )
    @pytest.mark.parametrize("mask_type", ["numpy", "dask", "BaseSignal", "LazySignal"])
    def test_sig_mask_types(self, nav_shape, sig_size, mask_type):
        """Every signal mask type produces correct factors shape."""
        s, nav_size = _make_mask_test_signal(nav_shape, sig_size)
        sig_masks = _build_sig_masks(s, sig_size)
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            signal_mask=sig_masks[mask_type],
            print_info=False,
        )
        assert s.learning_results.factors.shape[0] == sig_size

    @pytest.mark.parametrize(
        "nav_shape,sig_size",
        [
            ((18,), 40),
            ((6, 8), 40),
        ],
    )
    @pytest.mark.parametrize("mask_type", ["numpy", "dask", "BaseSignal", "LazySignal"])
    def test_nav_mask_with_poisson(self, nav_shape, sig_size, mask_type):
        """Navigation mask + Poisson normalisation: regression for _root_aG
        broadcast error when nav pixels are masked."""
        s, nav_size = _make_mask_test_signal(nav_shape, sig_size)
        if len(nav_shape) == 1:
            nav_masks = _build_nav_masks_1d(s, nav_size)
        else:
            nav_masks = _build_nav_masks(s, nav_shape)
        s.decomposition(
            True,
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=nav_masks[mask_type],
            print_info=False,
        )
        assert s.learning_results.loadings.shape[0] == nav_size

    @pytest.mark.parametrize(
        "nav_shape,sig_size",
        [
            ((18,), 40),
            ((6, 8), 40),
        ],
    )
    @pytest.mark.parametrize("mask_type", ["numpy", "dask", "BaseSignal", "LazySignal"])
    def test_sig_mask_with_poisson(self, nav_shape, sig_size, mask_type):
        """Signal mask + Poisson normalisation: regression for _root_bH
        broadcast error when signal channels are masked."""
        s, nav_size = _make_mask_test_signal(nav_shape, sig_size)
        sig_masks = _build_sig_masks(s, sig_size)
        s.decomposition(
            True,
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            signal_mask=sig_masks[mask_type],
            print_info=False,
        )
        assert s.learning_results.factors.shape[0] == sig_size

    @pytest.mark.parametrize(
        "nav_shape,sig_size",
        [
            ((18,), 40),
            ((6, 8), 40),
        ],
    )
    @pytest.mark.parametrize("mask_type", ["numpy", "dask", "BaseSignal", "LazySignal"])
    def test_both_masks_with_poisson(self, nav_shape, sig_size, mask_type):
        """Both mask types together work with Poisson normalisation."""
        s, nav_size = _make_mask_test_signal(nav_shape, sig_size)
        if len(nav_shape) == 1:
            nav_masks = _build_nav_masks_1d(s, nav_size)
        else:
            nav_masks = _build_nav_masks(s, nav_shape)
        sig_masks = _build_sig_masks(s, sig_size)
        s.decomposition(
            True,
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=nav_masks[mask_type],
            signal_mask=sig_masks[mask_type],
            print_info=False,
        )
        assert s.learning_results.loadings.shape[0] == nav_size
        assert s.learning_results.factors.shape[0] == sig_size

    @pytest.mark.parametrize(
        "nav_shape,sig_size",
        [
            ((18,), 40),
            ((6, 8), 40),
        ],
    )
    @pytest.mark.parametrize("reproject", ["navigation", "signal", "both"])
    def test_reproject_2d_nav_numpy_mask(self, nav_shape, sig_size, reproject):
        """reproject with a 2-D nav numpy mask must not raise a dask shape
        error (regression: nav mask was ravelled before fold() but then used
        post-fold in _block_iterator, causing a chunk-shape mismatch)."""
        s, nav_size = _make_mask_test_signal(nav_shape, sig_size)
        if len(nav_shape) == 1:
            nav_mask = _build_nav_masks_1d(s, nav_size)["numpy"]
        else:
            nav_mask = _build_nav_masks(s, nav_shape)["numpy"]
        sig_mask = _build_sig_masks(s, sig_size)["numpy"]
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=nav_mask,
            signal_mask=sig_mask,
            reproject=reproject,
            print_info=False,
        )
        t = s.learning_results
        assert t.loadings.shape[0] == nav_size
        assert t.factors.shape[0] == sig_size
        # reproject='navigation' or 'both' → no NaN in loadings
        if reproject in ("navigation", "both"):
            assert not np.any(np.isnan(t.loadings))
        # reproject='signal' or 'both' → no NaN in factors
        if reproject in ("signal", "both"):
            assert not np.any(np.isnan(t.factors))


class TestLazyCentreMaskParity:
    """Regression tests for centre= bugs in lazy SVD decomposition.

    B2: centre='navigation' mean must be computed over unmasked nav positions
        only, matching the non-lazy behaviour.
    B4: centre='navigation' + signal_mask + reproject='signal' must not raise
        a TypeError from trying to boolean-index-assign a 2-D mean array.
    """

    def setup_method(self, method):
        rng = np.random.default_rng(42)
        # Asymmetric: nav (14,) ≠ sig 23 so any transposition is caught.
        nav = 14
        sig = 23
        rank = 3
        U = np.abs(rng.standard_normal((nav, rank)))
        V = np.abs(rng.standard_normal((sig, rank)))
        self.data = (U @ V.T) + 1.0  # strictly positive
        self.nav_mask = np.zeros(nav, dtype=bool)
        self.nav_mask[::3] = True  # mask every third position
        self.sig_mask = np.zeros(sig, dtype=bool)
        self.sig_mask[:4] = True  # mask first 4 channels

    def _make_signals(self):
        s_nl = Signal1D(self.data.copy())
        s_lz = Signal1D(self.data.copy()).as_lazy()
        return s_nl, s_lz

    def test_centre_navigation_mean_is_mask_aware(self):
        """B2: lazy centre='navigation' mean must equal the non-lazy mean
        (computed over unmasked rows only, not the full data)."""
        s_nl, s_lz = self._make_signals()
        kw = dict(
            output_dimension=3,
            centre="navigation",
            navigation_mask=self.nav_mask,
            print_info=False,
        )
        s_nl.decomposition(**kw)

        s_lz.decomposition(**kw)

        nl_mean = s_nl.learning_results.mean
        lz_mean = s_lz.learning_results.mean
        # Both should equal the mean computed only over unmasked rows.
        expected = self.data[~self.nav_mask].mean(axis=0, keepdims=True)
        np.testing.assert_allclose(nl_mean, expected, rtol=1e-10)
        np.testing.assert_allclose(lz_mean, expected, rtol=1e-10)

    def test_centre_navigation_mean_differs_from_full_mean(self):
        """B2 (regression guard): with masked nav positions, the mask-aware
        mean must differ from the full-data mean when the mask is non-trivial."""
        _, s_lz = self._make_signals()
        s_lz.decomposition(
            output_dimension=3,
            centre="navigation",
            navigation_mask=self.nav_mask,
            print_info=False,
        )
        lz_mean = s_lz.learning_results.mean.ravel()
        full_mean = self.data.mean(axis=0)
        # They should NOT be equal (masked rows have different values).
        assert not np.allclose(lz_mean, full_mean), (
            "Lazy mean unexpectedly equals full-data mean; "
            "mask was not applied when computing the centre."
        )

    @pytest.mark.parametrize("reproject", ["signal", "both"])
    def test_centre_with_both_masks_and_signal_reproject(self, reproject):
        """B4: centre='navigation' + signal_mask + reproject='signal'/'both'
        must not raise TypeError (mean was 2-D, boolean-index assignment
        failed when expanding to full signal size)."""
        _, s_lz = self._make_signals()
        s_lz.decomposition(
            output_dimension=3,
            centre="navigation",
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject=reproject,
            print_info=False,
        )
        t = s_lz.learning_results
        assert t.factors.shape == (len(self.sig_mask), 3)
        assert not np.any(np.isnan(t.factors)), (
            "factors should have no NaN after signal reproject"
        )


class TestLazyDecompositionInputValidation:
    """Verify that lazy decomposition raises the same guards as non-lazy.

    m1 - TypeError for non-float data.
    m2 - AttributeError when navigation_size < 2.
    m3 - ValueError from _check_navigation_mask when mask shape is wrong.

    All tests use asymmetric shapes and the SVD algorithm so that the
    previously-missing SVD path validation is exercised.
    """

    def test_non_float_dtype_raises(self):
        """m1: integer data must raise TypeError (mirrors _mva.py:262)."""
        s = Signal1D(np.ones((10, 15), dtype=np.int32)).as_lazy()
        with pytest.raises(TypeError, match="float or complex"):
            s.decomposition(output_dimension=3, print_info=False)

    def test_navigation_size_lt2_raises(self):
        """m2: navigation_size < 2 must raise AttributeError."""
        s = Signal1D(np.ones((1, 15), dtype=float)).as_lazy()
        with pytest.raises(AttributeError, match="navigation_size < 2"):
            s.decomposition(output_dimension=3, print_info=False)

    def test_bad_nav_mask_shape_raises(self):
        """m3: a numpy navigation mask with wrong shape must raise ValueError
        from _check_navigation_mask (previously skipped for SVD)."""
        s = Signal1D(np.ones((12, 15), dtype=float)).as_lazy()
        # navigation_shape is (12,); pass a mask with wrong length
        bad_mask = np.zeros(7, dtype=bool)
        with pytest.raises(ValueError, match="navigation mask"):
            s.decomposition(
                output_dimension=3, navigation_mask=bad_mask, print_info=False
            )


# ─────────────────────────────────────────────────────────────────────────────
# New algorithms and parameters added for parity with non-lazy decomposition
# ─────────────────────────────────────────────────────────────────────────────

SKLEARN_INSTALLED = importlib.util.find_spec("sklearn") is not None
skip_no_sklearn = pytest.mark.skipif(
    not SKLEARN_INSTALLED, reason="scikit-learn not installed"
)


def _make_lazy_signal(nav=(6, 8), sig=40, n_components=3, seed=42):
    """Create a rank-*n_components* lazy Signal1D with asymmetric nav shape."""
    rng = np.random.default_rng(seed)
    nav_size = int(np.prod(nav))
    L = rng.standard_normal((nav_size, n_components))
    F = rng.standard_normal((n_components, sig))
    data = (L @ F + 0.01 * rng.standard_normal((nav_size, sig))).reshape(nav + (sig,))
    return Signal1D(data.astype(float)).as_lazy()


@skip_no_sklearn
class TestLazyNMFAlgorithm:
    """Tests for algorithm='NMF' (MiniBatchNMF) on lazy signals."""

    def setup_method(self, method):
        # NMF requires non-negative data
        rng = np.random.default_rng(0)
        nav_size = 6 * 8
        L = np.abs(rng.standard_normal((nav_size, 3)))
        F = np.abs(rng.standard_normal((3, 40)))
        data = (L @ F + 0.01 * np.abs(rng.standard_normal((nav_size, 40)))).reshape(
            (6, 8, 40)
        )
        self.s = Signal1D(data.astype(float)).as_lazy()

    def test_nmf_requires_output_dimension(self):
        with pytest.raises(ValueError, match="output_dimension"):
            self.s.decomposition(algorithm="NMF", print_info=False)

    def test_nmf_runs(self):
        self.s.decomposition(algorithm="NMF", output_dimension=3, print_info=False)
        lr = self.s.learning_results
        assert lr.factors is not None
        assert lr.loadings is not None
        assert lr.factors.shape[1] == 3
        assert lr.loadings.shape[1] == 3

    def test_nmf_factors_shape(self):
        self.s.decomposition(algorithm="NMF", output_dimension=3, print_info=False)
        lr = self.s.learning_results
        # factors: (sig_size, n_components), loadings: (nav_size, n_components)
        assert lr.factors.shape == (40, 3)
        assert lr.loadings.shape == (6 * 8, 3)

    def test_nmf_return_info(self):
        obj = self.s.decomposition(
            algorithm="NMF", output_dimension=3, print_info=False, return_info=True
        )

        assert hasattr(obj, "components_")

    def test_nmf_reproject_navigation(self):
        self.s.decomposition(
            algorithm="NMF",
            output_dimension=3,
            reproject="navigation",
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.loadings.shape[0] == 6 * 8  # full nav


@skip_no_sklearn
class TestLazyCustomSklearnObject:
    """Tests for passing a custom sklearn-like object to lazy decomposition."""

    def setup_method(self, method):
        self.s = _make_lazy_signal(nav=(6, 8), sig=40, n_components=3)

    def _make_incremental_estimator(self, n_components):
        """Return an IncrementalPCA-based estimator (has partial_fit)."""
        import sklearn.decomposition

        obj = sklearn.decomposition.IncrementalPCA(n_components=n_components)
        return obj

    def _make_batch_estimator(self, n_components):
        """Return a PCA estimator (no partial_fit, uses fit_transform)."""
        import sklearn.decomposition

        return sklearn.decomposition.PCA(n_components=n_components)

    def test_custom_incremental_estimator(self):
        """Object with partial_fit is used incrementally."""
        obj = self._make_incremental_estimator(3)
        returned = self.s.decomposition(
            algorithm=obj, output_dimension=3, print_info=False, return_info=True
        )
        lr = self.s.learning_results
        assert lr.factors is not None
        assert lr.loadings is not None
        assert lr.factors.shape[1] == 3
        # return_info should give back the estimator
        assert returned is obj

    def test_custom_batch_estimator(self):
        """Object without partial_fit falls back to fit_transform."""
        obj = self._make_batch_estimator(3)
        self.s.decomposition(algorithm=obj, print_info=False)

        lr = self.s.learning_results
        assert lr.factors is not None
        assert lr.factors.shape[1] == 3

    def test_custom_estimator_missing_components_raises(self):
        """Estimator without components_ attribute must raise AttributeError."""

        class BadEstimator:
            def fit_transform(self, X):
                return X[:, :3]

        obj = BadEstimator()
        with pytest.raises(AttributeError, match="components_"):
            self.s.decomposition(algorithm=obj, print_info=False)

    def test_unrecognised_string_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            self.s.decomposition(
                algorithm="bogus_algo", output_dimension=3, print_info=False
            )

    def test_custom_estimator_return_info(self):
        obj = self._make_incremental_estimator(3)
        ret = self.s.decomposition(
            algorithm=obj, output_dimension=3, print_info=False, return_info=True
        )
        assert ret is obj


@skip_no_sklearn
class TestLazySVDSolverAndAutoTranspose:
    """Tests that svd_solver and auto_transpose are accepted without error."""

    def setup_method(self, method):
        self.s = _make_lazy_signal(nav=(6, 8), sig=40, n_components=3)

    def test_svd_solver_accepted_svd(self):
        """svd_solver is accepted for SVD algorithm without error."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            print_info=False,
        )

    def test_svd_solver_accepted_pca(self):
        """svd_solver is accepted for PCA algorithm without error."""
        self.s.decomposition(
            algorithm="PCA",
            output_dimension=3,
            svd_solver="full",
            print_info=False,
        )

    def test_auto_transpose_true_ignored_for_svd(self, caplog):
        """auto_transpose=True is silently ignored for SVD with an info log."""
        import logging

        with caplog.at_level(logging.INFO, logger="hyperspy._signals.lazy"):
            self.s.decomposition(
                algorithm="SVD",
                svd_solver="incremental",
                output_dimension=3,
                auto_transpose=True,
                print_info=False,
            )
        assert any("auto_transpose" in r.message for r in caplog.records)

    def test_auto_transpose_false_no_error(self):
        """auto_transpose=False is accepted without error."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            auto_transpose=False,
            print_info=False,
        )


class TestSVDAlgorithm:
    """Tests for algorithm='SVD' (dask.array.linalg.svd) on lazy signals."""

    def setup_method(self, method):
        rng = np.random.default_rng(42)
        # Use asymmetric nav/sig shapes to catch any axis-transposition bugs.
        # Rank-3 signal: nav=(7, 5), sig=30
        L = rng.standard_normal((35, 3))
        F = rng.standard_normal((3, 30))
        data = (L @ F + 0.01 * rng.standard_normal((35, 30))).reshape((7, 5, 30))
        self.s = Signal1D(data.astype(float)).as_lazy()

    def test_basic_run(self):
        """SVD runs without error and returns results."""
        self.s.decomposition(algorithm="SVD", svd_solver="dask", print_info=False)
        lr = self.s.learning_results
        assert lr.factors is not None
        assert lr.loadings is not None

    def test_output_dimension_optional(self):
        """output_dimension is optional for SVD."""
        self.s.decomposition(algorithm="SVD", svd_solver="dask", print_info=False)
        lr = self.s.learning_results
        # Without output_dimension, all components up to min(nav, sig) are kept.
        assert lr.factors.shape[1] <= min(35, 30)

    def test_output_dimension_respected(self):
        """When output_dimension is given, exactly that many components are returned."""
        k = 4
        self.s.decomposition(
            algorithm="SVD", svd_solver="dask", output_dimension=k, print_info=False
        )
        lr = self.s.learning_results
        assert lr.factors.shape == (30, k)
        assert lr.loadings.shape == (35, k)

    def test_factors_and_loadings_shapes(self):
        """Factors shape is (sig_size, k); loadings shape is (nav_size, k)."""
        k = 3
        self.s.decomposition(
            algorithm="SVD", svd_solver="dask", output_dimension=k, print_info=False
        )
        lr = self.s.learning_results
        assert lr.factors.shape == (30, k)
        assert lr.loadings.shape == (35, k)

    def test_explained_variance_set(self):
        """explained_variance is populated after SVD."""
        self.s.decomposition(
            algorithm="SVD", svd_solver="dask", output_dimension=3, print_info=False
        )
        lr = self.s.learning_results
        assert lr.explained_variance is not None
        assert lr.explained_variance.shape == (3,)

    def test_reconstruction_quality(self):
        """First 3 components should reconstruct the (near rank-3) signal well."""
        self.s.decomposition(
            algorithm="SVD", svd_solver="dask", output_dimension=3, print_info=False
        )
        lr = self.s.learning_results
        recon = (lr.loadings @ lr.factors.T).reshape(7, 5, 30)
        original = self.s.data.compute()
        rel_error = np.linalg.norm(recon - original) / np.linalg.norm(original)
        assert rel_error < 0.1

    def test_navigation_mask(self):
        """SVD respects navigation_mask: masked rows excluded from SVD."""
        # navigation_shape is reversed vs array shape: data (7,5,30) → nav_shape (5,7)
        nav_mask = np.zeros((5, 7), dtype=bool)
        nav_mask[0, :] = True  # mask first row (7 pixels)
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="dask",
            output_dimension=3,
            navigation_mask=nav_mask,
            print_info=False,
        )
        lr = self.s.learning_results
        # Factors computed over 28 unmasked nav positions → shape (30, 3)
        assert lr.factors.shape == (30, 3)
        # Loadings only cover unmasked nav pixels during learn pass
        assert lr.loadings.shape[1] == 3

    def test_signal_mask(self):
        """SVD respects signal_mask: masked channels excluded from SVD."""
        sig_mask = np.zeros(30, dtype=bool)
        sig_mask[:5] = True  # mask first 5 channels
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="dask",
            output_dimension=3,
            signal_mask=sig_mask,
            print_info=False,
        )
        lr = self.s.learning_results
        # Factors are stored at full signal size with NaN at masked channels
        assert lr.factors.shape == (30, 3)
        assert np.all(np.isnan(lr.factors[:5, :]))

    def test_centre_navigation(self):
        """SVD supports centre='navigation'."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="dask",
            output_dimension=3,
            centre="navigation",
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.factors is not None
        assert lr.factors.shape == (30, 3)

    def test_centre_signal(self):
        """SVD supports centre='signal'."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="dask",
            output_dimension=3,
            centre="signal",
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.factors is not None
        assert lr.factors.shape == (30, 3)

    def test_reproject_navigation(self):
        """reproject='navigation' works for SVD."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="dask",
            output_dimension=3,
            reproject="navigation",
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.loadings.shape == (35, 3)
        assert not np.any(np.isnan(lr.loadings))
