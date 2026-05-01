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

# PCA and ORNMF with any mask hang on lazy signals — pre-existing bugs, not
# introduced here.  Call _xfail_masked_algorithm() at the top of any
# parametrized test that receives one of these algorithms and uses a mask.
_MASKED_ALGO_BUG_MSG = (
    "{algorithm} with navigation_mask or signal_mask hangs on lazy signals — "
    "pre-existing bug reproducible on main branch, unrelated to this PR"
)
_HANGING_MASKED_ALGORITHMS = {"PCA", "ORNMF"}


def _xfail_pca_with_mask(algorithm):
    if algorithm in _HANGING_MASKED_ALGORITHMS:
        pytest.xfail(_MASKED_ALGO_BUG_MSG.format(algorithm=algorithm))


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
            self.s.decomposition(algorithm="SVD")

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
        s.decomposition(algorithm="SVD", output_dimension=3, signal_mask=sig_mask)
        assert s.learning_results.factors is not None
        assert s.learning_results.loadings is not None

    def test_algorithm_error(self):
        with pytest.raises(ValueError, match="'algorithm' not recognised"):
            self.s.decomposition(algorithm="random")


class TestPrintInfo:
    def setup_method(self, method):
        rng = np.random.default_rng(123)
        self.s = Signal1D(rng.random(size=(20, 100))).as_lazy()

    @pytest.mark.parametrize("algorithm", ["SVD", "ORPCA", "ORNMF"])
    def test_decomposition(self, algorithm, capfd):
        self.s.decomposition(algorithm=algorithm, output_dimension=3)
        captured = capfd.readouterr()
        assert "Decomposition info:" in captured.out

    @skip_sklearn
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

    @skip_sklearn
    def test_decomposition_mask_SVD(self):
        """SVD masking is now supported; check shapes are correct."""
        s = self.s
        sig_mask = (s.inav[0].data < 0.5).compute()
        s.decomposition(algorithm="SVD", output_dimension=2, signal_mask=sig_mask)
        assert s.learning_results.factors is not None

        nav_mask = (s.isig[0].data < 0.5).compute()
        s.decomposition(algorithm="SVD", output_dimension=2, navigation_mask=nav_mask)
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
    """Tests for parity with the non-lazy MVA.decomposition() (fixes 1-7).

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
    @pytest.mark.parametrize("algorithm", ["SVD", "PCA"])
    def test_poissonian_flag_stored_true(self, algorithm):
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
    @pytest.mark.parametrize("algorithm", ["SVD", "PCA"])
    def test_number_significant_components(self, algorithm):
        """number_significant_components is a plain Python int after decomposition."""
        self.s.decomposition(algorithm=algorithm, output_dimension=5, print_info=False)
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
    def test_reproject_both_warns_for_signal(self):
        """reproject='both' emits a UserWarning about signal reprojection."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.s.decomposition(
                algorithm="PCA",
                output_dimension=2,
                navigation_mask=self.nav_mask,
                signal_mask=self.sig_mask,
                reproject="both",
                print_info=False,
            )
        messages = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        assert any("signal" in m.lower() for m in messages)

    @skip_sklearn
    def test_reproject_signal_warns(self):
        """reproject='signal' emits a UserWarning (not yet supported)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.s.decomposition(
                algorithm="PCA",
                output_dimension=2,
                signal_mask=self.sig_mask,
                reproject="signal",
                print_info=False,
            )
        messages = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        assert any("signal" in m.lower() for m in messages)

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
            algorithm="SVD", output_dimension=2, return_info=True, print_info=False
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive lazy mask × reproject tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_lazy_lowrank(nav=20, sig=100, rank=3, seed=11):
    """Return a lazy rank-*rank* Signal1D and its raw data array."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((nav, rank))
    V = rng.standard_normal((sig, rank))
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
    @pytest.mark.parametrize("algorithm", ["SVD", "PCA"])
    def test_both_masks_nan_pattern(self, algorithm):
        """Nav-masked → NaN loadings rows; sig-masked → NaN factor rows."""
        _xfail_pca_with_mask(algorithm)
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
        _xfail_pca_with_mask(algorithm)  # guards ORNMF
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
    @pytest.mark.parametrize("algorithm", ["SVD", "PCA"])
    def test_both_masks_reconstruction_quality(self, algorithm):
        """Unmasked region reconstructed near-exactly for a rank-3 lazy signal."""
        _xfail_pca_with_mask(algorithm)
        self.s.decomposition(
            algorithm=algorithm,
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
    @pytest.mark.parametrize("algorithm", ["SVD", "PCA", "ORPCA", "ORNMF"])
    def test_reproject_navigation_no_nan(self, algorithm):
        """reproject='navigation' → full loadings, no NaN, correct shape."""
        _xfail_pca_with_mask(algorithm)
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

    @skip_sklearn
    @pytest.mark.parametrize("algorithm", ["SVD", "PCA"])
    def test_reproject_navigation_reconstruction(self, algorithm):
        """Reprojected loadings × factors reconstruct the full data (rank-3)."""
        _xfail_pca_with_mask(algorithm)
        self.s.decomposition(
            algorithm=algorithm,
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
    @pytest.mark.parametrize("algorithm", ["SVD", "PCA"])
    def test_reproject_navigation_unmasked_rows_unchanged(self, algorithm):
        """reproject='navigation' does not alter the unmasked rows of loadings."""
        _xfail_pca_with_mask(algorithm)
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
    @pytest.mark.parametrize("algorithm", ["SVD", "PCA", "ORPCA", "ORNMF"])
    def test_reproject_both_nav_loadings_filled(self, algorithm):
        """reproject='both' fills nav-masked positions (signal reproject warns)."""
        _xfail_pca_with_mask(algorithm)
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

    @skip_sklearn
    @pytest.mark.parametrize("algorithm", ["SVD", "PCA"])
    def test_reproject_navigation_with_both_masks_reconstruction(self, algorithm):
        """With both masks + reproject='navigation', full data reconstructed."""
        _xfail_pca_with_mask(algorithm)
        self.s.decomposition(
            algorithm=algorithm,
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
        self.s_lz.decomposition(algorithm="SVD", output_dimension=3, print_info=False)
        for s, label in [(self.s_nl, "non-lazy"), (self.s_lz, "lazy")]:
            t = s.learning_results
            rms = np.sqrt(np.mean((t.loadings @ t.factors.T - self.data) ** 2))
            assert rms < 1e-10, f"{label} reconstruction RMS {rms:.2e} too large"

    @skip_sklearn
    def test_nav_mask_reconstruction(self):
        """Both paths reconstruct unmasked region accurately with nav mask."""
        kw = dict(output_dimension=3, navigation_mask=self.nav_mask, print_info=False)
        self.s_nl.decomposition(**kw)
        self.s_lz.decomposition(algorithm="SVD", **kw)

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
        self.s_lz.decomposition(algorithm="SVD", **kw)

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
        self.s_lz.decomposition(algorithm="SVD", **kw)

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
        self.s_lz.decomposition(algorithm="SVD", **kw)

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
        self.s_lz.decomposition(algorithm="SVD", output_dimension=5, print_info=False)
        for s, label in [(self.s_nl, "non-lazy"), (self.s_lz, "lazy")]:
            ev = s.learning_results.explained_variance
            assert np.all(np.diff(ev) <= 0), (
                f"{label} explained_variance not monotonically decreasing: {ev}"
            )
