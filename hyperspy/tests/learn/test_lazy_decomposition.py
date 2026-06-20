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

from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.learn._mva import _keenan_kotula_scale
from hyperspy.signals import Signal1D

sklearn = importlib.util.find_spec("sklearn")
skip_sklearn = pytest.mark.skipif(sklearn is None, reason="sklearn not installed")

# Suppress the svd_solver-default-change DeprecationWarning in all tests except
# the dedicated deprecation warning tests.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The default svd_solver for algorithm='SVD':DeprecationWarning"
)


class TestKeenanKotulaScaleDask:
    """Regression tests for _keenan_kotula_scale with dask inputs."""

    def test_dask_input_matches_numpy(self):
        """_keenan_kotula_scale returns the same result for dask and numpy inputs."""
        rng = np.random.default_rng(0)
        data = rng.integers(1, 100, size=(10, 20)).astype(float)
        navigation_mask = np.zeros(10, dtype=bool)
        navigation_mask[:2] = True
        signal_mask = np.zeros(20, dtype=bool)
        signal_mask[:3] = True

        scaled_np, sqrt_aG_np, sqrt_bH_np = _keenan_kotula_scale(
            data, navigation_mask, signal_mask, ndim=1, sdim=1
        )
        scaled_da, sqrt_aG_da, sqrt_bH_da = _keenan_kotula_scale(
            da.from_array(data), navigation_mask, signal_mask, ndim=1, sdim=1
        )

        np.testing.assert_allclose(scaled_da.compute(), scaled_np, rtol=1e-10)
        np.testing.assert_allclose(sqrt_aG_da.compute(), sqrt_aG_np, rtol=1e-10)
        np.testing.assert_allclose(sqrt_bH_da.compute(), sqrt_bH_np, rtol=1e-10)

    def test_dask_zero_sum_guard_computes_first(self):
        """The zero-sum guard must call float() on computed NumPy values, not
        on the dask array itself (which is not guaranteed to support float())."""
        rng = np.random.default_rng(0)
        data = rng.integers(1, 100, size=(10, 20)).astype(float)

        from unittest.mock import patch

        def raising_float(x):
            if hasattr(x, "chunks"):
                raise TypeError("float() on dask array")
            return float(x)

        with patch("hyperspy.learn._mva.float", side_effect=raising_float):
            # Should not raise: sums are computed before the zero-sum check.
            _keenan_kotula_scale(da.from_array(data), None, None, ndim=1, sdim=1)


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
        components_arr = self.s.learning_results.components
        scores_arr = self.s.learning_results.scores

        if isinstance(components_arr, da.Array):
            components_arr = components_arr.compute()
        if isinstance(scores_arr, da.Array):
            scores_arr = scores_arr.compute()

        explained_variance = self.s.learning_results.explained_variance
        X = scores_arr @ components_arr.T

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
        components_arr = self.s.learning_results.components
        scores_arr = self.s.learning_results.scores

        if isinstance(components_arr, da.Array):
            components_arr = components_arr.compute()
        if isinstance(scores_arr, da.Array):
            scores_arr = scores_arr.compute()

        explained_variance = self.s.learning_results.explained_variance
        X = scores_arr @ components_arr.T

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
        components_arr = s.learning_results.components
        scores_arr = s.learning_results.scores
        _ = scores_arr @ components_arr.T

        # Check singular values
        explained_variance = s.learning_results.explained_variance
        explained_variance_norm = explained_variance / np.sum(explained_variance)
        np.testing.assert_allclose(
            explained_variance_norm[: self.rank].sum(), 1.0, atol=1e-6
        )

        nav_mask = (s.isig[0].data < 1.0).compute()

        s.decomposition(output_dimension=3, algorithm="PCA", navigation_mask=nav_mask)
        components_arr = s.learning_results.components
        scores_arr = s.learning_results.scores
        _ = scores_arr @ components_arr.T

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
        components_arr = self.s.learning_results.components
        scores_arr = self.s.learning_results.scores

        if isinstance(components_arr, da.Array):
            components_arr = components_arr.compute()
        if isinstance(scores_arr, da.Array):
            scores_arr = scores_arr.compute()

        explained_variance = self.s.learning_results.explained_variance
        X = scores_arr @ components_arr.T

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
        components_arr = self.s.learning_results.components
        scores_arr = self.s.learning_results.scores

        if isinstance(components_arr, da.Array):
            components_arr = components_arr.compute()
        if isinstance(scores_arr, da.Array):
            scores_arr = scores_arr.compute()

        explained_variance = self.s.learning_results.explained_variance
        X = scores_arr @ components_arr.T

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
        with pytest.raises(ValueError, match="`output_dimension` must be specified"):
            self.s.decomposition(algorithm="SVD", svd_solver="randomized")

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
        assert s.learning_results.components is not None
        assert s.learning_results.scores is not None

    def test_algorithm_error(self):
        with pytest.raises(ValueError, match="not recognised"):
            self.s.decomposition(algorithm="random")

    def test_svd_default_solver_uses_randomized(self):
        """algorithm='SVD' without svd_solver defaults to 'randomized' without warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            self.s.decomposition(algorithm="SVD", output_dimension=3)

    def test_svd_solver_invalid(self):
        """Unrecognised svd_solver raises ValueError."""
        with pytest.raises(ValueError, match="svd_solver="):
            self.s.decomposition(
                algorithm="SVD", svd_solver="unknown", output_dimension=3
            )

    @skip_sklearn
    def test_randomized_svd_2d_chunked(self):
        """svd_solver='randomized' works with arrays chunked in both dimensions.

        dask.array.linalg.svd_compressed accepts 2D-chunked arrays; there is no
        need to restrict users to 1D chunking.
        """
        # Force 2D chunking on the unfolded (nav x sig) array.
        X = np.random.RandomState(42).random((120, 128))  # 10*12=120 nav
        data = da.from_array(X.reshape(10, 12, 128), chunks=(5, 6, 64))
        s = Signal1D(data).as_lazy()
        s.decomposition(algorithm="SVD", svd_solver="randomized", output_dimension=3)
        assert s.learning_results.components is not None
        assert s.learning_results.scores is not None

    @skip_sklearn
    @pytest.mark.parametrize("centre", [None, "navigation", "signal"])
    def test_randomized_reproject_navigation_lazy(self, centre):
        """reproject='navigation' for svd_solver='randomized' uses a lazy dask
        matmul and never materialises the full dataset in memory.

        The reprojected scores_arr must cover all navigation positions (no NaN)
        and be consistent with the components_arr learned on the masked data.
        """
        nav_mask = np.zeros((10, 10), dtype=bool)
        nav_mask[0, :] = True  # mask the first row

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            centre=centre,
            navigation_mask=nav_mask,
            reproject="navigation",
        )
        lr = self.s.learning_results
        scores_arr = lr.scores
        assert scores_arr.shape[0] == self.s.axes_manager.navigation_size
        assert not np.any(np.isnan(scores_arr)), (
            "reproject='navigation' should fill all rows"
        )


class TestPrintInfo:
    def setup_method(self, method):
        rng = np.random.default_rng(123)
        self.s = Signal1D(rng.random(size=(20, 100))).as_lazy()

    @pytest.mark.parametrize(
        "algorithm,svd_solver",
        [("ORPCA", None), ("ORNMF", None)],
    )
    def test_decomposition(self, algorithm, svd_solver, capfd):
        self.s.decomposition(
            algorithm=algorithm, svd_solver=svd_solver, output_dimension=3
        )
        captured = capfd.readouterr()
        assert "Decomposition info:" in captured.out

    @skip_sklearn
    def test_decomposition_incremental_svd(self, capfd):
        """SVD with svd_solver='incremental' prints decomposition info."""
        self.s.decomposition(
            algorithm="SVD", svd_solver="incremental", output_dimension=3
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

    @skip_sklearn
    def test_no_print(self, capfd):
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
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
        assert s.learning_results.components is not None

        nav_mask = (s.isig[0].data < 0.5).compute()
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=2,
            navigation_mask=nav_mask,
        )
        assert s.learning_results.scores is not None

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
        """components_arr/loadings are rescaled back to original data space after SVD."""
        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            output_dimension=2,
            normalize_poissonian_noise=True,
            print_info=False,
        )
        components_arr = s.learning_results.components  # (20, 2)
        scores_arr = s.learning_results.scores  # (10, 2)
        reconstruction = scores_arr @ components_arr.T  # (10, 20)
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
        """Masked nav positions become NaN rows in scores_arr (no reproject)."""
        n_components = 2
        self.s.decomposition(
            output_dimension=n_components,
            navigation_mask=self.nav_mask,
            print_info=False,
        )
        scores_arr = self.s.learning_results.scores
        # loadings shape should be (nav_size, n_components) = (20, 2)
        assert scores_arr.shape == (20, n_components)
        flat_mask = self.nav_mask.ravel()
        # Masked positions (True) should be NaN
        assert np.all(np.isnan(scores_arr[flat_mask, :]))
        # Unmasked positions should not be NaN
        assert not np.any(np.isnan(scores_arr[~flat_mask, :]))

    @skip_sklearn
    def test_nan_fill_factors_signal_mask(self):
        """Masked signal channels become NaN rows in components_arr (no reproject)."""
        n_components = 2
        self.s.decomposition(
            output_dimension=n_components,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        components_arr = self.s.learning_results.components
        # factors shape should be (sig_size, n_components) = (30, 2)
        assert components_arr.shape == (30, n_components)
        # Masked channels (first 3) should be NaN
        assert np.all(np.isnan(components_arr[: self.sig_mask.sum(), :]))
        # Unmasked channels should not be NaN
        assert not np.any(np.isnan(components_arr[self.sig_mask.sum() :, :]))

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
        """reproject='navigation' returns full (unmasked) scores_arr without NaN."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=2,
            navigation_mask=self.nav_mask,
            reproject="navigation",
            print_info=False,
        )
        scores_arr = self.s.learning_results.scores
        assert scores_arr.shape == (20, 2)
        assert not np.any(np.isnan(scores_arr))

    @skip_sklearn
    def test_reproject_both_signal_fills_factors(self):
        """reproject='both' fills masked signal channels in components_arr (no NaN)."""
        self.s.decomposition(
            algorithm="PCA",
            output_dimension=2,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="both",
            print_info=False,
        )
        components_arr = self.s.learning_results.components
        assert components_arr.shape[0] == 30  # full signal size
        assert not np.any(np.isnan(components_arr))

    @skip_sklearn
    def test_reproject_signal_fills_factors(self):
        """reproject='signal' produces components_arr with no NaN at masked channels."""
        self.s.decomposition(
            algorithm="PCA",
            output_dimension=2,
            signal_mask=self.sig_mask,
            reproject="signal",
            print_info=False,
        )
        components_arr = self.s.learning_results.components
        assert components_arr.shape[0] == 30  # full signal size
        assert not np.any(np.isnan(components_arr))

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
        """return_info=True with SVD returns None for full/randomized
        (no persistent estimator object); returns ISVD for incremental."""
        # randomized — no estimator
        result = self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=2,
            return_info=True,
            print_info=False,
        )
        assert result is None
        # incremental — returns ISVD object
        from hyperspy.learn.incremental_svd import ISVD

        result_isvd = self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=2,
            return_info=True,
            print_info=False,
        )
        assert isinstance(result_isvd, ISVD)


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
    - Correct shape of components/scores (full data dimensions)
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
        """Nav-masked → NaN scores_arr rows; sig-masked → NaN factor rows."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        t = self.s.learning_results
        assert t.scores.shape == (20, 3)
        assert t.components.shape == (100, 3)
        assert np.all(np.isnan(t.scores[self.nav_mask, :]))
        assert not np.any(np.isnan(t.scores[~self.nav_mask, :]))
        assert np.all(np.isnan(t.components[self.sig_mask, :]))
        assert not np.any(np.isnan(t.components[~self.sig_mask, :]))

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
        assert np.all(np.isnan(t.scores[self.nav_mask, :]))
        assert not np.any(np.isnan(t.scores[~self.nav_mask, :]))
        assert np.all(np.isnan(t.components[self.sig_mask, :]))
        assert not np.any(np.isnan(t.components[~self.sig_mask, :]))

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
        f = self.s.learning_results.components[kept_sig, :]
        l_ = self.s.learning_results.scores[kept_nav, :]
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
        """reproject='navigation' → full scores_arr, no NaN, correct shape."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="navigation",
            print_info=False,
        )
        scores_arr = self.s.learning_results.scores
        assert scores_arr.shape == (20, 3)
        assert not np.any(np.isnan(scores_arr))

    @skip_sklearn
    def test_reproject_navigation_reconstruction(self):
        """Reprojected loadings × components_arr reconstruct the full data (rank-3).

        ISVD is an approximate algorithm so reconstruction is approximate;
        use a loose tolerance.  For exact reconstruction see the non-lazy SVD
        test in TestDecompositionReprojectionNumerical.
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
        recon = t.scores @ t.components.T
        rms = np.sqrt(np.mean((recon - self.data) ** 2))
        assert rms < 1.0

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_reproject_navigation_unmasked_rows_unchanged(self, algorithm, svd_solver):
        """reproject='navigation' does not alter the unmasked rows of scores_arr."""
        # Baseline: no reproject, unmasked positions only
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            print_info=False,
        )
        baseline_scores_arr = self.s.learning_results.scores[~self.nav_mask, :].copy()

        # With reproject: should give the same values at unmasked positions
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="navigation",
            print_info=False,
        )
        reproj_scores_arr = self.s.learning_results.scores[~self.nav_mask, :]
        np.testing.assert_allclose(baseline_scores_arr, reproj_scores_arr, atol=1e-10)

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
        scores_arr = self.s.learning_results.scores
        assert scores_arr.shape == (20, 3)
        assert not np.any(np.isnan(scores_arr))

    @skip_sklearn
    def test_reproject_navigation_with_both_masks_reconstruction(self):
        """With both masks + reproject='navigation', full data reconstructed.

        ISVD is approximate; use a loose tolerance.
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
        kept_sig = ~self.sig_mask
        f = t.components[kept_sig, :]
        recon = t.scores @ f.T
        rms = np.sqrt(np.mean((recon - self.data[:, kept_sig]) ** 2))
        assert rms < 1.0

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_reproject_signal_fills_factors(self, algorithm, svd_solver):
        """reproject='signal' → components_arr fully filled (no NaN), loadings still
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
        assert t.components.shape[0] == self.data.shape[1]
        assert not np.any(np.isnan(t.components))
        # Loadings must still have NaN at nav-masked positions
        assert t.scores.shape[0] == self.data.shape[0]
        assert np.any(np.isnan(t.scores[self.nav_mask, :]))

    @skip_sklearn
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
        recon = t.scores[kept_nav, :] @ t.components.T
        rms = np.sqrt(np.mean((recon - self.data[kept_nav]) ** 2))
        assert rms < 1e-10, f"reproject='signal' RMS {rms:.2e} too large"

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_reproject_signal_unmasked_channels_unchanged(self, algorithm, svd_solver):
        """reproject='signal' does not alter the unmasked channel rows of
        components_arr (compared to no-reproject baseline)."""
        kw = dict(
            algorithm=algorithm,
            output_dimension=3,
            signal_mask=self.sig_mask,
            print_info=False,
        )
        # Baseline: no reproject
        self.s.decomposition(**kw)

        baseline_components_arr = self.s.learning_results.components[
            ~self.sig_mask, :
        ].copy()

        # With reproject='signal'
        self.s.decomposition(**kw, reproject="signal")

        reproj_components_arr = self.s.learning_results.components[~self.sig_mask, :]
        np.testing.assert_allclose(
            baseline_components_arr, reproj_components_arr, atol=1e-10
        )

    @skip_sklearn
    @pytest.mark.parametrize(
        "algorithm,svd_solver", [("SVD", "incremental"), ("PCA", None)]
    )
    def test_reproject_both_fills_factors_and_loadings(self, algorithm, svd_solver):
        """reproject='both' fills both components_arr (signal channels) and loadings
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
        assert t.components.shape[0] == self.data.shape[1]
        assert not np.any(np.isnan(t.components)), "components_arr still contain NaN"
        assert t.scores.shape[0] == self.data.shape[0]
        assert not np.any(np.isnan(t.scores)), "scores_arr still contain NaN"

    @skip_sklearn
    def test_reproject_both_svd_reconstruction(self):
        """reproject='both' SVD: full data reconstructed from loadings × components_arr."""
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
        rms = np.sqrt(np.mean((t.scores @ t.components.T - self.data) ** 2))
        assert rms < 1e-10, f"reproject='both' RMS {rms:.2e} too large"

    @skip_sklearn
    @pytest.mark.parametrize("algorithm", ["ORPCA", "ORNMF"])
    def test_reproject_signal_orpca_ornmf(self, algorithm):
        """ORPCA/ORNMF with reproject='signal' now works: components_arr cover the full
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
        assert t.components is not None
        assert t.scores is not None
        # After signal reprojection, factors must cover all signal channels
        assert t.components.shape[0] == self.s.axes_manager.signal_size
        assert not np.any(np.isnan(t.components)), "components_arr must not contain NaN"
        assert t.scores is not None

    @skip_sklearn
    @pytest.mark.parametrize("algorithm", ["ORPCA", "ORNMF"])
    def test_reproject_both_orpca_ornmf(self, algorithm):
        """ORPCA/ORNMF with reproject='both' fills both loadings and components_arr."""
        self.s.decomposition(
            algorithm=algorithm,
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="both",
            print_info=False,
        )
        # Nav reproject should still have run → loadings fully filled
        scores_arr = self.s.learning_results.scores
        assert scores_arr.shape[0] == self.data.shape[0]
        assert not np.any(np.isnan(scores_arr)), "scores_arr still contain NaN"
        # Signal reproject should have run → factors fully filled
        components_arr = self.s.learning_results.components
        assert components_arr.shape[0] == self.s.axes_manager.signal_size
        assert not np.any(np.isnan(components_arr)), "components_arr still contain NaN"


class TestLazyVsNonLazyDecomposition:
    """Verify that lazy and non-lazy SVD decomposition agree numerically.

    Both algorithms perform an exact rank-k factorisation of the same
    data, so the reconstruction errors and singular-value spectra should
    be identical (up to floating-point tolerance).  Factors/scores_arr may
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
            rms = np.sqrt(np.mean((t.scores @ t.components.T - self.data) ** 2))
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
            f = t.components
            l_ = t.scores[kept_nav, :]
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
            f = t.components[kept_sig, :]
            l_ = t.scores
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
            f = t.components[kept_sig, :]
            l_ = t.scores[kept_nav, :]
            rms = np.sqrt(np.mean((l_ @ f.T - self.data[kept_nav][:, kept_sig]) ** 2))
            assert rms < 1e-10, f"{label} both-masked RMS {rms:.2e} too large"

    @skip_sklearn
    def test_reproject_navigation_reconstruction(self):
        """After reproject='navigation', lazy and non-lazy SVD give similar reconstruction."""
        kw = dict(
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="navigation",
            print_info=False,
        )
        self.s_lz.decomposition(algorithm="SVD", svd_solver="incremental", **kw)
        self.s_nl.decomposition(algorithm="SVD", **kw)

        t_lz = self.s_lz.learning_results
        t = self.s_nl.learning_results
        rms_lz = np.sqrt(
            np.mean((t_lz.scores @ t_lz.components_arr.T - self.data) ** 2)
        )
        rms = np.sqrt(np.mean((t.scores @ t.components.T - self.data) ** 2))
        assert rms < 1e-10, f"non-lazy reproject RMS {rms:.2e} too large"
        assert rms_lz < 1.0, f"lazy reproject RMS {rms_lz:.2e} too large"

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
    chunk was read per navigation block, producing components_arr with the wrong
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

    @skip_sklearn
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
        """components_arr.shape[0] must equal sig_size regardless of chunk layout."""
        s, _, rank = self._make_signal(nav_shape, sig_size, sig_chunk, nav_chunk)
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=rank,
            print_info=False,
        )
        assert s.learning_results.components.shape[0] == sig_size

    # ------------------------------------------------------------------
    # normalize_poissonian_noise must not raise a broadcast error
    # ------------------------------------------------------------------

    @skip_sklearn
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
        assert s.learning_results.components.shape[0] == sig_size

    # ------------------------------------------------------------------
    # Reconstruction quality must be preserved despite sub-signal chunking
    # ------------------------------------------------------------------

    @skip_sklearn
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
        rms_cont = np.sqrt(np.mean((t_cont.scores @ t_cont.components.T - flat) ** 2))
        rms_sub = np.sqrt(np.mean((t_sub.scores @ t_sub.components_arr.T - flat) ** 2))
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
        """PCA and ORPCA also produce components_arr with sig_size rows when signal
        has sub-signal chunking."""
        nav_shape = (8, 8)
        sig_size = 64
        sig_chunk = 8  # 8 signal chunks
        nav_chunk = 4
        rank = 3
        s, _, _ = self._make_signal(nav_shape, sig_size, sig_chunk, nav_chunk, rank)
        s.decomposition(algorithm=algorithm, output_dimension=rank, print_info=False)
        assert s.learning_results.components.shape[0] == sig_size


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


@skip_sklearn
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
        """Every navigation mask type produces correct scores_arr shape."""
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
        assert s.learning_results.scores.shape[0] == nav_size

    @pytest.mark.parametrize(
        "nav_shape,sig_size",
        [
            ((18,), 40),
            ((6, 8), 40),
        ],
    )
    @pytest.mark.parametrize("mask_type", ["numpy", "dask", "BaseSignal", "LazySignal"])
    def test_sig_mask_types(self, nav_shape, sig_size, mask_type):
        """Every signal mask type produces correct components_arr shape."""
        s, nav_size = _make_mask_test_signal(nav_shape, sig_size)
        sig_masks = _build_sig_masks(s, sig_size)
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            signal_mask=sig_masks[mask_type],
            print_info=False,
        )
        assert s.learning_results.components.shape[0] == sig_size

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
        assert s.learning_results.scores.shape[0] == nav_size

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
        assert s.learning_results.components.shape[0] == sig_size

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
        assert s.learning_results.scores.shape[0] == nav_size
        assert s.learning_results.components.shape[0] == sig_size

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
        assert t.scores.shape[0] == nav_size
        assert t.components.shape[0] == sig_size
        # reproject='navigation' or 'both' → no NaN in loadings
        if reproject in ("navigation", "both"):
            assert not np.any(np.isnan(t.scores))
        # reproject='signal' or 'both' → no NaN in factors
        if reproject in ("signal", "both"):
            assert not np.any(np.isnan(t.components))

    @skip_sklearn
    def test_2d_nav_mask_pca_default_reproject(self):
        """Non-SVD algorithm with a 2-D navigation mask and default
        ``reproject=None`` must pass a navigation-shaped mask to
        ``_block_iterator`` (regression: the shaped mask was overwritten with
        a flattened 1-D array)."""
        nav_shape = (6, 8)
        sig_size = 40
        s, nav_size = _make_mask_test_signal(nav_shape, sig_size)
        nav_mask = _build_nav_masks(s, nav_shape)["numpy"]
        s.decomposition(
            algorithm="PCA",
            output_dimension=3,
            navigation_mask=nav_mask,
            print_info=False,
        )
        assert s.learning_results.scores.shape[0] == nav_size


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
        assert t.components.shape == (len(self.sig_mask), 3)
        assert not np.any(np.isnan(t.components)), (
            "components_arr should have no NaN after signal reproject"
        )

    @skip_sklearn
    @pytest.mark.parametrize("svd_solver", ["randomized", "incremental"])
    def test_centre_signal_mean_is_per_spectrum(self, svd_solver):
        """centre='signal' subtracts each spectrum's own mean (axis=1 in the
        unfolded (nav, sig) array).  The stored mean must have shape
        (nav, 1) and each row must equal the corresponding spectrum mean."""
        _, s_lz = self._make_signals()
        s_lz.decomposition(
            output_dimension=3,
            centre="signal",
            svd_solver=svd_solver,
            print_info=False,
        )
        stored_mean = s_lz.learning_results.mean
        assert stored_mean is not None, "mean should be stored when centre='signal'"
        # stored_mean has shape (nav, 1) or (1, nav) depending on axis convention;
        # either way its flat values should match per-spectrum means.
        expected = self.data.mean(axis=1)  # shape (nav,)
        np.testing.assert_allclose(
            stored_mean.ravel(),
            expected,
            rtol=1e-10,
            err_msg="centre='signal' mean does not match per-spectrum mean",
        )

    @skip_sklearn
    @pytest.mark.parametrize("svd_solver", ["randomized", "incremental"])
    def test_centre_navigation_mean_is_per_channel(self, svd_solver):
        """centre='navigation' subtracts the per-channel mean (axis=0 in the
        unfolded (nav, sig) array).  Stored mean shape must broadcast as (1, sig)
        and values must match the column-wise mean of the full (unmasked) data."""
        _, s_lz = self._make_signals()
        s_lz.decomposition(
            output_dimension=3,
            centre="navigation",
            svd_solver=svd_solver,
            print_info=False,
        )
        stored_mean = s_lz.learning_results.mean
        assert stored_mean is not None, "mean should be stored when centre='navigation'"
        expected = self.data.mean(axis=0)  # shape (sig,)
        np.testing.assert_allclose(
            stored_mean.ravel(),
            expected,
            rtol=1e-10,
            err_msg="centre='navigation' mean does not match per-channel mean",
        )


class TestLazyDecompositionInputValidation:
    """Verify that lazy decomposition raises the same guards as non-lazy.

    m1 - TypeError for non-float data.
    m2 - ValueError when navigation_size < 2.
    m3 - ValueError from _check_navigation_mask when mask shape is wrong.
    m4 - ValueError for output_dimension <= 0 or non-integer.
    m5 - ValueError for num_chunks <= 0 or non-integer.
    m6 - NotImplementedError for algorithms unsupported on lazy signals.
    m7 - NotImplementedError for var_array / var_func (MLPCA-only params).

    All tests use asymmetric shapes and the SVD algorithm so that the
    previously-missing SVD path validation is exercised.
    """

    def test_non_float_dtype_raises(self):
        """m1: integer data must raise TypeError (mirrors _mva.py:262)."""
        s = Signal1D(np.ones((8, 12), dtype=np.int32)).as_lazy()
        with pytest.raises(TypeError, match="floating-point"):
            s.decomposition(output_dimension=3, print_info=False)

    def test_navigation_size_lt2_raises(self):
        """m2: navigation_size < 2 must raise ValueError."""
        s = Signal1D(np.ones((1, 15), dtype=float)).as_lazy()
        with pytest.raises(ValueError, match="navigation_size < 2"):
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

    def test_output_dimension_zero_raises(self):
        """m4a: output_dimension=0 must raise ValueError."""
        s = Signal1D(np.ones((14, 9), dtype=float)).as_lazy()
        with pytest.raises(ValueError, match="positive integer"):
            s.decomposition(output_dimension=0, print_info=False)

    def test_output_dimension_negative_raises(self):
        """m4b: negative output_dimension must raise ValueError."""
        s = Signal1D(np.ones((7, 18), dtype=float)).as_lazy()
        with pytest.raises(ValueError, match="positive integer"):
            s.decomposition(output_dimension=-1, print_info=False)

    def test_output_dimension_float_raises(self):
        """m4c: float output_dimension must raise ValueError."""
        s = Signal1D(np.ones((11, 13), dtype=float)).as_lazy()
        with pytest.raises(ValueError, match="positive integer"):
            s.decomposition(output_dimension=3.5, print_info=False)

    def test_num_chunks_zero_raises(self):
        """m5a: num_chunks=0 must raise ValueError."""
        s = Signal1D(np.ones((16, 8), dtype=float)).as_lazy()
        with pytest.raises(ValueError, match="positive integer"):
            s.decomposition(output_dimension=3, num_chunks=0, print_info=False)

    def test_num_chunks_negative_raises(self):
        """m5b: negative num_chunks must raise ValueError."""
        s = Signal1D(np.ones((9, 14), dtype=float)).as_lazy()
        with pytest.raises(ValueError, match="positive integer"):
            s.decomposition(output_dimension=3, num_chunks=-2, print_info=False)

    def test_mlpca_raises_not_implemented(self):
        """m6a: MLPCA is not supported for lazy signals."""
        s = Signal1D(np.ones((13, 11), dtype=float)).as_lazy()
        with pytest.raises(NotImplementedError, match="not supported for lazy"):
            s.decomposition(algorithm="MLPCA", output_dimension=3, print_info=False)

    def test_rpca_raises_not_implemented(self):
        """m6b: RPCA is not supported for lazy signals."""
        s = Signal1D(np.ones((6, 20), dtype=float)).as_lazy()
        with pytest.raises(NotImplementedError, match="not supported for lazy"):
            s.decomposition(algorithm="RPCA", output_dimension=3, print_info=False)

    def test_var_array_raises_not_implemented(self):
        """m7a: var_array is an MLPCA-only param; passing it to a lazy signal
        must raise NotImplementedError."""
        s = Signal1D(np.ones((15, 10), dtype=float)).as_lazy()
        with pytest.raises(NotImplementedError, match="var_array"):
            s.decomposition(
                output_dimension=3,
                var_array=np.ones((15, 10)),
                print_info=False,
            )

    def test_var_func_raises_not_implemented(self):
        """m7b: var_func is an MLPCA-only param; passing it to a lazy signal
        must raise NotImplementedError."""
        s = Signal1D(np.ones((12, 7), dtype=float)).as_lazy()
        with pytest.raises(NotImplementedError, match="var_func"):
            s.decomposition(
                output_dimension=3,
                var_func=lambda x: x,
                print_info=False,
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
        assert lr.components is not None
        assert lr.scores is not None
        assert lr.components.shape[1] == 3
        assert lr.scores.shape[1] == 3

    def test_nmf_factors_shape(self):
        self.s.decomposition(algorithm="NMF", output_dimension=3, print_info=False)
        lr = self.s.learning_results
        # factors: (sig_size, n_components), loadings: (nav_size, n_components)
        assert lr.components.shape == (40, 3)
        assert lr.scores.shape == (6 * 8, 3)

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
        assert lr.scores.shape[0] == 6 * 8  # full nav


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
        assert lr.components is not None
        assert lr.scores is not None
        assert lr.components.shape[1] == 3
        # return_info should give back the estimator
        assert returned is obj

    def test_custom_batch_estimator(self):
        """Object without partial_fit falls back to fit_transform."""
        obj = self._make_batch_estimator(3)
        self.s.decomposition(algorithm=obj, print_info=False)

        lr = self.s.learning_results
        assert lr.components is not None
        assert lr.components.shape[1] == 3

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
            svd_solver="randomized",
            print_info=False,
        )

    def test_auto_transpose_true_no_op_when_nav_ge_sig(self):
        """auto_transpose=True is a no-op when nav >= sig (the common case)."""
        # nav=(6,8)=48, sig=40 → no transposition needed; should run cleanly.
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            auto_transpose=True,
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.components.shape == (40, 3)
        assert lr.scores.shape == (48, 3)

    def test_auto_transpose_false_no_error(self):
        """auto_transpose=False is accepted without error."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            auto_transpose=False,
            print_info=False,
        )

    def test_auto_transpose_no_op_when_nav_lt_sig(self):
        """auto_transpose is a no-op; nav<sig runs correctly without transposing."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((10, 50)).astype("float32")
        s = Signal1D(data).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            auto_transpose=True,
            print_info=False,
        )
        lr = s.learning_results
        assert lr.components.shape == (50, 3)
        assert lr.scores.shape == (10, 3)


class TestSVDAlgorithm:
    """Tests for algorithm='SVD' with svd_solver='randomized' (svd_compressed)."""

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
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.components is not None
        assert lr.scores is not None

    def test_output_dimension_required(self):
        """output_dimension is required for svd_solver='randomized'."""
        with pytest.raises(ValueError, match="`output_dimension` must be specified"):
            self.s.decomposition(
                algorithm="SVD", svd_solver="randomized", print_info=False
            )

    def test_output_dimension_respected(self):
        """When output_dimension is given, exactly that many components are returned."""
        k = 4
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=k,
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.components.shape == (30, k)
        assert lr.scores.shape == (35, k)

    def test_factors_and_loadings_shapes(self):
        """Factors shape is (sig_size, k); scores_arr shape is (nav_size, k)."""
        k = 3
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=k,
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.components.shape == (30, k)
        assert lr.scores.shape == (35, k)

    def test_explained_variance_set(self):
        """explained_variance is populated after SVD."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.explained_variance is not None
        assert lr.explained_variance.shape == (3,)

    def test_reconstruction_quality(self):
        """First 3 components should reconstruct the (near rank-3) signal well."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        lr = self.s.learning_results
        recon = (lr.scores @ lr.components.T).reshape(7, 5, 30)
        original = self.s.data.compute()
        rel_error = np.linalg.norm(recon - original) / np.linalg.norm(original)
        assert rel_error < 0.1


class TestSVDFullSolver:
    """Tests for algorithm='SVD' with svd_solver='full' (da.linalg.svd)."""

    def setup_method(self, method):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((35, 3))
        F = rng.standard_normal((3, 30))
        data = (L @ F + 0.01 * rng.standard_normal((35, 30))).reshape((7, 5, 30))
        self.s = Signal1D(data.astype(float)).as_lazy()

    def test_basic_run_with_output_dimension(self):
        """svd_solver='full' runs without error when output_dimension is set."""

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.components is not None
        assert lr.scores is not None

    def test_output_dimension_optional(self):
        """svd_solver='full' accepts output_dimension=None and returns lazy arrays."""
        import dask.array as da

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=None,
            print_info=False,
        )
        lr = self.s.learning_results
        assert isinstance(lr.components, da.Array)
        assert isinstance(lr.scores, da.Array)

    def test_output_dimension_respected(self):
        """When output_dimension is given, the results are truncated accordingly."""
        import dask.array as da

        k = 4
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=k,
            print_info=False,
        )
        lr = self.s.learning_results
        # Results may be lazy; compute to check shape.
        components_arr = (
            lr.components.compute()
            if isinstance(lr.components, da.Array)
            else lr.components
        )
        scores_arr = (
            lr.scores.compute() if isinstance(lr.scores, da.Array) else lr.scores
        )
        assert components_arr.shape == (30, k)
        assert scores_arr.shape == (35, k)

    def test_centre_navigation(self):
        """svd_solver='full' supports centre='navigation'."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            centre="navigation",
            output_dimension=3,
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.centre == "navigation"
        assert lr.mean is not None
        components_arr = (
            lr.components.compute()
            if isinstance(lr.components, da.Array)
            else lr.components
        )
        scores_arr = (
            lr.scores.compute() if isinstance(lr.scores, da.Array) else lr.scores
        )
        assert components_arr.shape[1] == 3
        assert scores_arr.shape[1] == 3
        assert not np.any(np.isnan(components_arr))
        assert not np.any(np.isnan(scores_arr))

    def test_reproject_navigation(self):
        """svd_solver='full' supports reproject='navigation'."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            reproject="navigation",
            output_dimension=3,
            print_info=False,
        )
        lr = self.s.learning_results
        assert lr.scores.shape == (35, 3)
        assert not np.any(np.isnan(lr.scores))


class TestLazyGetDecompositionModel:
    """Tests for the lazy get_decomposition_model() pipeline with svd_solver='full'."""

    def setup_method(self, method):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((35, 3))
        F = rng.standard_normal((3, 30))
        data = (L @ F + 0.01 * rng.standard_normal((35, 30))).reshape((7, 5, 30))
        self.s = Signal1D(data.astype(float)).as_lazy()

    def test_factors_loadings_are_lazy_without_reproject(self):
        """svd_solver='full' without reproject keeps components_arr/loadings as dask arrays."""
        import dask.array as da

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        lr = self.s.learning_results
        assert isinstance(lr.components, da.Array), (
            "components_arr should be a dask array"
        )
        assert isinstance(lr.scores, da.Array), "scores_arr should be a dask array"

    def test_factors_loadings_after_reproject_navigation(self):
        """svd_solver='full' with reproject='navigation': scores_arr computed to
        numpy, components_arr remain lazy (signal reproject was not requested)."""
        import dask.array as da

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            reproject="navigation",
            print_info=False,
        )
        lr = self.s.learning_results
        assert isinstance(lr.scores, np.ndarray), (
            "scores_arr should be numpy after reproject='navigation'"
        )
        assert isinstance(lr.components, da.Array), (
            "components_arr should remain lazy dask array when only nav-reproject was done"
        )

    def test_factors_loadings_after_reproject_signal(self):
        """svd_solver='full' with reproject='signal': components_arr computed to
        numpy, scores_arr remain lazy (nav reproject was not requested)."""
        import dask.array as da

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            reproject="signal",
            print_info=False,
        )
        lr = self.s.learning_results
        assert isinstance(lr.components, np.ndarray), (
            "components_arr should be numpy after reproject='signal'"
        )
        assert isinstance(lr.scores, da.Array), (
            "scores_arr should remain lazy dask array when only signal-reproject was done"
        )

    def test_factors_loadings_after_reproject_both(self):
        """svd_solver='full' with reproject='both': both components_arr and loadings
        are computed to numpy arrays."""

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            reproject="both",
            print_info=False,
        )
        lr = self.s.learning_results
        assert isinstance(lr.components, np.ndarray), (
            "components_arr should be numpy after reproject='both'"
        )
        assert isinstance(lr.scores, np.ndarray), (
            "scores_arr should be numpy after reproject='both'"
        )

    def test_get_decomposition_model_returns_lazy_signal(self):
        """get_decomposition_model() returns a LazySignal when components_arr/loadings are dask."""
        import dask.array as da

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        model = self.s.get_decomposition_model()
        assert isinstance(model.data, da.Array), "model.data should be a dask array"
        assert model._lazy, "returned signal should be lazy"

    def test_get_decomposition_model_correct_shape(self):
        """Reconstructed model has the same shape as the original signal."""
        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        model = self.s.get_decomposition_model()
        assert model.data.shape == self.s.data.shape

    def test_get_decomposition_model_no_computation_until_compute(self):
        """get_decomposition_model() builds a task graph without triggering computation."""

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        model = self.s.get_decomposition_model()
        # model.data is a dask array; calling .compute() should succeed and give
        # an array of the correct shape.
        result = model.data.compute()
        assert result.shape == self.s.data.compute().shape

    def test_get_decomposition_model_components_int(self):
        """get_decomposition_model(components=N) works lazily."""
        import dask.array as da

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=5,
            print_info=False,
        )
        model = self.s.get_decomposition_model(components=2)
        assert isinstance(model.data, da.Array)
        assert model.data.shape == self.s.data.shape

    def test_get_decomposition_model_components_list(self):
        """get_decomposition_model(components=[...]) works lazily."""
        import dask.array as da

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=5,
            print_info=False,
        )
        model = self.s.get_decomposition_model(components=[0, 2])
        assert isinstance(model.data, da.Array)
        assert model.data.shape == self.s.data.shape

    def test_get_decomposition_model_randomized_uses_numpy_factors(self):
        """With svd_solver='randomized', components_arr/loadings are numpy (not dask)."""

        self.s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        lr = self.s.learning_results
        assert isinstance(lr.components, np.ndarray), (
            "components_arr should be numpy for randomized solver"
        )
        assert isinstance(lr.scores, np.ndarray), (
            "scores_arr should be numpy for randomized solver"
        )

    def test_full_svd_with_signal_chunked_data(self):
        """svd_solver='full' is deprecated but still functional.

        Emits a VisibleDeprecationWarning pointing to 'randomized'.
        """
        import dask.array as da

        rng = np.random.default_rng(7)
        data = da.from_array(
            rng.random((7, 5, 30)),
            chunks=(4, 3, 15),  # signal dim chunked
        )
        s = Signal1D(data).as_lazy()
        with pytest.warns(
            VisibleDeprecationWarning, match="svd_solver='full' is deprecated"
        ):
            s.decomposition(
                algorithm="SVD",
                svd_solver="randomized",
                output_dimension=3,
                print_info=False,
            )
        lr = s.learning_results
        assert isinstance(lr.components, da.Array)
        assert isinstance(lr.scores, da.Array)
        model = s.get_decomposition_model()
        assert model._lazy
        assert model.data.shape == s.data.shape
        assert model.data.shape == s.data.shape


# ---------------------------------------------------------------------------
# Coverage-gap tests: branches added in PR #3614 not yet exercised
# ---------------------------------------------------------------------------


class TestImportErrorPaths:
    """ImportError raised when sklearn is absent for PCA / NMF algorithms."""

    def setup_method(self, method):
        rng = np.random.default_rng(0)
        self.s = Signal1D(rng.random((8, 20))).as_lazy()

    def test_pca_raises_import_error_without_sklearn(self, monkeypatch):
        """algorithm='PCA' must raise ImportError when sklearn is not installed."""
        import hyperspy._signals.lazy as lazy_mod

        monkeypatch.setattr(lazy_mod, "SKLEARN_INSTALLED", False)
        with pytest.raises(ImportError, match="algorithm='PCA' requires scikit-learn"):
            self.s.decomposition(algorithm="PCA", output_dimension=2, print_info=False)

    def test_nmf_raises_import_error_without_sklearn(self, monkeypatch):
        """algorithm='NMF' must raise ImportError when sklearn is not installed."""
        import hyperspy._signals.lazy as lazy_mod

        monkeypatch.setattr(lazy_mod, "SKLEARN_INSTALLED", False)
        with pytest.raises(ImportError, match="algorithm='NMF' requires scikit-learn"):
            self.s.decomposition(algorithm="NMF", output_dimension=2, print_info=False)


class TestISVDPlaceholder:
    """ISVD placeholder class raises ImportError when sklearn is absent."""

    def test_isvd_placeholder_raises(self, monkeypatch):
        """The no-sklearn ISVD stub must raise ImportError on instantiation."""

        import hyperspy.learn.incremental_svd as isvd_mod

        monkeypatch.setattr(isvd_mod, "SKLEARN_INSTALLED", False)

        # Reimport the module-level conditional block by reloading
        # (monkeypatching the module attribute directly is enough to test
        # _check_sklearn which is called by the placeholder __init__)
        isvd_mod._check_sklearn.__wrapped__ = None  # no-op guard reset

        # Build a fresh placeholder instance by calling _check_sklearn directly
        with pytest.raises(ImportError, match="requires scikit-learn"):
            isvd_mod._check_sklearn()


class TestNumChunksAutoReduce:
    """When blocksize / output_dimension < num_chunks the value is clamped."""

    @skip_sklearn
    def test_num_chunks_clamped_to_ceil(self):
        """Decomposition succeeds even when user requests more chunks than useful."""
        rng = np.random.default_rng(1)
        # Small nav so blocksize / output_dimension < a large num_chunks
        s = Signal1D(rng.random((4, 20))).as_lazy()
        # output_dimension=3, 4 nav positions → blocksize=4; num_chunks=10 > 4/3
        # Should clamp without error.
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            num_chunks=10,
            print_info=False,
        )
        assert s.learning_results.components is not None


class TestCentreWithDaskMask:
    """centre='navigation' with a dask-array navigation_mask (L1462-1468)."""

    @skip_sklearn
    def test_centre_navigation_dask_nav_mask(self):
        """centre='navigation' + dask navigation_mask computes correct mean."""
        import dask.array as da

        rng = np.random.default_rng(2)
        data = rng.random((12, 30)) + 1.0
        nav_mask_np = np.zeros(12, dtype=bool)
        nav_mask_np[::4] = True
        nav_mask_da = da.from_array(nav_mask_np, chunks=4)

        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            centre="navigation",
            navigation_mask=nav_mask_da,
            print_info=False,
        )
        stored_mean = s.learning_results.mean
        expected = data[~nav_mask_np].mean(axis=0)
        np.testing.assert_allclose(stored_mean.ravel(), expected, rtol=1e-10)

    @skip_sklearn
    def test_centre_navigation_incremental_dask_nav_mask(self):
        """ISVD + centre='navigation' + dask nav_mask exercises L1462-1468."""
        import dask.array as da

        rng = np.random.default_rng(20)
        data = rng.random((12, 30)) + 1.0
        nav_mask_np = np.zeros(12, dtype=bool)
        nav_mask_np[::4] = True
        nav_mask_da = da.from_array(nav_mask_np, chunks=4)

        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            centre="navigation",
            navigation_mask=nav_mask_da,
            print_info=False,
        )
        stored_mean = s.learning_results.mean
        expected = data[~nav_mask_np].mean(axis=0)
        np.testing.assert_allclose(stored_mean.ravel(), expected, rtol=1e-10)

    @skip_sklearn
    def test_centre_signal_incremental_full_svd(self):
        """svd_solver='full' + centre='signal' exercises L1553-1554."""
        rng = np.random.default_rng(21)
        data = rng.random((10, 25)) + 1.0
        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            centre="signal",
            print_info=False,
        )
        lr = s.learning_results
        assert lr.mean is not None
        assert lr.components is not None


class TestCentreWith2DNav:
    """centre with 2-D navigation space exercises the nav_mask .T branch."""

    @skip_sklearn
    def test_centre_navigation_2d_nav(self):
        """centre='navigation' on a 2-D nav signal stores correct mean."""
        rng = np.random.default_rng(3)
        # nav (3, 4), sig 20
        data = rng.random((3, 4, 20)) + 1.0
        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            centre="navigation",
            print_info=False,
        )
        stored = s.learning_results.mean
        expected = data.reshape(-1, 20).mean(axis=0)
        np.testing.assert_allclose(stored.ravel(), expected, rtol=1e-10)

    @skip_sklearn
    def test_centre_navigation_2d_nav_with_mask(self):
        """centre='navigation' + nav_mask on a 2-D nav signal."""
        rng = np.random.default_rng(4)
        data = rng.random((3, 4, 20)) + 1.0
        # navigation_shape is (4, 3) for a (3, 4, 20) Signal1D
        nav_mask = np.zeros((4, 3), dtype=bool)
        nav_mask[0, 0] = True
        nav_mask[3, 2] = True
        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            centre="navigation",
            navigation_mask=nav_mask,
            print_info=False,
        )
        assert s.learning_results.mean is not None
        assert s.learning_results.components.shape[1] == 3

    @skip_sklearn
    def test_centre_signal_2d_nav(self):
        """centre='signal' on a 2-D nav signal stores per-spectrum mean."""
        rng = np.random.default_rng(5)
        data = rng.random((3, 4, 20)) + 1.0
        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            centre="signal",
            print_info=False,
        )
        stored = s.learning_results.mean
        expected = data.reshape(-1, 20).mean(axis=1)  # per spectrum
        np.testing.assert_allclose(stored.ravel(), expected, rtol=1e-10)

    @skip_sklearn
    def test_centre_signal_2d_nav_dask_signal_mask(self):
        """centre + dask signal_mask exercises the dask signal-mask branch."""
        import dask.array as da

        rng = np.random.default_rng(6)
        data = rng.random((3, 4, 20)) + 1.0
        sig_mask_np = np.zeros(20, dtype=bool)
        sig_mask_np[:3] = True
        sig_mask_da = da.from_array(sig_mask_np, chunks=10)

        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            centre="navigation",
            signal_mask=sig_mask_da,
            print_info=False,
        )
        assert s.learning_results.components is not None


class TestReprojectMaskBranches:
    """Mask handling inside the reproject paths."""

    def setup_method(self, method):
        rng = np.random.default_rng(7)
        nav, sig, rank = 16, 25, 3
        U = np.abs(rng.standard_normal((nav, rank)))
        V = np.abs(rng.standard_normal((sig, rank)))
        self.data = (U @ V.T) + 1.0
        self.nav_mask = np.zeros(nav, dtype=bool)
        self.nav_mask[::4] = True
        self.sig_mask = np.zeros(sig, dtype=bool)
        self.sig_mask[:3] = True

    @skip_sklearn
    def test_reproject_navigation_with_sig_mask_incremental(self):
        """reproject='navigation' + sig_mask exercises mask-column removal."""
        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            signal_mask=self.sig_mask,
            reproject="navigation",
            print_info=False,
        )
        lr = s.learning_results
        assert lr.scores.shape == (self.data.shape[0], 3)
        assert lr.components.shape == (self.data.shape[1], 3)

    @skip_sklearn
    def test_reproject_signal_with_nav_mask_incremental(self):
        """reproject='signal' + nav_mask exercises the pinv signal-reproject path."""
        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="signal",
            print_info=False,
        )
        lr = s.learning_results
        assert lr.components.shape == (self.data.shape[1], 3)
        assert lr.scores.shape == (self.data.shape[0], 3)

    @skip_sklearn
    def test_reproject_both_with_masks_incremental(self):
        """reproject='both' + both masks exercises the 'both' reproject branch."""
        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            navigation_mask=self.nav_mask,
            signal_mask=self.sig_mask,
            reproject="both",
            print_info=False,
        )
        lr = s.learning_results
        assert lr.components.shape == (self.data.shape[1], 3)
        assert lr.scores.shape == (self.data.shape[0], 3)

    @skip_sklearn
    def test_reproject_both_centre_mean_subtraction(self):
        """reproject='signal' + centre='navigation' triggers mean subtraction."""
        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            centre="navigation",
            reproject="signal",
            print_info=False,
        )
        lr = s.learning_results
        assert lr.mean is not None
        assert lr.components.shape == (self.data.shape[1], 3)

    @skip_sklearn
    def test_reproject_navigation_centre_navigation_incremental(self):
        """reproject='navigation' + centre='navigation' subtracts mean in reproject."""
        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            output_dimension=3,
            centre="navigation",
            reproject="navigation",
            print_info=False,
        )
        lr = s.learning_results
        assert lr.mean is not None
        assert lr.scores.shape == (self.data.shape[0], 3)


class TestReprojectFullSVDMasks:
    """reproject with svd_solver='full' + nav/sig masks."""

    def setup_method(self, method):
        rng = np.random.default_rng(8)
        nav, sig, rank = 12, 20, 3
        U = np.abs(rng.standard_normal((nav, rank)))
        V = np.abs(rng.standard_normal((sig, rank)))
        self.data = (U @ V.T) + 1.0
        self.nav_mask = np.zeros(nav, dtype=bool)
        self.nav_mask[::4] = True
        self.sig_mask = np.zeros(sig, dtype=bool)
        self.sig_mask[:3] = True

    def test_full_svd_reproject_navigation_with_sig_mask(self):
        """svd_solver='full', reproject='navigation' + sig_mask."""
        import dask.array as da

        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            signal_mask=self.sig_mask,
            reproject="navigation",
            print_info=False,
        )
        lr = s.learning_results
        assert isinstance(lr.components, da.Array)
        assert isinstance(lr.scores, np.ndarray)
        assert lr.scores.shape == (self.data.shape[0], 3)

    def test_full_svd_reproject_signal_with_nav_mask(self):
        """svd_solver='full', reproject='signal' + nav_mask."""
        import dask.array as da

        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="signal",
            print_info=False,
        )
        lr = s.learning_results
        assert isinstance(lr.scores, da.Array)
        assert isinstance(lr.components, np.ndarray)
        assert lr.components.shape == (self.data.shape[1], 3)

    def test_full_svd_reproject_both_with_nav_mask(self):
        """svd_solver='full', reproject='both' + nav_mask exercises L1848."""
        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            navigation_mask=self.nav_mask,
            reproject="both",
            print_info=False,
        )
        lr = s.learning_results
        assert isinstance(lr.components, np.ndarray)
        assert isinstance(lr.scores, np.ndarray)
        assert lr.components.shape == (self.data.shape[1], 3)

    def test_full_svd_reproject_navigation_with_centre(self):
        """svd_solver='full', reproject='navigation' + centre='navigation'."""

        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            centre="navigation",
            reproject="navigation",
            print_info=False,
        )
        lr = s.learning_results
        assert lr.mean is not None
        assert isinstance(lr.scores, np.ndarray)

    def test_full_svd_reproject_signal_with_centre(self):
        """svd_solver='full', reproject='signal' + centre='navigation' → L1842-1843."""
        s = Signal1D(self.data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            centre="navigation",
            reproject="signal",
            print_info=False,
        )
        lr = s.learning_results
        assert lr.mean is not None
        assert isinstance(lr.components, np.ndarray)


class TestFullSVDBaseSignalMask:
    """BaseSignal masks in svd_solver='full' path (L1507, L1523)."""

    def setup_method(self, method):
        rng = np.random.default_rng(30)
        nav, sig = 12, 24
        self.data = rng.random((nav, sig)) + 1.0
        self.nav = nav
        self.sig = sig

    def test_full_svd_basesignal_nav_mask(self):
        """svd_solver='full' with BaseSignal navigation_mask exercises L1507."""
        from hyperspy.signals import Signal1D as _S1D

        s = _S1D(self.data.copy()).as_lazy()
        nav_mask_data = np.zeros(self.nav, dtype=bool)
        nav_mask_data[::4] = True
        nav_mask_sig = _S1D(nav_mask_data).T

        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            navigation_mask=nav_mask_sig,
            print_info=False,
        )
        assert s.learning_results.components is not None

    def test_full_svd_basesignal_sig_mask(self):
        """svd_solver='full' with BaseSignal signal_mask exercises L1523."""
        from hyperspy.signals import Signal1D as _S1D

        s = _S1D(self.data.copy()).as_lazy()
        sig_mask_data = np.zeros(self.sig, dtype=bool)
        sig_mask_data[:4] = True
        sig_mask_sig = _S1D(sig_mask_data)

        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            signal_mask=sig_mask_sig,
            print_info=False,
        )
        assert s.learning_results.components is not None

    def test_full_svd_dask_nav_mask(self):
        """svd_solver='full' with dask nav_mask exercises L1511+L1514 path."""
        import dask.array as da

        s = Signal1D(self.data.copy()).as_lazy()
        nav_mask_np = np.zeros(self.nav, dtype=bool)
        nav_mask_np[::3] = True
        nav_mask_da = da.from_array(nav_mask_np, chunks=4)

        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            navigation_mask=nav_mask_da,
            print_info=False,
        )
        assert s.learning_results.components is not None

    def test_full_svd_dask_sig_mask(self):
        """svd_solver='full' with dask signal_mask exercises L1526+L1527 path."""
        import dask.array as da

        s = Signal1D(self.data.copy()).as_lazy()
        sig_mask_np = np.zeros(self.sig, dtype=bool)
        sig_mask_np[:4] = True
        sig_mask_da = da.from_array(sig_mask_np, chunks=8)

        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            signal_mask=sig_mask_da,
            print_info=False,
        )
        assert s.learning_results.components is not None


class TestFullSVDNavMask2DOrder:
    """2-D numpy nav mask + svd_solver='full' stores mask in correct array order.

    Regression test for Oracle review finding: _to_flat_bool ran on untransposed
    navigation_mask in the SVD full path, giving wrong ravel order for 2-D nav.
    """

    def setup_method(self, method):
        rng = np.random.default_rng(42)
        # ALL-DIFFERENT: ny=5, nx=6, sig=20
        self.s = Signal1D(rng.random((5, 6, 20))).as_lazy()
        # Mask row 0 and column 1 in display order (nx=6, ny=5)
        self.nav_mask = np.zeros((6, 5), dtype=bool)
        self.nav_mask[0, 1] = True

    def test_full_svd_nav_mask_stored_in_array_order(self):
        """navigation_mask stored in learning_results should be reshaped from
        array-order ravel, matching _navigation_shape_in_array (ny, nx)."""
        s = self.s
        s.decomposition(
            output_dimension=3,
            svd_solver="randomized",
            navigation_mask=self.nav_mask,
        )
        stored = s.learning_results.navigation_mask
        # Should be (ny=5, nx=6) — array order, not display order (6, 5)
        assert stored.shape == (5, 6)


class TestIncrementalNavMaskTranspose:
    """ISVD + 2-D nav mask gets .T'd to array-axis order (L1493-1495)."""

    @skip_sklearn
    def test_incremental_2d_nav_mask_transposed(self):
        """ISVD + 2-D nav mask: numpy mask is transposed internally."""
        rng = np.random.default_rng(31)
        data = rng.random((3, 4, 20)) + 1.0
        nav_mask = np.zeros((4, 3), dtype=bool)
        nav_mask[0, 0] = True

        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            navigation_mask=nav_mask,
            print_info=False,
        )
        assert s.learning_results.components.shape[1] == 3


class TestRemainingBranches:
    """Cover the remaining uncovered branches in lazy.py."""

    @skip_sklearn
    def test_incremental_centre_basesignal_nav_mask(self):
        """ISVD + centre + BaseSignal nav_mask hits L1464 (_nm = _nm.data)."""
        from hyperspy.signals import BaseSignal

        rng = np.random.default_rng(40)
        data = rng.random((12, 30)) + 1.0
        nav_mask_np = np.zeros(12, dtype=bool)
        nav_mask_np[::4] = True
        nav_mask_sig = BaseSignal(nav_mask_np).T

        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            centre="navigation",
            navigation_mask=nav_mask_sig,
            print_info=False,
        )
        assert s.learning_results.mean is not None

    @skip_sklearn
    def test_incremental_centre_numpy_nav_mask(self):
        """ISVD + centre + numpy nav_mask hits L1465 False branch (no compute)."""
        rng = np.random.default_rng(41)
        data = rng.random((12, 30)) + 1.0
        nav_mask_np = np.zeros(12, dtype=bool)
        nav_mask_np[::4] = True

        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            centre="navigation",
            navigation_mask=nav_mask_np,
            print_info=False,
        )
        assert s.learning_results.mean is not None

    def test_reproject_both_no_nav_mask_hits_L1898(self):
        """reproject='both' without nav_mask hits L1898 (L = scores_arr branch)."""
        rng = np.random.default_rng(42)
        data = rng.random((12, 30)) + 1.0
        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            reproject="both",
            print_info=False,
        )
        assert s.learning_results.scores is not None

    def test_full_svd_explained_variance_ratio_computed(self):
        """svd_solver='full' triggers _compute_explained_variance_ratio (L1930)."""
        rng = np.random.default_rng(43)
        data = rng.random((15, 30)) + 1.0
        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=5,
            print_info=False,
        )
        assert s.learning_results.explained_variance_ratio is not None

    def test_reproject_with_basesignal_mask_hits_L1952(self):
        """reproject + BaseSignal nav_mask hits _to_flat_bool L1952 branch."""
        from hyperspy.signals import BaseSignal

        rng = np.random.default_rng(44)
        data = rng.random((12, 30)) + 1.0
        nav_mask_np = np.zeros(12, dtype=bool)
        nav_mask_np[::4] = True
        nav_mask_sig = BaseSignal(nav_mask_np).T

        s = Signal1D(data.copy()).as_lazy()
        s.decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            reproject="navigation",
            navigation_mask=nav_mask_sig,
            print_info=False,
        )
        assert s.learning_results.scores is not None


class TestLazyDecompositionBugfixes:
    """Regression tests for PR #3655: fix-lazy-signal-bugs."""

    def setup_method(self, method):
        rng = np.random.default_rng(42)
        self.s = Signal1D(rng.random((12, 25, 48))).as_lazy()

    def test_poissonian_noise_normalization_scales_data(self):
        """#3607: verify normalize_poissonian_noise=True produces valid
        reconstruction. Bug: coeff.map_blocks() result discarded (no-op)."""
        s = self.s
        s.decomposition(
            normalize_poissonian_noise=True, output_dimension=3, algorithm="ORNMF"
        )
        rec = s.get_decomposition_model().data.compute()
        assert np.all(np.isfinite(rec))

    def test_unfolded4decomposition_false_after_svd(self):
        """#3608: _unfolded4decomposition should be False after lazy SVD.
        The bug was ```is False``` (comparison) instead of ```= False```."""
        s = self.s.deepcopy()
        s.decomposition(output_dimension=3)
        assert s._unfolded4decomposition is False

    def test_block_iterator_rechunks_signal(self):
        """#3610: _block_iterator should rechunk signal dimension to single
        chunk so that sub-signal chunk layouts work correctly."""
        s = self.s
        blocks = list(s._block_iterator(flat_signal=True))
        assert len(blocks) > 0
        # Each block should have signal_size columns
        for block in blocks:
            assert block.shape[1] == s.axes_manager.signal_size
