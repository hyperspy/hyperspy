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

"""Regression tests for hyperspy.learn.incremental_svd.ISVD.

The ISVD class disables sklearn IncrementalPCA centering by overriding the
mean_ property.  These tests verify that the centering override keeps working
correctly if sklearn's internal implementation changes.  A failure here is a
signal to audit the ISVD implementation against the new sklearn version.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn is required for ISVD tests")

from hyperspy.learn.incremental_svd import ISVD


class TestISVDNoCentering:
    """Verify that ISVD does not subtract the column mean from the data."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        # Data with a large constant offset so that centering would
        # produce clearly different results.
        self.X = rng.standard_normal((200, 40)) + 50.0
        self.n_components = 5

    def _fit_isvd(self, X, n_components, n_chunks=4):
        isvd = ISVD(n_components=n_components)
        for chunk in np.array_split(X, n_chunks):
            isvd.partial_fit(chunk)
        return isvd

    def test_mean_property_is_zeros_after_fit(self):
        """After fitting on shifted data, mean_ must read back as all zeros.

        If sklearn changes how IncrementalPCA stores or uses mean_ internally
        this test will catch it before silent numerical breakage occurs.
        """
        isvd = self._fit_isvd(self.X, self.n_components)
        mean = isvd.mean_
        assert mean.shape == (self.X.shape[1],), (
            f"mean_ shape {mean.shape} != ({self.X.shape[1]},)"
        )
        np.testing.assert_array_equal(
            mean,
            0.0,
            err_msg=(
                "ISVD.mean_ must always be zero to disable centering. "
                "A non-zero value means sklearn changed the mean_ usage "
                "internals and the ISVD override no longer works."
            ),
        )

    def test_mean_setter_converts_array_to_zeros(self):
        """Assigning a non-zero array to mean_ must store zeros."""
        isvd = ISVD(n_components=3)
        isvd.mean_ = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(isvd.mean_, 0.0)

    def test_mean_setter_preserves_scalar(self):
        """Assigning a scalar to mean_ (sklearn init convention) is preserved."""
        isvd = ISVD(n_components=3)
        isvd.mean_ = 0.0
        assert isvd.mean_ == 0.0

    def test_transform_does_not_center(self):
        """ISVD.transform(X) must NOT subtract the column mean.

        With a strong mean offset, centered loadings would have near-zero
        column means.  Uncentered loadings inherit the offset of X projected
        onto the components, so their column means are far from zero.
        """
        from sklearn.decomposition import IncrementalPCA

        isvd = self._fit_isvd(self.X, self.n_components)
        isvd_scores = isvd.transform(self.X)

        # IncrementalPCA centers the data, so its scores are approximately
        # zero-mean along each component.
        ipca = IncrementalPCA(n_components=self.n_components)
        for chunk in np.array_split(self.X, 4):
            ipca.partial_fit(chunk)
        ipca_scores = ipca.transform(self.X)

        # IncrementalPCA scores are near-zero mean (centered output).
        np.testing.assert_allclose(
            ipca_scores.mean(axis=0),
            0.0,
            atol=0.5,
            err_msg="IncrementalPCA scores should be approximately zero-mean",
        )

        # ISVD scores are NOT near-zero mean because no centering was applied.
        isvd_col_means = np.abs(isvd_scores.mean(axis=0))
        assert isvd_col_means.max() > 1.0, (
            "ISVD scores should have non-zero column means when data has a "
            "large mean offset — centering appears to be active, which is wrong."
        )

    def test_reconstruction_consistent_with_numpy_svd(self):
        """The subspace spanned by ISVD components matches numpy SVD.

        ISVD is an incremental approximation so components/scores may differ
        in sign and order, but the low-rank reconstruction X ≈ L @ F.T should
        agree with the numpy reference up to a generous tolerance.
        """
        isvd = self._fit_isvd(self.X, self.n_components)
        isvd_components = isvd.components_.T  # (n_features, k)
        isvd_scores = isvd.transform(self.X)  # (n_samples, k)
        isvd_recon = isvd_scores @ isvd_components.T

        # Numpy SVD reference (no centering).
        _, S, Vt = np.linalg.svd(self.X, full_matrices=False)
        Vt_k = Vt[: self.n_components]
        # Project X onto the top-k right singular vectors and back.
        numpy_recon = (self.X @ Vt_k.T) @ Vt_k

        # The two reconstructions should be close; ISVD accumulates small
        # floating-point errors so we allow a loose tolerance.
        np.testing.assert_allclose(isvd_recon, numpy_recon, atol=2.0, rtol=0.05)

    def test_explained_variance_ratio_sums_to_at_most_one(self):
        """explained_variance_ratio_ values must be in (0, 1] and sum ≤ 1."""
        isvd = self._fit_isvd(self.X, self.n_components)
        evr = isvd.explained_variance_ratio_
        assert evr.shape == (self.n_components,)
        assert np.all(evr > 0), "All explained variance ratios must be positive"
        assert np.all(evr <= 1.0), "No explained variance ratio may exceed 1"
        assert evr.sum() <= 1.0 + 1e-10, (
            f"explained_variance_ratio_ sums to {evr.sum():.6f} > 1"
        )
