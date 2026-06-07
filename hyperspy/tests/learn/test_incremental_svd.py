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

The ISVD class implements plain incremental SVD (no centering) by
overriding C{partial_fit} of C{sklearn.decomposition.IncrementalPCA}.
These tests verify that the centering override keeps working correctly
if sklearn's internal implementation changes.  A failure here is a
signal to audit the ISVD implementation against the new sklearn version.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn is required for ISVD tests")

from hyperspy.learn.incremental_svd import ISVD


class TestISVDNoCentering:
    """Verify that ISVD computes plain SVD without subtracting the column mean."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        U = rng.standard_normal((200, 5))
        V = rng.standard_normal((40, 5))
        self.X = U @ V.T  # exact rank-5 data
        self.n_components = 5

    def _fit_isvd(self, X, n_components, n_chunks=4):
        isvd = ISVD(n_components=n_components)
        for chunk in np.array_split(X, n_chunks):
            isvd.partial_fit(chunk)
        return isvd

    # ---------- mean_ property tests ----------

    def test_mean_property_is_zeros_after_fit(self):
        """After fitting, mean_ must read back as all zeros.

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

    # ---------- centering tests ----------

    def test_transform_does_not_center(self):
        """ISVD.transform(X) must NOT subtract the column mean.

        With a strong mean offset, centered loadings would have near-zero
        column means.  Uncentered loadings inherit the offset of X projected
        onto the components, so their column means are far from zero.
        """
        from sklearn.decomposition import IncrementalPCA

        X_offset = self.X + 50.0

        isvd = self._fit_isvd(X_offset, self.n_components)
        isvd_loadings = isvd.transform(X_offset)

        # IncrementalPCA centers the data, so its loadings are approximately
        # zero-mean along each component.
        ipca = IncrementalPCA(n_components=self.n_components)
        for chunk in np.array_split(X_offset, 4):
            ipca.partial_fit(chunk)
        ipca_loadings = ipca.transform(X_offset)

        # IncrementalPCA loadings are near-zero mean (centered output).
        np.testing.assert_allclose(
            ipca_loadings.mean(axis=0),
            0.0,
            atol=0.5,
            err_msg="IncrementalPCA loadings should be approximately zero-mean",
        )

        # ISVD loadings are NOT near-zero mean because no centering was applied.
        isvd_col_means = np.abs(isvd_loadings.mean(axis=0))
        assert isvd_col_means.max() > 1.0, (
            "ISVD loadings should have non-zero column means when data has a "
            "large mean offset — centering appears to be active, which is wrong."
        )

    # ---------- reconstruction quality test ----------

    def test_reconstruction_consistent_with_numpy_svd(self):
        """The subspace spanned by ISVD matches numpy SVD near-exactly.

        For exact low-rank data, the incremental (batched) SVD algorithm
        preserves the subspace, so the rank-k reconstruction should match
        numpy's full SVD reconstruction to near machine precision.
        """
        isvd = self._fit_isvd(self.X, self.n_components)
        isvd_recon = isvd.transform(self.X) @ isvd.components_

        # Numpy SVD reference (no centering).
        _, S, Vt = np.linalg.svd(self.X, full_matrices=False)
        Vt_k = Vt[: self.n_components]
        numpy_recon = (self.X @ Vt_k.T) @ Vt_k

        np.testing.assert_allclose(isvd_recon, numpy_recon, atol=1e-10)

    # ---------- explained variance tests ----------

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
