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

"""Out-of-core (incremental) SVD via a no-centering subclass of
:class:`sklearn.decomposition.IncrementalPCA`.

:class:`sklearn.decomposition.IncrementalPCA` always subtracts a running mean
from each batch, which turns the decomposition into a PCA rather than a plain
SVD.  :class:`ISVD` disables this centering by overriding the ``mean_``
property so that it always reads back as zeros, making both the
``partial_fit`` mean-correction step and the ``transform`` mean-shift
step no-ops while leaving the rest of the sklearn implementation intact.
"""

import importlib

import numpy as np

SKLEARN_INSTALLED = importlib.util.find_spec("sklearn") is not None


def _check_sklearn():
    if not SKLEARN_INSTALLED:
        raise ImportError(
            "ISVD requires scikit-learn. Install it with:  pip install scikit-learn"
        )


if SKLEARN_INSTALLED:
    from sklearn.decomposition import IncrementalPCA as _IncrementalPCA

    class ISVD(_IncrementalPCA):
        """Out-of-core incremental SVD (no centering).

        A subclass of :class:`sklearn.decomposition.IncrementalPCA` that
        disables centering so the decomposition computes a plain SVD rather
        than PCA.  Data is fed in batches via ``partial_fit``; after all
        batches have been processed, call ``transform`` to obtain the
        loadings.

        The centering is disabled at two levels.  The ``mean_`` property
        always returns zeros, which neutralises the mean-shift in
        sklearn's ``transform``.  The ``partial_fit`` method is completely
        overridden to implement plain incremental SVD — data is never
        mean-subtracted during fitting, and no mean-correction term is
        added between batches.

        Parameters
        ----------
        n_components : int
            Number of singular components to compute.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`sklearn.decomposition.IncrementalPCA`.

        Attributes
        ----------
        singular_values_ : ndarray
            Singular values after fitting, shape ``(n_components,)``.
        components_ : ndarray
            Right singular vectors (rows are components), shape
            ``(n_components, n_features)``.
        explained_variance_ : ndarray
            Approximate explained variance as computed by sklearn's
            IncrementalPCA (``S² / (n_samples - 1)``).  Note that this
            uses Bessel's correction and may be inconsistent with HyperSpy's
            standard ``S² / N`` formula used by all other decomposition
            paths.  The ``LazySignal.decomposition`` method overrides this
            attribute for the ``learning_results`` output.
        explained_variance_ratio_ : ndarray
            Fraction of the top-``n_components`` variance captured by each
            component, computed as ``ev / ev.sum()`` — the same
            normalisation used by HyperSpy's eager and lazy SVD paths.
            (The ``LazySignal.decomposition`` method recomputes this
            independently from ``obj.singular_values_``, so the
            attribute on the ISVD object is used only for standalone
            ISVD usage.)

        Examples
        --------
        >>> import numpy as np
        >>> from hyperspy.learn.incremental_svd import ISVD
        >>> X = np.random.randn(200, 50)
        >>> obj = ISVD(n_components=3)
        >>> for chunk in np.array_split(X, 4):
        ...     obj.partial_fit(chunk)
        >>> factors = obj.components_.T          # shape (n_features, n_components)
        >>> loadings = obj.transform(X)          # shape (n_samples, n_components)
        """

        def partial_fit(self, X, y=None, check_input=False):
            """Fit one batch without centering (plain incremental SVD).

            Overrides :meth:`sklearn.decomposition.IncrementalPCA.partial_fit`
            to skip all centering and mean-correction steps.  Data is never
            mean-subtracted inside this method, so the decomposition is a
            plain SVD rather than PCA.

            Parameters
            ----------
            X : ndarray, shape (n_samples, n_features)
                Batch of training data.
            y : Ignored
                Exists for API compatibility with sklearn.
            check_input : bool
                Ignored (HyperSpy always passes ``False``).

            Returns
            -------
            self : ISVD
                The fitted estimator.
            """
            from scipy import linalg
            from sklearn.utils.extmath import svd_flip

            n_samples, n_features = X.shape

            # --- first-call initialization ---
            if not hasattr(self, "n_samples_seen_"):
                self.n_samples_seen_ = 0

                if self.n_components is None:
                    self.n_components_ = min(n_samples, n_features)
                elif self.n_components > n_features:
                    raise ValueError(
                        f"n_components={self.n_components} must be less "
                        f"than or equal to n_features={n_features}"
                    )
                elif self.n_components > n_samples:
                    raise ValueError(
                        f"n_components={self.n_components} must be less "
                        f"than or equal to the batch number of samples "
                        f"{n_samples} for the first partial_fit call."
                    )
                else:
                    self.n_components_ = self.n_components

            elif self.components_ is not None:
                if self.components_.shape[0] != self.n_components_:
                    raise ValueError(
                        f"n_components has changed from "
                        f"{self.components_.shape[0]} to "
                        f"{self.n_components_} between calls to "
                        f"partial_fit!"
                    )

            # --- build the matrix for SVD ---
            if self.n_samples_seen_ == 0:
                # First batch: plain SVD of raw (uncentered) data.
                X_stacked = X
            else:
                # Subsequent batch: stack previous top-k subspace (scaled
                # by singular values) with the new raw batch — the Ross
                # et al. (2008) incremental SVD algorithm, without
                # centering or mean-correction.
                prev = self.singular_values_.reshape((-1, 1)) * self.components_
                X_stacked = np.vstack([prev, X])

            U, S, Vt = linalg.svd(X_stacked, full_matrices=False, check_finite=False)
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

            self.n_samples_seen_ += n_samples
            k = self.n_components_
            self.components_ = Vt[:k]
            self.singular_values_ = S[:k]

            # --- sklearn-compatible explained_variance_ (S² / (N-1)) ---
            n_total = self.n_samples_seen_
            self.explained_variance_ = (S[:k] ** 2) / (n_total - 1)

            # --- HyperSpy convention explained_variance_ratio_ ---
            # ev / ev.sum()  (the lazy.py decomposition code overrides
            # these attributes anyway, so this is mainly for standalone
            # ISVD usage and test compatibility.)
            top_k_ev = S[:k] ** 2
            self.explained_variance_ratio_ = top_k_ev / top_k_ev.sum()

            if k not in (n_samples, n_features):
                self.noise_variance_ = (S[k:] ** 2).mean() / (n_total - 1)
            else:
                self.noise_variance_ = 0.0

            self.mean_ = np.zeros(n_features)

            return self

        @property
        def mean_(self):
            return self.__dict__.get("_isvd_mean", 0.0)

        @mean_.setter
        def mean_(self, value):
            # sklearn initialises mean_ to the scalar 0.0 on first call;
            # on subsequent calls it passes a float array.  We always store
            # zeros so that centering has no effect.
            if np.isscalar(value):
                self.__dict__["_isvd_mean"] = value
            else:
                self.__dict__["_isvd_mean"] = np.zeros_like(value)

else:

    class ISVD:  # type: ignore[no-redef]
        """Stub that raises ``ImportError`` on instantiation when scikit-learn is not installed."""

        def __init__(self, *args, **kwargs):
            _check_sklearn()
