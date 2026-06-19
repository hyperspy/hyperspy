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
        scores.

        The centering is disabled by overriding the ``mean_`` property to
        always return zeros (either a scalar ``0.0`` or an array of zeros,
        whichever sklearn set last).  This neutralises both the
        mean-correction term computed during ``partial_fit`` and the
        mean-shift applied during ``transform``, without touching any
        other part of the sklearn implementation.

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
            Approximate explained variance per component, shape
            ``(n_components,)``.
        explained_variance_ratio_ : ndarray
            Fraction of total variance explained by each component, shape
            ``(n_components,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from hyperspy.learn.incremental_svd import ISVD
        >>> X = np.random.randn(200, 50)
        >>> obj = ISVD(n_components=3)
        >>> for chunk in np.array_split(X, 4):
        ...     obj.partial_fit(chunk)
        >>> components = obj.components_.T       # shape (n_features, n_components)
        >>> scores = obj.transform(X)            # shape (n_samples, n_components)
        """

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
