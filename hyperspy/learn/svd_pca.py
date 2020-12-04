# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

import logging
import warnings
from distutils.version import LooseVersion

import numpy as np
import scipy
from scipy.linalg import svd
from scipy.sparse.linalg import svds

from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.misc.machine_learning.import_sklearn import (
    randomized_svd,
    sklearn_installed,
)

_logger = logging.getLogger(__name__)


def svd_flip_signs(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u, v : numpy array
        u and v are the outputs of a singular value decomposition.
    u_based_decision : bool, default True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u, v : numpy array
        Adjusted outputs with same dimensions as inputs.

    """
    # Derived from `sklearn.utils.extmath.svd_flip`.
    # Copyright (c) 2007-2020 The scikit-learn developers.
    # All rights reserved.

    if u_based_decision:
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    else:
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])

    u *= signs
    v *= signs[:, np.newaxis]

    return u, v


def svd_solve(
    data,
    output_dimension=None,
    svd_solver="auto",
    svd_flip=True,
    u_based_decision=True,
    **kwargs,
):
    """Apply singular value decomposition to input data.

    Parameters
    ----------
    data : numpy array, shape (m, n)
        Input data array
    output_dimension : None or int
        Number of components to keep/calculate
    svd_solver : {"auto", "full", "arpack", "randomized"}, default "auto"
        If auto:
            The solver is selected by a default policy based on `data.shape` and
            `output_dimension`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient "randomized"
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full:
            run exact SVD, calling the standard LAPACK solver via
            :py:func:`scipy.linalg.svd`, and select the components by postprocessing
        If arpack:
            use truncated SVD, calling ARPACK solver via
            :py:func:`scipy.sparse.linalg.svds`. It requires strictly
            `0 < output_dimension < min(data.shape)`
        If randomized:
            use truncated SVD, calling :py:func:`sklearn.utils.extmath.randomized_svd`
            to estimate a limited number of components
    svd_flip : bool, default True
        If True, adjusts the signs of the loadings and factors such that
        the loadings that are largest in absolute value are always positive.
        See :py:func:`~.learn.svd_pca.svd_flip` for more details.
    u_based_decision : bool, default True
        If True, and svd_flip is True, use the columns of u as the basis for sign-flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    U, S, V : numpy array
        Output of SVD such that X = U*S*V.T

    """
    # Derived from `sklearn.decomposition.PCA`.
    # Copyright (c) 2007-2020 The scikit-learn developers.
    # All rights reserved.

    m, n = data.shape

    if output_dimension is None:
        output_dimension = min(m, n)
        if svd_solver == "arpack":
            output_dimension -= 1

    if svd_solver == "auto":
        if max(m, n) <= 500:
            svd_solver = "full"
        elif (
            output_dimension >= 1
            and output_dimension < 0.8 * min(m, n)
            and sklearn_installed
        ):
            svd_solver = "randomized"
        else:
            svd_solver = "full"

    if svd_solver == "randomized":
        if not sklearn_installed:  # pragma: no cover
            raise ImportError(
                "svd_solver='randomized' requires scikit-learn to be installed"
            )
        U, S, V = randomized_svd(data, n_components=output_dimension, **kwargs)
    elif svd_solver == "arpack":
        if LooseVersion(scipy.__version__) < LooseVersion("1.4.0"):  # pragma: no cover
            raise ValueError('`svd_solver="arpack"` requires scipy >= 1.4.0')

        if output_dimension >= min(m, n):
            raise ValueError(
                "svd_solver='arpack' requires output_dimension "
                "to be strictly less than min(data.shape)."
            )
        U, S, V = svds(data, k=output_dimension, **kwargs)
        # svds doesn't follow scipy.linalg.svd conventions,
        # so reverse its outputs
        S = S[::-1]
        # flip eigenvectors' sign to enforce deterministic output
        if svd_flip:
            U, V = svd_flip_signs(
                U[:, ::-1], V[::-1], u_based_decision=u_based_decision
            )
    elif svd_solver == "full":
        U, S, V = svd(data, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        if svd_flip:
            U, V = svd_flip_signs(U, V, u_based_decision=u_based_decision)

        U = U[:, :output_dimension]
        S = S[:output_dimension]
        V = V[:output_dimension, :]

    return U, S, V


def svd_pca(
    data,
    output_dimension=None,
    svd_solver="auto",
    centre=None,
    auto_transpose=True,
    svd_flip=True,
    **kwargs,
):
    """Perform PCA using singular value decomposition (SVD).

    Read more in the :ref:`User Guide <mva.pca>`.

    Parameters
    ----------
    data : numpy array
        MxN array of input data (M features, N samples)
    output_dimension : None or int
        Number of components to keep/calculate
    svd_solver : {"auto", "full", "arpack", "randomized"}, default "auto"
        If auto:
            The solver is selected by a default policy based on `data.shape` and
            `output_dimension`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient "randomized"
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full:
            run exact SVD, calling the standard LAPACK solver via
            :py:func:`scipy.linalg.svd`, and select the components by postprocessing
        If arpack:
            use truncated SVD, calling ARPACK solver via
            :py:func:`scipy.sparse.linalg.svds`. It requires strictly
            `0 < output_dimension < min(data.shape)`
        If randomized:
            use truncated SVD, calling :py:func:`sklearn.utils.extmath.randomized_svd`
            to estimate a limited number of components
    centre : {None, "navigation", "signal"}, default None
        * If None, the data is not centered prior to decomposition.
        * If "navigation", the data is centered along the navigation axis.
        * If "signal", the data is centered along the signal axis.
    auto_transpose : bool, default True
        If True, automatically transposes the data to boost performance.
    svd_flip : bool, default True
        If True, adjusts the signs of the loadings and factors such that
        the loadings that are largest in absolute value are always positive.
        See :py:func:`~.learn.svd_pca.svd_flip` for more details.

    Returns
    -------
    factors : numpy array
    loadings : numpy array
    explained_variance : numpy array
    mean : numpy array or None (if centre is None)

    """
    N, M = data.shape

    if centre is None:
        mean = None

    else:
        # To avoid confusion between terminology in different
        # machine learning fields, we map the argument here.
        # See #1159 for some discussion.
        if centre in ["variables", "trials"]:
            centre_map = {
                "trials": "navigation",
                "variables": "signal",
            }
            centre_new = centre_map.get(centre, None)

            warnings.warn(
                f"centre='{centre}' has been deprecated and will be "
                f"removed in HyperSpy 2.0. Please use '{centre_new}' instead.",
                VisibleDeprecationWarning,
            )

            centre = centre_new

        if centre == "signal":
            mean = data.mean(axis=1)[:, np.newaxis]
        elif centre == "navigation":
            mean = data.mean(axis=0)[np.newaxis, :]
        else:
            raise ValueError("'centre' must be one of [None, 'navigation', 'signal']")

        data -= mean

    if auto_transpose is True:
        if N < M:
            _logger.info("Auto-transposing the data")
            data = data.T
        else:
            auto_transpose = False

    U, S, V = svd_solve(
        data,
        output_dimension=output_dimension,
        svd_solver=svd_solver,
        svd_flip=svd_flip,
        **kwargs,
    )

    explained_variance = S ** 2 / N

    if auto_transpose is False:
        factors = V.T
        loadings = U * S
    else:
        loadings = V.T
        factors = U * S

    return factors, loadings, explained_variance, mean
