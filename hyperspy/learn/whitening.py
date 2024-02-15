# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

from hyperspy.learn.svd_pca import svd_solve


def whiten_data(X, centre=True, method="PCA", epsilon=1e-10):
    """Centre and whiten the data X.

    A whitening transformation is used to decorrelate
    the variables, such that the new covariance matrix
    of the whitened data is the identity matrix.

    If X is a random vector with non-singular covariance
    matrix C, and W is a whitening matrix satisfying
    W^T W = C^-1, then the transformation Y = W X will
    yield a whitened random vector Y with unit diagonal
    covariance. In ZCA whitening, the matrix W = C^-1/2,
    while in PCA whitening, the matrix W is the
    eigensystem of C. More details can be found in [Kessy2015]_.

    Parameters
    ----------
    X : numpy,ndarray
        The input data with shape (m, n).
    centre : bool, default True
        If True, centre the data along the features axis.
        If False, do not centre the data.
    method : {"PCA", "ZCA"}
        How to whiten the data. The default is PCA whitening.
    epsilon : float, default 1e-10
        Small floating-point value to avoid divide-by-zero errors.

    Returns
    -------
    Y : numpy.ndarray
        The centred and whitened data with shape (m, n).
    W : numpy.ndarray
        The whitening matrix with shape (n, n).

    References
    ----------
    .. [Kessy2015] A. Kessy, A. Lewin, and K. Strimmer, "Optimal
        Whitening and Decorrelation", arXiv:1512.00809, (2015),
        https://arxiv.org/pdf/1512.00809.pdf

    """
    Y = X

    # Centre the variables
    if centre:
        Y -= Y.mean(axis=0)

    # Calculate the whitening matrix
    R = (Y.T @ Y) / Y.shape[0]
    U, S, _ = svd_solve(R, svd_solver="full")
    S = np.sqrt(S + epsilon)[:, np.newaxis]

    if method == "PCA":
        # PCA whitening was the default in HyperSpy < 1.6.0,
        # we keep it as the default here.
        W = U.T / S

    elif method == "ZCA":
        W = U @ (U.T / S)

    else:
        raise ValueError(f"method must be one of ['PCA', 'zca'], got {method}")

    # Whiten the data
    Y = Y @ W.T

    return Y, W
