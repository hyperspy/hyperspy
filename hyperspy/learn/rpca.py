# -*- coding: utf-8 -*-
# This file is a transcription of a MATLAB code obtained from the
# following research paper:
#   Jiashi Feng, Huan Xu and Shuicheng Yuan, "Online Robust PCA via
#   Stochastic Optimization", Advances in Neural Information Processing
#   Systems 26, (2013), pp. 404-412.
#
# Copyright 2013 Jiashi Feng
# Copyright 2016 The HyperSpy developers
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

import numpy as np
import scipy.linalg

from hyperspy.misc.machine_learning.import_sklearn import (
    fast_svd, sklearn_installed)

_logger = logging.getLogger(__name__)

def _solveproj(z, X, I, lambda2):
    m, n = X.shape
    s = np.zeros(m)
    x = np.zeros(n)
    converged = False
    maxiter = 1e9
    iter = 0

    ddt = np.dot(scipy.linalg.inv(np.dot(X.T, X) + I), X.T)

    while converged is False:
        iter += 1
        xtmp = x
        x = np.dot(ddt, (z - s))
        stmp = s
        s = np.maximum(z - np.dot(X, x) - lambda2, 0.0)
        stopx = np.sqrt(np.dot(x - xtmp, (x - xtmp).conj()))
        stops = np.sqrt(np.dot(s - stmp, (s - stmp).conj()))
        stop = max(stopx, stops) / m
        if stop < 1e-5 or iter > maxiter:
            converged = True

    return x, s

def _updatecol(X, A, B, I):
    tmp, n = X.shape
    L = X
    A = A + I

    for i in range(n):
        b = B[:, i]
        x = X[:, i]
        a = A[:, i]
        temp = (b - np.dot(X, a)) / A[i, i] + x
        L[:, i] = temp / max(np.sqrt(np.dot(temp, temp.conj())), 1)

    return L

def orpca(X, rank, fast=False, lambda1=None,
          lambda2=None, method=None,
          mask=None):
    """
    This function performs Online Robust PCA with
    with missing or corrupted data.

    Parameters
    ----------
    X : numpy array
        is the [nfeatures x nsamples] matrix of observations.
    rank : int
        The model dimensionality.
    lambda1 : None | float
        Nuclear norm regularization parameter. If None, set to 1/sqrt(nfeatures)
    lambda2 : None | float
        Sparse error regularization parameter. If None, set to 1/sqrt(nfeatures)
    method : None | 'CF' | 'BCD'
        If None, set to 'CF'
        'CF'  - Closed-form
        'BCD' - Block-coordinate descent
    mask : numpy array
        is an initial estimate of the sparse error matrix

    Returns
    -------
    L : numpy array
        is the [nfeatures x rank] basis array
    R : numpy array
        is the [rank x nsamples] coefficient array
    E : numpy array
        is the sparse error
    U, S, V : numpy arrays
        are the results of an SVD on Y

    """
    if fast is True and sklearn_installed is True:
        def svd(X):
            return fast_svd(X, rank)
    else:
        def svd(X):
            return scipy.linalg.svd(X, full_matrices=False)

    # Get shape
    m, n = X.shape

    # Check options if None
    if method is None:
        _logger.warning("No method specified. Defaulting to "
                        "'CF' (closed-form solver)")
        method = 'CF'
    if lambda1 is None:
        _logger.warning("Nuclear norm regularization parameter "
                        "is set to default: 1/sqrt(nfeatures)")
        lambda1 = 1.0 / np.sqrt(n)
    if lambda2 is None:
        _logger.warning("Sparse regularization parameter "
                        "is set to default: 1/sqrt(nfeatures)")
        lambda2 = 1.0 / np.sqrt(n)

    # Check options are valid
    if method not in ('CF', 'BCD'):
        raise ValueError("'method' not recognised")

    # Use random initialization
    Y2 = np.random.randn(m, rank)
    L, tmp = scipy.linalg.qr(Y2, mode='economic')

    R = np.zeros((rank, n))
    I = lambda1 * np.eye(rank)

    # Allow the error matrix to be initialized
    if mask is None:
        E = np.zeros((m, n))
    else:
        E = mask.reshape((m, n))

    A = np.zeros((rank, rank))
    B = np.zeros((m, rank))

    for t in range(n):
        if t == 0 or np.mod(t + 1, np.round(n / 10)) == 0:
            _logger.info("Processing sample : %s" % (t + 1))

        z = X[:, t]
        r, e = _solveproj(z, L, I, lambda2)

        R[:, t] = r
        E[:, t] = e

        if method == 'CF':
            # Closed-form solution
            A = A + np.outer(r, r.T)
            B = B + np.outer((z - e), r.T)
            L = np.dot(B, scipy.linalg.inv(A + I))
        else:
            # Block-coordinate descent
            A = A + np.outer(r, r.T)
            B = B + np.outer((z - e), r.T)
            L = _updatecol(L, A, B, I)

    # Do final SVD
    U, S, Vh = svd(np.dot(L, R))
    V = Vh.T

    # Chop small singular values which
    # likely arise from numerical noise
    # in the SVD.
    #S[S<=1e-9] = 0.0
    S[rank:] = 0.

    return L, R, E, U, S, V
