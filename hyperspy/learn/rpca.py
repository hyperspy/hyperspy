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
          init=None, training=None):
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
        Nuclear norm regularization parameter.
        If None, set to 1 / sqrt(nsamples)
    lambda2 : None | float
        Sparse error regularization parameter.
        If None, set to 1 / sqrt(nsamples)
    method : None | 'CF' | 'BCD'
        'CF'  - Closed-form
        'BCD' - Block-coordinate descent
        If None, set to 'CF'
    init : None | 'rand' | 'qr'
        'rand' - Random initialization
        'qr'   - QR-based initialization
        If None, set to 'rand'
    training : integer
        Specifies the number of training samples to use in
        the 'qr' initialization (ignored for 'rand')
        If None, set to 10

    Returns
    -------
    Xhat : numpy array
        is the [nfeatures x nsamples] low-rank matrix
    Ehat : numpy array
        is the [nfeatures x nsamples] sparse error matrix
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
                        "is set to default: 1 / sqrt(nsamples)")
        lambda1 = 1.0 / np.sqrt(n)
    if lambda2 is None:
        _logger.warning("Sparse regularization parameter "
                        "is set to default: 1 / sqrt(nsamples)")
        lambda2 = 1.0 / np.sqrt(n)
    if init is None:
        _logger.warning("No initialization specified. Defaulting to "
                        "random initialization")
        init = 'rand'
    if training is None:
        if init is 'rand':
            _logger.warning("Training samples only used for 'qr' method. "
                            "Parameter ignored")
        elif init is 'qr':
            if rank >= 10:
                _logger.warning("Number of training samples for 'qr' method "
                                "not specified. Defaulting to %d samples" % rank)
                training = rank
            else:
                _logger.warning("Number of training samples for 'qr' method "
                                "not specified. Defaulting to 10 samples")
                training = 10

    # Check options are valid
    if method not in ('CF', 'BCD'):
        raise ValueError("'method' not recognised")
    if init not in ('rand', 'qr'):
        raise ValueError("'method' not recognised")
    if init == 'qr' and training < rank:
        raise ValueError("'training' must be >= 'output_dimension'")

    # Get min & max of data matrix for scaling
    X_max = np.max(X)
    X_min = np.min(X)
    X = (X - X_min) / X_max

    # Initialize the subspace estimate
    if init == 'rand':
        Y2 = np.random.randn(m, rank)
        L, tmp = scipy.linalg.qr(Y2, mode='economic')
    elif init == 'qr':
        Y2 = X[:, :training]
        L, tmp = scipy.linalg.qr(Y2, mode='economic')
        L = L[:, :rank]

    R = np.zeros((rank, n))
    I = lambda1 * np.eye(rank)
    E = np.zeros((m, n))
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

    # Rescale
    Xhat = (np.dot(L, R) * X_max) + X_min
    Ehat = (E * X_max) + X_min

    # Do final SVD
    U, S, Vh = svd(Xhat)
    V = Vh.T

    # Chop small singular values which
    # likely arise from numerical noise
    # in the SVD.
    #S[S<=1e-9] = 0.0
    S[rank:] = 0.

    return Xhat, Ehat, U, S, V
