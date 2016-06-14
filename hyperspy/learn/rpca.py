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
# but WITHOUT ANdata WARRANTdata; without even the implied warranty of
# MERCHANTABILITdata or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# dataou should have received a copy of the GNU General Public License
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

def orpca(X, rank, lambda1=None, lambda2=None,
          method=None, init=None, fast=False):
    """
    This function performs Online Robust PCA with
    with missing or corrupted data.

    Parameters
    ----------
    X : numpy array
        is the [m x n] matrix of observations.
    rank : int
        The model dimensionality.
    lambda1 : float
        Nuclear norm regularization parameter.
    lambda2 : float
        Sparse error regularization parameter.
    method : 'CF' | 'BCD'
        CF  - Closed-form (default)
        BCD - Block-coordinate descent
    init : 'rand' | 'BRP'
        rand - Random initialization (default)
        BRP  - Bilateral random projection

    Returns
    -------
    Y : numpy array
        is the [m x n] low-rank component.
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

    # Initialize by rescaling to [0,1]
    Xmin = X.min()
    Xmax = X.max()
    X = (X - Xmin) / (Xmax - Xmin)

    m, n = X.shape

    # Check options
    if method not in ('CF', 'BCD'):
        raise ValueError("'method' not recognised")
    if init not in ('rand', 'BRP'):
        raise ValueError("'init' not recognised")

    if lambda1 is None:
        _logger.warning("Nuclear norm regularization parameter "
                        "is set to default.")
        lambda1 = 1.0 / np.sqrt(n)
    if lambda2 is None:
        _logger.warning("Sparse regularization parameter "
                        "is set to default.")
        lambda2 = 1.0 / np.sqrt(n)

    if init == 'rand':
        # Use random initialization
        Y2 = np.random.randn(m, rank)
        L, tmp = scipy.linalg.qr(Y2, mode='economic')
    else:
        # Use bilateral random projections
        power = 1
        Z = X[:, 0:rank]
        Y2 = np.random.randn(rank, rank)
        for i in range(power):
            Y1 = np.dot(Z, Y2)
            Y2 = np.dot(Z.T, Y1)
        Q, tmp = scipy.linalg.qr(Y2, mode='economic')
        L = np.dot(np.dot(Z, Q), Q.T)

    R = np.zeros((rank, n))
    E = np.zeros((m, n))
    I = lambda1 * np.eye(rank)

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

    # Scale back
    Y = np.dot(L, R)
    Y = Y * (Xmax - Xmin) + Xmin
    E = E * (Xmax - Xmin) + Xmin

    # Do final SVD
    U, S, Vh = svd(Y)
    V = Vh.T

    return Y, E, U, S, V
