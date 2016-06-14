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
        iter = iter + 1
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

def orpca(X, rank, lambda1, lambda2, method='BCD'):
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
    method : 'BCD' or 'CF'
        BCD - Block-coordinate descent (default)
        CF  - Closed-form

    Returns
    -------
    L : numpy array
        is the [m x r] basis.
    R : numpy array
        is the [r x n] coefficients.
    S : numpy array
        is the sparse error
    """

    methods = {'BCD':'Block coordinate descent',
               'CF':'Closed-form'}
    if method not in methods:
        raise ValueError("'method' must be one of " + methods.keys())

    _logger.info("Performing Online Robust PCA")

    # Initialize
    m, n = X.shape

    # Use random initialization
    Y2 = np.random.rand(m, rank)
    L, tmp = scipy.linalg.qr(Y2, mode='economic')

    R = np.zeros((rank, n))
    S = np.zeros((m, n))
    I = lambda1 * np.eye(rank)

    A = np.zeros((rank, rank))
    B = np.zeros((m, rank))

    for t in range(n):
        if t == 0 or np.mod(t + 1, np.round(n / 10)) == 0:
            _logger.info("Iteration: %s" % t + 1)

        z = X[:, t]
        r, s = _solveproj(z, L, I, lambda2)

        R[:, t] = r
        S[:, t] = s

        if method == 'BCD':
            # Block-coordinate descent
            A = A + np.outer(r, r.T)
            B = B + np.outer((z - s), r.T)
            L = _updatecol(L, A, B, I)
        else:
            # Closed-form
            A = A + np.outer(r, r.T)
            B = B + np.outer((z - s), r.T)
            L = np.dot(B, scipy.linalg.inv(A + I))

    return L, R, S
