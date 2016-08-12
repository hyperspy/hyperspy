# -*- coding: utf-8 -*-
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


def _thresh(X, lambda1):
    res = np.abs(X) - lambda1
    return np.sign(X) * ((res > 0) * res)


def rpca_godec(X, rank, fast=False, lambda1=None,
               power=None, tol=None, maxiter=None):
    """
    This function performs Robust PCA with missing or corrupted data,
    using the GoDec algorithm.

    Parameters
    ----------
    X : numpy array
        is the [nfeatures x nsamples] matrix of observations.
    rank : int
        The model dimensionality.
    lambda1 : None | float
        Regularization parameter.
        If None, set to 1 / sqrt(nsamples)
    power : None | integer
        The number of power iterations used in the initialization
        If None, set to 0 for speed
    tol : None | float
        Convergence tolerance
        If None, set to 1e-3
    maxiter : None | integer
        Maximum number of iterations
        If None, set to 1e3

    Returns
    -------
    Xhat : numpy array
        is the [nfeatures x nsamples] low-rank matrix
    Ehat : numpy array
        is the [nfeatures x nsamples] sparse error matrix
    Ghat : numpy array
        is the [nfeatures x nsamples] Gaussian noise matrix
    U, S, V : numpy arrays
        are the results of an SVD on Xhat

    Notes
    -----
    Algorithm based on the following research paper:
       Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Low-rank & Sparse Matrix
       Decomposition in Noisy Case", ICML-11, (2011), pp. 33-40.

    Code: https://sites.google.com/site/godecomposition/matrix/artifact-1

    """
    if fast is True and sklearn_installed is True:
        def svd(X):
            return fast_svd(X, rank)
    else:
        def svd(X):
            return scipy.linalg.svd(X, full_matrices=False)

    # Get shape
    m, n = X.shape

    # Operate on transposed matrix for speed
    transpose = False
    if m < n:
        transpose = True
        X = X.T

    # Get shape
    m, n = X.shape

    # Check options if None
    if lambda1 is None:
        _logger.warning("Threshold 'lambda1' is set to "
                        "default: 1 / sqrt(nsamples)")
        lambda1 = 1.0 / np.sqrt(n)
    if power is None:
        _logger.warning("Number of power iterations not specified. "
                        "Defaulting to 0")
        power = 0
    if tol is None:
        _logger.warning("Convergence tolerance not specifed. "
                        "Defaulting to 1e-3")
        tol = 1e-3
    if maxiter is None:
        _logger.warning("Maximum iterations not specified. "
                        "Defaulting to 1e3")
        maxiter = 1e3

    # Get min & max of data matrix for scaling
    X_max = np.max(X)
    X_min = np.min(X)
    X = (X - X_min) / X_max

    # Initialize L and E
    L = X
    E = np.zeros(L.shape)

    itr = 0
    while True:
        itr += 1

        # Initialization with bilateral random projections
        Y2 = np.random.randn(n, rank)
        for i in range(power + 1):
            Y2 = np.dot(L.T, np.dot(L, Y2))
        Q, tmp = scipy.linalg.qr(Y2, mode='economic')

        # Estimate the new low-rank and sparse matrices
        Lnew = np.dot(np.dot(L, Q), Q.T)
        A = L - Lnew + E
        L = Lnew
        E = _thresh(A, lambda1)
        A -= E
        L += A

        # Check convergence
        eps = np.linalg.norm(A)
        if (eps < tol):
            _logger.info("Converged to %f in %d iterations" % (eps, itr))
            break
        elif (itr >= maxiter):
            _logger.warning("Maximum iterations reached")
            break

    # Get the remaining Gaussian noise matrix
    G = X - L - E

    # Transpose back
    if transpose:
        L = L.T
        E = E.T
        G = G.T

    # Rescale
    Xhat = (L * X_max) + X_min
    Ehat = (E * X_max) + X_min
    Ghat = (G * X_max) + X_min

    # Do final SVD
    U, S, Vh = svd(Xhat)
    V = Vh.T

    # Chop small singular values which
    # likely arise from numerical noise
    # in the SVD.
    S[rank:] = 0.

    return Xhat, Ehat, Ghat, U, S, V


def _solveproj(z, X, I, lambda2):
    m, n = X.shape
    s = np.zeros(m)
    x = np.zeros(n)
    maxiter = 1e9
    itr = 0

    ddt = np.dot(scipy.linalg.inv(np.dot(X.T, X) + I), X.T)

    while True:
        itr += 1
        xtmp = x
        x = np.dot(ddt, (z - s))
        stmp = s
        s = np.maximum(z - np.dot(X, x) - lambda2, 0.0)
        stopx = np.sqrt(np.dot(x - xtmp, (x - xtmp).conj()))
        stops = np.sqrt(np.dot(s - stmp, (s - stmp).conj()))
        stop = max(stopx, stops) / m
        if stop < 1e-6 or itr > maxiter:
            break

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


def orpca(X, rank, fast=False,
          lambda1=None,
          lambda2=None,
          method=None,
          learning_rate=None,
          init=None,
          training_samples=None):
    """
    This function performs Online Robust PCA
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
    method : None | 'CF' | 'BCD' | 'SGD'
        'CF'  - Closed-form solver
        'BCD' - Block-coordinate descent
        'SGD' - Stochastic gradient descent
        If None, set to 'CF'
    learning_rate : None | float
        Learning rate for the stochastic gradient
        descent algorithm
        If None, set to 1
    init : None | 'qr' | 'rand'
        'qr'   - QR-based initialization
        'rand' - Random initialization
        If None, set to 'qr'
    training_samples : integer
        Specifies the number of training samples to use in
        the 'qr' initialization
        If None, set to 10

    Returns
    -------
    Xhat : numpy array
        is the [nfeatures x nsamples] low-rank matrix
    Ehat : numpy array
        is the [nfeatures x nsamples] sparse error matrix
    U, S, V : numpy arrays
        are the results of an SVD on Xhat

    Notes
    -----
    The ORPCA code is based on a transcription of MATLAB code obtained from
    the following research paper:
       Jiashi Feng, Huan Xu and Shuicheng Yuan, "Online Robust PCA via
       Stochastic Optimization", Advances in Neural Information Processing
       Systems 26, (2013), pp. 404-412.

    It has been updated to include a new initialization method based
    on a QR decomposition of the first n "training" samples of the data.
    A stochastic gradient descent solver is also implemented.

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
                        "'qr' initialization")
        init = 'qr'
    if training_samples is None:
        if init == 'qr':
            if rank >= 10:
                _logger.warning("Number of training samples for 'qr' method "
                                "not specified. Defaulting to %d samples" % rank)
                training_samples = rank
            else:
                _logger.warning("Number of training samples for 'qr' method "
                                "not specified. Defaulting to 10 samples")
                training_samples = 10
    if learning_rate is None:
        if method == 'SGD':
            _logger.warning("Learning rate for SGD algorithm is "
                            "set to default: 1.0")
            learning_rate = 1.0

    # Check options are valid
    if method not in ('CF', 'BCD', 'SGD'):
        raise ValueError("'method' not recognised")
    if init not in ('qr', 'rand'):
        raise ValueError("'method' not recognised")
    if init == 'qr' and training_samples < rank:
        raise ValueError("'training_samples' must be >= 'output_dimension'")

    # Get min & max of data matrix for scaling
    X_max = np.max(X)
    X_min = np.min(X)
    X = (X - X_min) / X_max

    # Initialize the subspace estimate
    if init == 'qr':
        Y2 = X[:, :training_samples]
        L, tmp = scipy.linalg.qr(Y2, mode='economic')
        L = L[:, :rank]
    elif init == 'rand':
        Y2 = np.random.randn(m, rank)
        L, tmp = scipy.linalg.qr(Y2, mode='economic')

    R = np.zeros((rank, n))
    I = lambda1 * np.eye(rank)
    E = np.zeros((m, n))

    # Extra variables for CF and BCD methods
    if method in ('CF', 'BCD'):
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
        elif method == 'BCD':
            # Block-coordinate descent
            A = A + np.outer(r, r.T)
            B = B + np.outer((z - e), r.T)
            L = _updatecol(L, A, B, I)
        elif method == 'SGD':
            # Stochastic gradient descent
            learn = learning_rate * (1 + learning_rate * lambda1 * t)
            L = L - (np.dot(L, np.outer(r, r.T))
                     - np.outer((z - e), r.T)
                     + lambda1 * L) / learn

    # Rescale
    Xhat = (np.dot(L, R) * X_max) + X_min
    Ehat = (E * X_max) + X_min

    # Do final SVD
    U, S, Vh = svd(Xhat)
    V = Vh.T

    # Chop small singular values which
    # likely arise from numerical noise
    # in the SVD.
    S[rank:] = 0.

    return Xhat, Ehat, U, S, V
