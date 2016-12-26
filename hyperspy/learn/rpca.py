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
from itertools import chain

import numpy as np
import scipy.linalg

from hyperspy.misc.machine_learning.import_sklearn import (
    fast_svd, sklearn_installed)
from hyperspy.external.progressbar import progressbar

_logger = logging.getLogger(__name__)


def _thresh(X, lambda1):
    res = np.abs(X) - lambda1
    return np.sign(X) * ((res > 0) * res)


def _normalize(arr, ar_min=None, ar_max=None, undo=False):
    if not undo:
        if ar_min is None:
            ar_min = arr.min()
        if ar_max is None:
            ar_max = arr.max()
    else:
        if ar_min is None or ar_max is None:
            raise ValueError("min / max values have to be passed when undoing "
                             "the normalization")
    if undo:
        return (arr * (ar_max - ar_min)) + ar_min
    else:
        return (arr - ar_min) / (ar_max - ar_min)


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


class ORPCA:

    def __init__(self, rank, fast=False, lambda1=None, lambda2=None,
                 method=None, learning_rate=None, init=None,
                 training_samples=None, momentum=None):

        self.nfeatures = None
        self.normalize = False
        if fast is True and sklearn_installed is True:
            def svd(X):
                return fast_svd(X, rank)
        else:
            def svd(X):
                return scipy.linalg.svd(X, full_matrices=False)

        self.svd = svd
        self.t = 0

        # Check options if None
        if method is None:
            _logger.warning("No method specified. Defaulting to "
                            "'CF' (closed-form solver)")
            method = 'CF'
        if init is None:
            _logger.warning("No initialization specified. Defaulting to "
                            "'qr' initialization")
            init = 'qr'
        if training_samples is None:
            if init == 'qr':
                if rank >= 10:
                    training_samples = rank
                else:
                    training_samples = 10
                _logger.warning("Number of training samples for 'qr' method "
                                "not specified. Defaulting to %d samples" %
                                training_samples)
        if learning_rate is None:
            if method in ('SGD', 'MomentumSGD'):
                _logger.warning("Learning rate for SGD algorithm is "
                                "set to default: 1.0")
                learning_rate = 1.0
        if momentum is None:
            if method == 'MomentumSGD':
                _logger.warning("Momentum parameter for SGD algorithm is "
                                "set to default: 0.5")
                momentum = 0.5


        self.rank = rank
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.method = method
        self.init = init
        self.training_samples = training_samples
        self.learning_rate = learning_rate
        self.momentum=momentum

        # Check options are valid
        if method not in ('CF', 'BCD', 'SGD', 'MomentumSGD'):
            raise ValueError("'method' not recognised")
        if not isinstance(init, np.ndarray) and init not in ('qr', 'rand'):
            raise ValueError("'method' not recognised")
        if init == 'qr' and training_samples < rank:
            raise ValueError(
                "'training_samples' must be >= 'output_dimension'")
        if method == 'MomentumSGD' and (momentum > 1. or momentum < 0.):
            raise ValueError("'momentum' must be a float between 0 and 1")

    def _setup(self, X, normalize=False):

        if isinstance(X, np.ndarray):
            n, m = X.shape
            iterating = False
            if normalize:
                self.X_min = X.min()
                self.X_max = X.max()
                self.normalize = normalize
                # actually scale the data to be between 0 and 1, not just close
                # to it..
                X = _normalize(X, ar_min=self.X_min, ar_max=self.X_max)
                # X = (X - self.X_min) / (self.X_max - self.X_min)
        else:
            if normalize:
                _logger.warning("Normalization with an iterator is not"
                                " possible, option ignored.")
            x = next(X)
            m = len(x)
            X = chain([x], X)
            iterating = True

        self.nfeatures = m
        self.iterating = iterating

        if self.lambda1 is None:
            _logger.warning("Nuclear norm regularization parameter "
                            "is set to default: 1 / sqrt(nfeatures)")
            self.lambda1 = 1.0 / np.sqrt(m)
        if self.lambda2 is None:
            _logger.warning("Sparse regularization parameter "
                            "is set to default: 1 / sqrt(nfeatures)")
            self.lambda2 = 1.0 / np.sqrt(m)

        self.L = self._initialize(X)
        self.I = self.lambda1 * np.eye(self.rank)
        self.R = []
        self.E = []

        # Extra variables for CF and BCD methods
        if self.method in ('CF', 'BCD'):
            self.A = np.zeros((self.rank, self.rank))
            self.B = np.zeros((m, self.rank))
        if self.method == 'MomentumSGD':
            self.vnew = np.zeros_like(self.L)
        return X

    def _initialize(self, X):
        m = self.nfeatures
        iterating = self.iterating

        # Initialize the subspace estimate
        if self.init in ('qr', 'rand'):
            if self.init == 'qr':
                if iterating:
                    Y2 = np.stack([next(X) for _ in range(self.training_samples)],
                                  axis=-1)
                    X = chain(iter(Y2.T.copy()), X)
                else:
                    Y2 = X[:self.training_samples, :].T
                # normalize the init data here..
                # Y2 = (Y2 - Y2.min()) / (Y2.max() - Y2.min())
                Y2 = _normalize(Y2)
            elif self.init == 'rand':
                Y2 = np.random.randn(m, self.rank)
            L, _ = scipy.linalg.qr(Y2, mode='economic')
            return L[:, :self.rank]
        elif isinstance(self.init, np.ndarray):
            if init.ndim != 2:
                raise ValueError("'init' has to be a two-dimensional matrix")
            init_m, init_r = init.shape
            if init_m != m or init_r != self.rank:
                raise ValueError(
                    "'init' has to be of shape [nfeatures x rank]")
            return init.copy()
        else:
            raise ValueError('Bad initialization options')

    def fit(self, X, iterating=None):
        if self.nfeatures is None:
            X = self._setup(X)

        if iterating is None:
            iterating = self.iterating
        else:
            self.iterating = iterating
        num = None
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)

        for z in progressbar(X, leave=False, total=num):
            if not self.t or not (self.t + 1) % 10:
                _logger.info("Processing sample : %s" % (self.t + 1))

            # TODO: what about z.min()?
            thislambda2 = self.lambda2  # * z.max()
            thislambda1 = self.lambda1  # * z.max()

            r, e = _solveproj(z, self.L, self.I, thislambda2)

            self.R.append(r)
            if not iterating:
                self.E.append(e)

            if self.method == 'CF':
                # Closed-form solution
                self.A += np.outer(r, r.T)
                self.B += np.outer((z - e), r.T)
                self.L = np.dot(self.B, scipy.linalg.inv(self.A + self.I))
            elif self.method == 'BCD':
                # Block-coordinate descent
                self.A += np.outer(r, r.T)
                self.B += np.outer((z - e), r.T)
                self.L = _updatecol(self.L, self.A, self.B, self.I)
            elif self.method == 'SGD':
                # Stochastic gradient descent
                learn = self.learning_rate * (1 + self.learning_rate *
                                              thislambda1 * self.t)
                self.L -= (np.dot(self.L, np.outer(r, r.T))
                           - np.outer((z - e), r.T)
                           + thislambda1 * self.L) / learn
            elif self.method == 'MomentumSGD':
                # Stochastic gradient descent with momentum
                learn = self.learning_rate * (1 + self.learning_rate *
                                              thislambda1 * self.t)
                vold = self.momentum * self.vnew
                self.vnew = (np.dot(self.L, np.outer(r, r.T))
                         - np.outer((z - e), r.T)
                         + thislambda1 * self.L) / learn
                self.L -= (vold + self.vnew)
            self.t += 1

    def project(self, X):
        num = None
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)
        for v in progressbar(X, leave=False, total=num):
            r, _ = _solveproj(v, self.L, self.I, self.lambda2)
            self.R.append(r.copy())


    def finish(self):

        R = np.stack(self.R, axis=-1)

        Xhat = np.dot(self.L, R)
        if len(self.E):
            Ehat = np.stack(self.E, axis=-1)
            # both keep an indicator that we had something and remove the
            # duplicate data
            self.E = [1]
            if self.normalize:
                Ehat = _normalize(Ehat, ar_min=self.X_min, ar_max=self.X_max,
                                  undo=True)

        if self.normalize:
            Xhat = _normalize(Xhat, ar_min=self.X_min, ar_max=self.X_max,
                              undo=True)

        # Do final SVD
        U, S, Vh = self.svd(Xhat)
        V = Vh.T

        # Chop small singular values which
        # likely arise from numerical noise
        # in the SVD.
        S[self.rank:] = 0.
        if len(self.E):
            return Xhat.T, Ehat, U, S, V
        else:
            return Xhat.T, 1, U, S, V


def orpca(X, rank, fast=False,
          lambda1=None,
          lambda2=None,
          method=None,
          learning_rate=None,
          init=None,
          training_samples=None,
          momentum=None):
    """
    This function performs Online Robust PCA
    with missing or corrupted data.

    Parameters
    ----------
    X : {numpy array, iterator}
        [nfeatures x nsamples] matrix of observations
        or an iterator that yields samples, each with nfeatures elements.
    rank : int
        The model dimensionality.
    lambda1 : {None, float}
        Nuclear norm regularization parameter.
        If None, set to 1 / sqrt(nsamples)
    lambda2 : {None, float}
        Sparse error regularization parameter.
        If None, set to 1 / sqrt(nsamples)
    method : {None, 'CF', 'BCD', 'SGD', 'MomentumSGD'}
        'CF'  - Closed-form solver
        'BCD' - Block-coordinate descent
        'SGD' - Stochastic gradient descent
        'MomentumSGD' - Stochastic gradient descent with momentum
        If None, set to 'CF'
    learning_rate : {None, float}
        Learning rate for the stochastic gradient
        descent algorithm
        If None, set to 1
    init : {None, 'qr', 'rand', np.ndarray}
        'qr'   - QR-based initialization
        'rand' - Random initialization
        np.ndarray if the shape [nfeatures x rank].
        If None, set to 'qr'
    training_samples : {None, integer}
        Specifies the number of training samples to use in
        the 'qr' initialization
        If None, set to 10
    momentum : {None, float}
        Momentum parameter for 'MomentumSGD' method, should be
        a float between 0 and 1.
        If None, set to 0.5

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
    A stochastic gradient descent (SGD) solver is also implemented,
    along with a MomentumSGD solver for improved convergence and robustness
    with respect to local minima. More information about the gradient descent
    methods and choosing appropriate parameters can be found here:
       Sebastian Ruder, "An overview of gradient descent optimization
       algorithms", arXiv:1609.04747, (2016), http://arxiv.org/abs/1609.04747.

    """
    _orpca = ORPCA(rank, fast=fast, lambda1=lambda1,
                   lambda2=lambda2, method=method,
                   learning_rate=learning_rate, init=init,
                   training_samples=training_samples,
                   momentum=momentum)
    _orpca._setup(X, normalize=True)
    _orpca.fit(X)
    return _orpca.finish()
