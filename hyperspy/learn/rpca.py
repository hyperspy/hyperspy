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
from itertools import chain

import numpy as np
import scipy.linalg

from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.external.progressbar import progressbar
from hyperspy.learn.svd_pca import svd_solve

_logger = logging.getLogger(__name__)


def _soft_thresh(X, lambda1):
    """Soft-thresholding of array X."""
    res = np.abs(X) - lambda1
    np.maximum(res, 0.0, out=res)
    res *= np.sign(X)
    return res


def rpca_godec(X, rank, lambda1=None, power=None, tol=None, maxiter=None, **kwargs):
    """Perform Robust PCA with missing or corrupted data, using the GoDec algorithm.

    Parameters
    ----------
    X : numpy array
        is the [n_features x n_samples] matrix of observations.
    rank : int
        The model dimensionality.
    lambda1 : None | float
        Regularization parameter.
        If None, set to 1 / sqrt(n_samples)
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
        is the [n_features x n_samples] low-rank matrix
    Ehat : numpy array
        is the [n_features x n_samples] sparse error matrix
    Ghat : numpy array
        is the [n_features x n_samples] Gaussian noise matrix
    U, S, V : numpy arrays
        are the results of an SVD on Xhat

    Notes
    -----
    GoDec algorithm based on the Matlab code from [Zhou2011]_. See code here:
    https://sites.google.com/site/godecomposition/matrix/artifact-1

    References
    ----------
    .. [Zhou2011] Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Low-rank & Sparse
           Matrix Decomposition in Noisy Case", ICML-11, (2011), pp. 33-40.

    """
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
        _logger.info("Threshold 'lambda1' is set to default: 1 / sqrt(n_features)")
        lambda1 = 1.0 / np.sqrt(m)
    if power is None:
        _logger.info("Number of power iterations not specified. Defaulting to 0")
        power = 0
    if tol is None:
        _logger.info("Convergence tolerance not specifed. Defaulting to 1e-3")
        tol = 1e-3
    if maxiter is None:
        _logger.info("Maximum iterations not specified. Defaulting to 1e3")
        maxiter = 1e3

    # Initialize L and E
    L = X
    E = np.zeros(L.shape)

    for itr in range(int(maxiter)):
        # Initialization with bilateral random projections
        Y2 = np.random.randn(n, rank)
        for _ in range(power + 1):
            Y2 = L.T @ (L @ Y2)

        Q, _ = scipy.linalg.qr(Y2, mode="economic")

        # Estimate the new low-rank and sparse matrices
        Lnew = (L @ Q) @ Q.T
        A = L - Lnew + E
        L = Lnew
        E = _soft_thresh(A, lambda1)
        A -= E
        L += A

        # Check convergence
        eps = np.linalg.norm(A)
        if eps < tol:
            _logger.info("Converged to {} in {} iterations".format(eps, itr))
            break

    # Get the remaining Gaussian noise matrix
    G = X - L - E

    # Transpose back
    if transpose:
        L = L.T
        E = E.T
        G = G.T

    # Rescale
    Xhat = L
    Ehat = E
    Ghat = G

    # Do final SVD
    U, S, Vh = svd_solve(Xhat, output_dimension=rank, **kwargs)
    V = Vh.T

    # Chop small singular values which
    # likely arise from numerical noise
    # in the SVD.
    S[rank:] = 0.0

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
        s = _soft_thresh(z - np.dot(X, x), lambda2)
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
    """This class performs Online Robust PCA with missing or corrupted data.

    Methods
    -------
    fit
        Learn factors from the given data.
    project
        Project the learnt factors on the given data.
    finish
        Return the learnt factors and loadings.

    """

    def __init__(
        self,
        rank,
        lambda1=None,
        lambda2=None,
        method=None,
        init=None,
        training_samples=None,
        subspace_learning_rate=None,
        subspace_momentum=None,
        learning_rate=None,
        momentum=None,
    ):
        """Creates Online Robust PCA instance that can learn a representation.

        Parameters
        ----------
        rank : int
            The rank of the representation (number of components/factors)
        lambda1 : {None, float}
            Nuclear norm regularization parameter.
            If None, set to 1 / sqrt(n_samples)
        lambda2 : {None, float}
            Sparse error regularization parameter.
            If None, set to 1 / sqrt(n_samples)
        method : {None, 'CF', 'BCD', 'SGD', 'MomentumSGD'}
            'CF'  - Closed-form solver
            'BCD' - Block-coordinate descent
            'SGD' - Stochastic gradient descent
            'MomentumSGD' - Stochastic gradient descent with momentum
            If None, set to 'CF'
        init : {None, 'qr', 'rand', np.ndarray}
            * 'qr'   - QR-based initialization
            * 'rand' - Random initialization
            * np.ndarray if the shape [n_features x rank].
            * If None (default), set to 'qr'
        training_samples : {None, integer}
            Specifies the number of training samples to use in
            the 'qr' initialization.
            If None, set to 10
        subspace_learning_rate : {None, float}
            Learning rate for the stochastic gradient
            descent algorithm
            If None, set to 1
        subspace_momentum : {None, float}
            Momentum parameter for 'MomentumSGD' method, should be
            a float between 0 and 1.
            If None, set to 0.5
        learning_rate : {None, float}
            Deprecated in favour of subspace_learning_rate
        momentum : {None, float}
            Deprecated in favour of subspace_momentum

        """
        self.n_features = None
        self.normalize = False

        self.t = 0

        # Check options if None
        if method is None:
            _logger.info("No method specified. Defaulting to 'CF' (closed-form solver)")
            method = "CF"
        if init is None:
            _logger.info(
                "No initialization specified. Defaulting to 'qr' initialization"
            )
            init = "qr"
        if training_samples is None and not isinstance(init, np.ndarray):
            if init == "qr":
                if rank >= 10:
                    training_samples = rank
                else:
                    training_samples = 10
                _logger.info(
                    "Number of training samples for 'qr' method "
                    "not specified. Defaulting to %d samples" % training_samples
                )
        if subspace_learning_rate is None:
            if method in ("SGD", "MomentumSGD"):
                _logger.info("Learning rate for SGD algorithm is set to default: 1.0")
                subspace_learning_rate = 1.0
        if subspace_momentum is None:
            if method == "MomentumSGD":
                _logger.info(
                    "Momentum parameter for SGD algorithm is set to default: 0.5"
                )
                subspace_momentum = 0.5

        if learning_rate is not None:
            warnings.warn(
                "The argument `learning_rate` has been deprecated and may "
                "be removed in future. Please use `subspace_learning_rate` instead.",
                VisibleDeprecationWarning,
            )
        if momentum is not None:
            warnings.warn(
                "The argument `momentum` has been deprecated and may "
                "be removed in future. Please use `subspace_momentum` instead.",
                VisibleDeprecationWarning,
            )

        self.rank = rank
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.method = method
        self.init = init
        self.training_samples = training_samples
        self.subspace_learning_rate = subspace_learning_rate
        self.subspace_momentum = subspace_momentum

        # Check options are valid
        if method not in ("CF", "BCD", "SGD", "MomentumSGD"):
            raise ValueError("'method' not recognised")
        if not isinstance(init, np.ndarray) and init not in ("qr", "rand"):
            raise ValueError("'init' not recognised")
        if not isinstance(init, np.ndarray):
            if init == "qr" and training_samples < rank:
                raise ValueError("'training_samples' must be >= 'output_dimension'")
        if method == "MomentumSGD" and (
            subspace_momentum > 1.0 or subspace_momentum < 0.0
        ):
            raise ValueError("'subspace_momentum' must be a float between 0 and 1")

    def _setup(self, X, normalize=False):

        if isinstance(X, np.ndarray):
            n, m = X.shape
            iterating = False
        else:
            x = next(X)
            m = len(x)
            X = chain([x], X)
            iterating = True

        self.n_features = m
        self.iterating = iterating

        if self.lambda1 is None:
            _logger.info(
                "Nuclear norm regularization parameter "
                "is set to default: 1 / sqrt(n_features)"
            )
            self.lambda1 = 1.0 / np.sqrt(m)
        if self.lambda2 is None:
            _logger.info(
                "Sparse regularization parameter "
                "is set to default: 1 / sqrt(n_features)"
            )
            self.lambda2 = 1.0 / np.sqrt(m)

        self.L = self._initialize(X)
        self.K = self.lambda1 * np.eye(self.rank)
        self.R = []
        self.E = []

        # Extra variables for CF and BCD methods
        if self.method in ("CF", "BCD"):
            self.A = np.zeros((self.rank, self.rank))
            self.B = np.zeros((m, self.rank))
        if self.method == "MomentumSGD":
            self.vnew = np.zeros_like(self.L)
        return X

    def _initialize(self, X):
        m = self.n_features
        iterating = self.iterating

        # Initialize the subspace estimate
        if isinstance(self.init, np.ndarray):
            if self.init.ndim != 2:
                raise ValueError("'init' has to be a two-dimensional matrix")
            init_m, init_r = self.init.shape
            if init_m != m or init_r != self.rank:
                raise ValueError("'init' has to be of shape [n_features x rank]")
            return self.init.copy()
        elif self.init == "qr":
            if iterating:
                Y2 = np.stack([next(X) for _ in range(self.training_samples)], axis=-1)
                X = chain(iter(Y2.T.copy()), X)
            else:
                Y2 = X[: self.training_samples, :].T
            L, _ = scipy.linalg.qr(Y2, mode="economic")
            return L[:, : self.rank]
        elif self.init == "rand":
            Y2 = np.random.randn(m, self.rank)
            L, _ = scipy.linalg.qr(Y2, mode="economic")
            return L[:, : self.rank]

    def fit(self, X, iterating=None):
        """Learn RPCA components from the data.

        Parameters
        ----------
        X : {numpy.ndarray, iterator}
            [n_samples x n_features] matrix of observations
            or an iterator that yields samples, each with n_features elements.
        iterating : {None, int}
            If not None, learn the data in batches.

        """
        if self.n_features is None:
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
            lambda2 = self.lambda2  # * z.max()
            lambda1 = self.lambda1  # * z.max()

            r, e = _solveproj(z, self.L, self.K, lambda2)

            self.R.append(r)
            if not iterating:
                self.E.append(e)

            if self.method == "CF":
                # Closed-form solution
                self.A += np.outer(r, r.T)
                self.B += np.outer((z - e), r.T)
                self.L = np.dot(self.B, scipy.linalg.inv(self.A + self.K))
            elif self.method == "BCD":
                # Block-coordinate descent
                self.A += np.outer(r, r.T)
                self.B += np.outer((z - e), r.T)
                self.L = _updatecol(self.L, self.A, self.B, self.K)
            elif self.method == "SGD":
                # Stochastic gradient descent
                learn = self.subspace_learning_rate * (
                    1 + self.subspace_learning_rate * lambda1 * self.t
                )
                self.L -= (
                    np.dot(self.L, np.outer(r, r.T))
                    - np.outer((z - e), r.T)
                    + lambda1 * self.L
                ) / learn
            elif self.method == "MomentumSGD":
                # Stochastic gradient descent with momentum
                learn = self.subspace_learning_rate * (
                    1 + self.subspace_learning_rate * lambda1 * self.t
                )
                vold = self.subspace_momentum * self.vnew
                self.vnew = (
                    np.dot(self.L, np.outer(r, r.T))
                    - np.outer((z - e), r.T)
                    + lambda1 * self.L
                ) / learn
                self.L -= vold + self.vnew
            self.t += 1

    def project(self, X):
        """Project the learnt components on the data.

        Parameters
        ----------
        X : {numpy.ndarray, iterator}
            [n_samples x n_features] matrix of observations
            or an iterator that yields n_samples, each with n_features elements.

        """
        R = []

        num = None
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)
        for v in progressbar(X, leave=False, total=num):
            r, _ = _solveproj(v, self.L, self.K, self.lambda2)
            R.append(r.copy())

        return np.stack(R, axis=-1)

    def finish(self, **kwargs):
        """Return the learnt factors and loadings."""
        R = np.stack(self.R, axis=-1)
        Xhat = np.dot(self.L, R)

        if len(self.E):
            Ehat = np.stack(self.E, axis=-1)
            # both keep an indicator that we had something and remove the
            # duplicate data
            self.E = [1]

        if len(self.E):
            return Xhat, Ehat
        else:
            return Xhat, 1


def orpca(
    X,
    rank,
    project=False,
    lambda1=None,
    lambda2=None,
    method=None,
    init=None,
    training_samples=None,
    subspace_learning_rate=None,
    subspace_momentum=None,
    learning_rate=None,
    momentum=None,
    **kwargs
):
    """Perform online, robust PCA on the data X.

    This is a wrapper function for the ORPCA class.

    Parameters
    ----------
    X : {numpy array, iterator}
        [n_features x n_samples] matrix of observations
        or an iterator that yields samples, each with n_features elements.
    rank : int
        The rank of the representation (number of components/factors)
    project : bool, default False
        If True, project the data X onto the learnt model.
    lambda1 : {None, float}
        Nuclear norm regularization parameter.
        If None, set to 1 / sqrt(n_samples)
    lambda2 : {None, float}
        Sparse error regularization parameter.
        If None, set to 1 / sqrt(n_samples)
    method : {None, 'CF', 'BCD', 'SGD', 'MomentumSGD'}
        'CF'  - Closed-form solver
        'BCD' - Block-coordinate descent
        'SGD' - Stochastic gradient descent
        'MomentumSGD' - Stochastic gradient descent with momentum
        If None, set to 'CF'
    init : {None, 'qr', 'rand', np.ndarray}
        * 'qr'   - QR-based initialization
        * 'rand' - Random initialization
        * np.ndarray if the shape [n_features x rank].
        * If None (default), set to 'qr'
    training_samples : {None, integer}
        Specifies the number of training samples to use in
        the 'qr' initialization.
        If None, set to 10
    subspace_learning_rate : {None, float}
        Learning rate for the stochastic gradient
        descent algorithm
        If None, set to 1
    subspace_momentum : {None, float}
        Momentum parameter for 'MomentumSGD' method, should be
        a float between 0 and 1.
        If None, set to 0.5
    learning_rate : {None, float}
        Deprecated in favour of subspace_learning_rate
    momentum : {None, float}
        Deprecated in favour of subspace_momentum

    Returns
    -------
    If project:
        Returns L, R the low-rank factors and loadings only
    If not project:
        Xhat : numpy array
            is the [n_features x n_samples] low-rank matrix
        Ehat : numpy array
            is the [n_features x n_samples] sparse error matrix
        U, S, V : numpy arrays
            are the results of an SVD on Xhat

    Notes
    -----
    The ORPCA code is based on a transcription of MATLAB code from [Feng2013]_.
    It has been updated to include a new initialization method based
    on a QR decomposition of the first n "training" samples of the data.
    A stochastic gradient descent (SGD) solver is also implemented,
    along with a MomentumSGD solver for improved convergence and robustness
    with respect to local minima. More information about the gradient descent
    methods and choosing appropriate parameters can be found in [Ruder2016]_.

    References
    ----------
    .. [Feng2013] Jiashi Feng, Huan Xu and Shuicheng Yuan, "Online Robust PCA via
           Stochastic Optimization", Advances in Neural Information Processing
           Systems 26, (2013), pp. 404-412.
    .. [Ruder2016] Sebastian Ruder, "An overview of gradient descent optimization
           algorithms", arXiv:1609.04747, (2016), http://arxiv.org/abs/1609.04747.

    """
    X = X.T
    _orpca = ORPCA(
        rank,
        lambda1=lambda1,
        lambda2=lambda2,
        method=method,
        init=init,
        training_samples=training_samples,
        subspace_learning_rate=subspace_learning_rate,
        subspace_momentum=subspace_momentum,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    _orpca._setup(X, normalize=True)
    _orpca.fit(X)

    if project:
        L = _orpca.L
        R = _orpca.project(X)

        return L, R
    else:
        Xhat, Ehat = _orpca.finish(**kwargs)

        # Do final SVD
        U, S, Vh = svd_solve(Xhat, output_dimension=rank)
        V = Vh.T

        # Chop small singular values which
        # likely arise from numerical noise
        # in the SVD.
        S[rank:] = 0.0

        return Xhat, Ehat, U, S, V
