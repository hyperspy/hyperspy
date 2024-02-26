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

import logging
from itertools import chain

import numpy as np
import scipy.linalg

from hyperspy.external.progressbar import progressbar
from hyperspy.learn.svd_pca import svd_solve
from hyperspy.misc.math_tools import check_random_state

_logger = logging.getLogger(__name__)


def _soft_thresh(X, lambda1):
    """Soft-thresholding of array X."""
    res = abs(X) - lambda1
    np.maximum(res, 0.0, out=res)
    res *= np.sign(X)
    return res


def rpca_godec(
    X, rank, lambda1=None, power=0, tol=1e-3, maxiter=1000, random_state=None, **kwargs
):
    """Perform Robust PCA with missing or corrupted data, using the GoDec algorithm.

    Decomposes a matrix Y = X + E, where X is low-rank and E
    is a sparse error matrix. This algorithm is based on the
    Matlab code from [Zhou2011]_. See code here:
    https://sites.google.com/site/godecomposition/matrix/artifact-1

    Read more in the :ref:`User Guide <mva.rpca>`.

    Parameters
    ----------
    X : numpy.ndarray
        The matrix of observations with shape (n_features, n_samples)
    rank : int
        The model dimensionality.
    lambda1 : None or float
        Regularization parameter.
        If None, set to 1 / sqrt(n_features)
    power : int, default 0
        The number of power iterations used in the initialization
    tol : float, default 1e-3
        Convergence tolerance
    maxiter : int, default 1000
        Maximum number of iterations
    random_state : None, int or RandomState, default None
        Used to initialize the subspace on the first iteration.

    Returns
    -------
    Xhat : numpy.ndarray
        The low-rank matrix with shape (n_features, n_samples)
    Ehat : numpy.ndarray
        The sparse error matrix with shape (n_features, n_samples)
    U, S, V : numpy.ndarray
        The results of an SVD on Xhat

    References
    ----------
    .. [Zhou2011] Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Low-rank &
        Sparse Matrix Decomposition in Noisy Case", ICML-11, (2011), pp. 33-40.

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

    # Initialize L and E
    L = X
    E = np.zeros(L.shape)

    random_state = check_random_state(random_state)

    for itr in range(int(maxiter)):
        # Initialization with bilateral random projections
        Y2 = random_state.normal(size=(n, rank))
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
            _logger.info(f"Converged to {eps} in {itr} iterations")
            break

    # Transpose back
    if transpose:
        L = L.T
        E = E.T

    # Rescale
    Xhat = L
    Ehat = E

    # Do final SVD
    U, S, Vh = svd_solve(Xhat, output_dimension=rank, **kwargs)
    V = Vh.T

    # Chop small singular values which
    # likely arise from numerical noise
    # in the SVD.
    S[rank:] = 0.0

    return Xhat, Ehat, U, S, V


def _solveproj(z, X, Id, lambda2, r=None, e=None):
    m, n = X.shape
    z = z.T

    if len(z.shape) == 2:
        batch_size = z.shape[1]
        eshape = (m, batch_size)
        rshape = (n, batch_size)
    else:
        eshape = (m,)
        rshape = (n,)
    if r is None or r.shape != rshape:
        r = np.zeros(rshape)
    if e is None or e.shape != eshape:
        e = np.zeros(eshape)

    ddt = np.linalg.solve(X.T @ X + Id, X.T)
    maxiter = 1e6
    itr = 0

    while True:
        itr += 1
        # Solve for r
        rtmp = r
        r = ddt @ (z - e)

        # Solve for e
        etmp = e
        e = _soft_thresh(z - X @ r, lambda2)

        # Stop conditions
        stopr = np.linalg.norm(r - rtmp, 2)
        stope = np.linalg.norm(e - etmp, 2)
        stop = max(stopr, stope) / m
        if stop < 1e-5 or itr > maxiter:
            break

    return r, e


def _updatecol(X, A, B, Id):
    tmp, n = X.shape
    L = X
    A = A + Id

    for i in range(n):
        b = B[:, i]
        x = X[:, i]
        a = A[:, i]
        temp = (b - X @ a) / A[i, i] + x
        L[:, i] = temp / max(np.linalg.norm(temp, 2), 1)

    return L


class ORPCA:
    """Performs Online Robust PCA with missing or corrupted data.

    The ORPCA code is based on a transcription of MATLAB code from [Feng2013]_.
    It has been updated to include a new initialization method based
    on a QR decomposition of the first n "training" samples of the data.
    A stochastic gradient descent (SGD) solver is also implemented,
    along with a MomentumSGD solver for improved convergence and robustness
    with respect to local minima. More information about the gradient descent
    methods and choosing appropriate parameters can be found in [Ruder2016]_.

    Read more in the :ref:`User Guide <mva.rpca>`.

    References
    ----------
    .. [Feng2013] Jiashi Feng, Huan Xu and Shuicheng Yuan, "Online Robust PCA
        via Stochastic Optimization", Advances in Neural Information Processing
        Systems 26, (2013), pp. 404-412.
    .. [Ruder2016] Sebastian Ruder, "An overview of gradient descent optimization
        algorithms", arXiv:1609.04747, (2016), https://arxiv.org/abs/1609.04747.

    """

    def __init__(
        self,
        rank,
        store_error=False,
        lambda1=0.1,
        lambda2=1.0,
        method="BCD",
        init="qr",
        training_samples=10,
        subspace_learning_rate=1.0,
        subspace_momentum=0.5,
        random_state=None,
    ):
        """Creates Online Robust PCA instance that can learn a representation.

        Parameters
        ----------
        rank : int
            The rank of the representation (number of components/factors)
        store_error : bool, default False
            If True, stores the sparse error matrix.
        lambda1 : float, default 0.1
            Nuclear norm regularization parameter.
        lambda2 : float, default 1.0
            Sparse error regularization parameter.
        method : {'CF', 'BCD', 'SGD', 'MomentumSGD'}, default 'BCD'
            * ``'CF'``  - Closed-form solver
            * ``'BCD'`` - Block-coordinate descent
            * ``'SGD'`` - Stochastic gradient descent
            * ``'MomentumSGD'`` - Stochastic gradient descent with momentum
        init : numpy.ndarray, {'qr', 'rand'}, default 'qr'
            * ``'qr'``   - QR-based initialization
            * ``'rand'`` - Random initialization
            * numpy.ndarray if the shape (n_features x rank)
        training_samples : int, default 10
            Specifies the number of training samples to use in
            the 'qr' initialization.
        subspace_learning_rate : float, default 1.0
            Learning rate for the 'SGD' and 'MomentumSGD' methods.
            Should be a float > 0.0
        subspace_momentum : float, default 0.5
            Momentum parameter for 'MomentumSGD' method, should be
            a float between 0 and 1.
        random_state : None, int or RandomState, default None
            Used to initialize the subspace on the first iteration.

        """
        self.n_features = None
        self.iterating = False
        self.t = 0

        if store_error:
            self.E = []
        else:
            self.E = None

        self.rank = rank
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.method = method
        self.init = init
        self.training_samples = training_samples
        self.subspace_learning_rate = subspace_learning_rate
        self.subspace_momentum = subspace_momentum
        self.random_state = check_random_state(random_state)

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

    def _setup(self, X):
        self.r, self.e, self.v = None, None, None
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

        self.L = self._initialize_subspace(X)
        self.K = self.lambda1 * np.eye(self.rank)
        self.R = []

        # Extra variables for CF and BCD methods
        if self.method in ("CF", "BCD"):
            self.A = np.zeros((self.rank, self.rank))
            self.B = np.zeros((m, self.rank))
        elif self.method == "MomentumSGD":
            self.vnew = np.zeros_like(self.L)

        return X

    def _initialize_subspace(self, X):
        """Initialize the subspace estimate."""
        m = self.n_features

        if isinstance(self.init, np.ndarray):
            if self.init.ndim != 2:
                raise ValueError("'init' has to be a two-dimensional matrix")
            init_m, init_r = self.init.shape
            if init_m != m or init_r != self.rank:
                raise ValueError("'init' has to be of shape [n_features x rank]")
            return self.init.copy()
        elif self.init == "qr":
            if self.iterating:
                Y2 = np.stack([next(X) for _ in range(self.training_samples)], axis=-1)
                X = chain(iter(Y2.T.copy()), X)
            else:
                Y2 = X[: self.training_samples, :].T
            L, _ = scipy.linalg.qr(Y2, mode="economic")
            return L[:, : self.rank]
        elif self.init == "rand":
            Y2 = self.random_state.normal(size=(m, self.rank))
            L, _ = scipy.linalg.qr(Y2, mode="economic")
            return L[:, : self.rank]

    def fit(self, X, batch_size=None):
        """Learn RPCA components from the data.

        Parameters
        ----------
        X : array-like
            The matrix of observations with shape (n_samples, n_features)
            or an iterator that yields samples, each with n_features elements.
        batch_size : None or int
            If not None, learn the data in batches, each of batch_size samples
            or less.

        """
        if self.n_features is None:
            X = self._setup(X)

        num = None
        prod = np.outer
        if batch_size is not None:
            if not isinstance(X, np.ndarray):
                raise ValueError("can't batch iterating data")
            else:
                prod = np.dot
                length = X.shape[0]
                num = max(length // batch_size, 1)
                X = np.array_split(X, num, axis=0)

        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)

        r, e = self.r, self.e

        for v in progressbar(X, leave=False, total=num, disable=num == 1):
            r, e = _solveproj(v, self.L, self.K, self.lambda2, r=r, e=e)
            self.v = v
            self.r = r
            self.e = e
            self.R.append(r)
            if self.E is not None:
                self.E.append(e)

            self._solve_L(prod(r, r.T), prod((v.T - e), r.T))
            self.t += 1

        self.r = r
        self.e = e

    def _solve_L(self, A, B):
        if self.method == "CF":
            # Closed-form solution
            self.A += A
            self.B += B
            self.L = self.B @ np.linalg.inv(self.A + self.K)
        elif self.method == "BCD":
            # Block-coordinate descent
            self.A += A
            self.B += B
            self.L = _updatecol(self.L, self.A, self.B, self.K)
        elif self.method == "SGD":
            # Stochastic gradient descent
            learn = self.subspace_learning_rate * (
                1 + self.subspace_learning_rate * self.lambda1 * self.t
            )
            self.L -= (self.L @ A - B + self.lambda1 * self.L) / learn
        elif self.method == "MomentumSGD":
            # Stochastic gradient descent with momentum
            learn = self.subspace_learning_rate * (
                1 + self.subspace_learning_rate * self.lambda1 * self.t
            )
            vold = self.subspace_momentum * self.vnew
            self.vnew = (self.L @ A - B + self.lambda1 * self.L) / learn
            self.L -= vold + self.vnew

    def project(self, X, return_error=False):
        """Project the learnt components on the data.

        Parameters
        ----------
        X : array-like
            The matrix of observations with shape (n_samples, n_features)
            or an iterator that yields n_samples, each with n_features elements.
        return_error : bool, default False
            If True, returns the sparse error matrix as well. Otherwise only
            the weights (loadings)

        """
        R = []
        if return_error:
            E = []

        num = None
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)
        for v in progressbar(X, leave=False, total=num):
            r, e = _solveproj(v, self.L, self.K, self.lambda2)
            R.append(r.copy())
            if return_error:
                E.append(e.copy())

        R = np.stack(R, axis=-1)
        if return_error:
            return R, np.stack(E, axis=-1)
        else:
            return R

    def finish(self, **kwargs):
        """Return the learnt factors and loadings."""
        if len(self.R) > 0:
            if len(self.R[0].shape) == 1:
                R = np.stack(self.R, axis=-1)
            else:
                R = np.concatenate(self.R, axis=1)
            return self.L, R
        else:
            return self.L, 1


def orpca(
    X,
    rank,
    store_error=False,
    project=False,
    batch_size=None,
    lambda1=0.1,
    lambda2=1.0,
    method="BCD",
    init="qr",
    training_samples=10,
    subspace_learning_rate=1.0,
    subspace_momentum=0.5,
    random_state=None,
    **kwargs,
):
    """Perform online, robust PCA on the data X.

    This is a wrapper function for the ORPCA class.

    Parameters
    ----------
    X : array-like
        The matrix of observations with shape (n_features x n_samples)
        or an iterator that yields samples, each with n_features elements.
    rank : int
        The rank of the representation (number of components/factors)
    store_error : bool, default False
        If True, stores the sparse error matrix.
    project : bool, default False
        If True, project the data X onto the learnt model.
    batch_size : None, int, default None
        If not None, learn the data in batches, each of batch_size samples
        or less.
    lambda1 : float, default 0.1
        Nuclear norm regularization parameter.
    lambda2 : float, default 1.0
        Sparse error regularization parameter.
    method : {'CF', 'BCD', 'SGD', 'MomentumSGD'}, default 'BCD'
        * ``'CF'``  - Closed-form solver
        * ``'BCD'`` - Block-coordinate descent
        * ``'SGD'`` - Stochastic gradient descent
        * ``'MomentumSGD'`` - Stochastic gradient descent with momentum
    init : numpy.ndarray, {'qr', 'rand'}, default 'qr'
        * ``'qr'``   - QR-based initialization
        * ``'rand'`` - Random initialization
        * numpyp.ndarray if the shape [n_features x rank]
    training_samples : int, default 10
        Specifies the number of training samples to use in
        the 'qr' initialization.
    subspace_learning_rate : float, default 1.0
        Learning rate for the 'SGD' and 'MomentumSGD' methods.
        Should be a float > 0.0
    subspace_momentum : float, default 0.5
        Momentum parameter for 'MomentumSGD' method, should be
        a float between 0 and 1.
    random_state : None or int or RandomState, default None
        Used to initialize the subspace on the first iteration.

    Returns
    -------
    numpy.ndarray
        * If project is True, returns the low-rank factors and loadings only
        * Otherwise, returns the low-rank and sparse error matrices, as well
          as the results of a singular value decomposition (SVD) applied to
          the low-rank matrix.

    """

    X = X.T

    _orpca = ORPCA(
        rank,
        store_error=store_error,
        lambda1=lambda1,
        lambda2=lambda2,
        method=method,
        init=init,
        training_samples=training_samples,
        subspace_learning_rate=subspace_learning_rate,
        subspace_momentum=subspace_momentum,
        random_state=random_state,
    )
    _orpca.fit(X, batch_size=batch_size)

    if project:
        L = _orpca.L
        R = _orpca.project(X)
    else:
        L, R = _orpca.finish()

    if store_error:
        Xhat = L @ R
        Ehat = np.array(_orpca.E).T

        # Do final SVD
        U, S, Vh = svd_solve(Xhat, output_dimension=rank)
        V = Vh.T

        # Chop small singular values which
        # likely arise from numerical noise
        # in the SVD.
        S[rank:] = 0.0

        return Xhat, Ehat, U, S, V
    else:
        return L, R
