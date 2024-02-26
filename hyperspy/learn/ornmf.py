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
from scipy.stats import halfnorm

from hyperspy.external.progressbar import progressbar
from hyperspy.misc.math_tools import check_random_state

_logger = logging.getLogger(__name__)


def _thresh(X, lambda1, vmax):
    """Soft-thresholding with clipping."""
    res = abs(X) - lambda1
    np.maximum(res, 0.0, out=res)
    res *= np.sign(X)
    np.clip(res, -vmax, vmax, out=res)
    return res


def _mrdivide(B, A):
    """Solves xB = A as per Matlab."""
    if isinstance(B, np.ndarray):
        if len(B.shape) == 2 and B.shape[0] == B.shape[1]:
            # square array
            return np.linalg.solve(A.T, B.T).T
        else:
            # Set rcond default value to match numpy 1.14 default value with
            # previous numpy version
            rcond = np.finfo(float).eps * max(A.shape)
            return np.linalg.lstsq(A.T, B.T, rcond=rcond)[0].T
    else:
        return B / A


def _project(W):
    newW = W.copy()
    np.maximum(newW, 0, out=newW)
    sumsq = np.sqrt(np.sum(W**2, axis=0))
    np.maximum(sumsq, 1, out=sumsq)
    return _mrdivide(newW, np.diag(sumsq))


def _solveproj(v, W, lambda1, kappa=1, h=None, e=None, vmax=None):
    m, n = W.shape
    v = v.T
    if vmax is None:
        vmax = v.max()

    if len(v.shape) == 2:
        batch_size = v.shape[1]
        eshape = (m, batch_size)
        hshape = (n, batch_size)
    else:
        eshape = (m,)
        hshape = (n,)
    if h is None or h.shape != hshape:
        h = np.zeros(hshape)
    if e is None or e.shape != eshape:
        e = np.zeros(eshape)

    eta = kappa / np.linalg.norm(W, "fro") ** 2

    maxiter = 1e6
    iters = 0

    while True:
        iters += 1
        # Solve for h
        htmp = h
        h = h - eta * W.T @ (W @ h + e - v)
        np.maximum(h, 0.0, out=h)

        # Solve for e
        etmp = e
        e = _thresh(v - W @ h, lambda1, vmax)

        # Stop conditions
        stoph = np.linalg.norm(h - htmp, 2)
        stope = np.linalg.norm(e - etmp, 2)
        stop = max(stoph, stope) / m
        if stop < 1e-5 or iters > maxiter:
            break

    return h, e


class ORNMF:
    """Performs Online Robust NMF with missing or corrupted data.

    The ORNMF code is based on a transcription of the online proximal gradient
    descent (PGD) algorithm MATLAB code obtained from the authors of [Zhao2016]_.
    It has been updated to also include L2-normalization cost function that
    is able to deal with sparse corruptions and/or outliers slightly faster
    (please see ORPCA implementation for details). A further modification
    has been made to allow for a changing subspace W, where X ~= WH^T + E
    in the ORNMF framework.

    Read more in the :ref:`User Guide <mva.rnmf>`.

    References
    ----------
    .. [Zhao2016] Zhao, Renbo, and Vincent YF Tan. "Online nonnegative matrix
        factorization with outliers." Acoustics, Speech and Signal Processing
        (ICASSP), 2016 IEEE International Conference on. IEEE, 2016.

    """

    def __init__(
        self,
        rank,
        store_error=False,
        lambda1=1.0,
        kappa=1.0,
        method="PGD",
        subspace_learning_rate=1.0,
        subspace_momentum=0.5,
        random_state=None,
    ):
        """Creates Online Robust NMF instance that can learn a representation.

        Parameters
        ----------
        rank : int
            The rank of the representation (number of components/factors)
        store_error : bool, default False
            If True, stores the sparse error matrix.
        lambda1 : float
            Nuclear norm regularization parameter.
        kappa : float
            Step-size for projection solver.
        method : {``'PGD'``, ``'RobustPGD'``, ``'MomentumSGD'``}, default ``'PGD'``
            * ``'PGD'`` - Proximal gradient descent
            * ``'RobustPGD'`` - Robust proximal gradient descent
            * ``'MomentumSGD'`` - Stochastic gradient descent with momentum
        subspace_learning_rate : float
            Learning rate for the ``'MomentumSGD'`` method. Should be a
            float > 0.0
        subspace_momentum : float
            Momentum parameter for ``'MomentumSGD'`` method, should be
            a float between 0 and 1.
        random_state : None or int or RandomState, default None
            Used to initialize the subspace on the first iteration.
            See :func:`numpy.random.default_rng` for more information.

        """
        self.n_features = None
        self.iterating = False
        self.t = 0

        if store_error:
            self.E = []
        else:
            self.E = None

        self.rank = rank
        self.robust = False
        self.subspace_tracking = False
        self.lambda1 = lambda1
        self.kappa = kappa
        self.subspace_learning_rate = subspace_learning_rate
        self.subspace_momentum = subspace_momentum
        self.random_state = check_random_state(random_state)

        # Check options are valid
        if method not in ("PGD", "RobustPGD", "MomentumSGD"):
            raise ValueError("'method' not recognised")

        if method == "RobustPGD":
            self.robust = True

        if method == "MomentumSGD":
            self.subspace_tracking = True
            if subspace_momentum < 0.0 or subspace_momentum > 1:
                raise ValueError("'subspace_momentum' must be a float between 0 and 1")

    def _setup(self, X):
        self.h, self.e, self.v = None, None, None
        if isinstance(X, np.ndarray):
            n, m = X.shape
            avg = np.sqrt(X.mean() / m)
            iterating = False
        else:
            x = next(X)
            m = len(x)
            avg = np.sqrt(x.mean() / m)
            X = chain([x], X)
            iterating = True

        self.n_features = m
        self.iterating = iterating

        self.W = halfnorm.rvs(
            size=(self.n_features, self.rank), random_state=self.random_state
        )
        self.W = abs(avg * self.W / np.sqrt(self.rank))
        self.H = []

        if self.subspace_tracking:
            self.vnew = np.zeros_like(self.W)
        else:
            self.A = np.zeros((self.rank, self.rank))
            self.B = np.zeros((self.n_features, self.rank))

        return X

    def fit(self, X, batch_size=None):
        """Learn NMF components from the data.

        Parameters
        ----------
        X : array-like
            [n_samples x n_features] matrix of observations
            or an iterator that yields samples, each with n_features elements.
        batch_size : {None, int}
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

        h, e = self.h, self.e

        for v in progressbar(X, leave=False, total=num, disable=num == 1):
            h, e = _solveproj(v, self.W, self.lambda1, self.kappa, h=h, e=e)
            self.v = v
            self.e = e
            self.h = h
            self.H.append(h)
            if self.E is not None:
                self.E.append(e)

            self._solve_W(prod(h, h.T), prod((v.T - e), h.T))
            self.t += 1

        self.h = h
        self.e = e

    def _solve_W(self, A, B):
        if not self.subspace_tracking:
            self.A += A
            self.B += B
            eta = self.kappa / np.linalg.norm(self.A, "fro")

        if self.robust:
            # exactly as in the Zhao & Tan paper
            n = 0
            lasttwo = np.zeros(2)
            while n <= 2 or (
                abs((lasttwo[1] - lasttwo[0]) / lasttwo[0]) > 1e-5 and n < 1e9
            ):
                self.W -= eta * (self.W @ self.A - self.B)
                self.W = _project(self.W)
                n += 1
                lasttwo[0] = lasttwo[1]
                lasttwo[1] = 0.5 * np.trace(
                    self.W.T.dot(self.W).dot(self.A)
                ) - np.trace(self.W.T.dot(self.B))
        else:
            # Tom Furnival (@tjof2) approach
            # - copied from the ORPCA implementation
            #   of gradient descent in ./rpca.py
            if self.subspace_tracking:
                learn = self.subspace_learning_rate * (
                    1 + self.subspace_learning_rate * self.lambda1 * self.t
                )
                vold = self.subspace_momentum * self.vnew
                self.vnew = (self.W @ A - B) / learn
                self.W -= vold + self.vnew
            else:
                self.W -= eta * (self.W @ self.A - self.B)

            np.maximum(self.W, 0.0, out=self.W)
            self.W /= max(np.linalg.norm(self.W, "fro"), 1.0)

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
        H = []
        if return_error:
            E = []

        num = None
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)
        for v in progressbar(X, leave=False, total=num):
            h, e = _solveproj(v, self.W, self.lambda1, self.kappa, vmax=np.inf)
            H.append(h.copy())
            if return_error:
                E.append(e.copy())

        H = np.stack(H, axis=-1)
        if return_error:
            return H, np.stack(E, axis=-1)
        else:
            return H

    def finish(self):
        """Return the learnt factors and loadings."""
        if len(self.H) > 0:
            if len(self.H[0].shape) == 1:
                H = np.stack(self.H, axis=-1)
            else:
                H = np.concatenate(self.H, axis=1)
            return self.W, H
        else:
            return self.W, 1


def ornmf(
    X,
    rank,
    store_error=False,
    project=False,
    batch_size=None,
    lambda1=1.0,
    kappa=1.0,
    method="PGD",
    subspace_learning_rate=1.0,
    subspace_momentum=0.5,
    random_state=None,
):
    """Perform online, robust NMF on the data X.

    This is a wrapper function for the ORNMF class.

    Parameters
    ----------
    X : numpy.ndarray
        The [n_samples, n_features] input data.
    rank : int
        The rank of the representation (number of components/factors)
    store_error : bool, default False
        If True, stores the sparse error matrix.
    project : bool, default False
        If True, project the data X onto the learnt model.
    batch_size : None or int, default None
        If not None, learn the data in batches, each of batch_size samples
        or less.
    lambda1 : float, default 1.0
        Nuclear norm regularization parameter.
    kappa : float, default 1.0
        Step-size for projection solver.
    method : {'PGD', 'RobustPGD', 'MomentumSGD'}, default 'PGD'
        * ``'PGD'`` - Proximal gradient descent
        * ``'RobustPGD'`` - Robust proximal gradient descent
        * ``'MomentumSGD'`` - Stochastic gradient descent with momentum
    subspace_learning_rate : float, default 1.0
        Learning rate for the 'MomentumSGD' method. Should be a
        float > 0.0
    subspace_momentum : float, default 0.5
        Momentum parameter for 'MomentumSGD' method, should be
        a float between 0 and 1.
    random_state : None or int or RandomState, default None
        Used to initialize the subspace on the first iteration.

    Returns
    -------
    Xhat : numpy.ndarray
        The non-negative matrix with shape (n_features x n_samples).
        Only returned if store_error is True.
    Ehat : numpy.ndarray
        The sparse error matrix with shape (n_features x n_samples).
        Only returned if store_error is True.
    W : numpy.ndarray
        The non-negative factors matrix with shape (n_features, rank).
    H : numpy.ndarray
        The non-negative loadings matrix with shape (rank, n_samples).

    """
    X = X.T

    _ornmf = ORNMF(
        rank,
        store_error=store_error,
        lambda1=lambda1,
        kappa=kappa,
        method=method,
        subspace_learning_rate=subspace_learning_rate,
        subspace_momentum=subspace_momentum,
        random_state=random_state,
    )
    _ornmf.fit(X, batch_size=batch_size)

    if project:
        W = _ornmf.W
        H = _ornmf.project(X)
    else:
        W, H = _ornmf.finish()

    if store_error:
        Xhat = W @ H
        Ehat = np.array(_ornmf.E).T

        return Xhat, Ehat, W, H
    else:
        return W, H
