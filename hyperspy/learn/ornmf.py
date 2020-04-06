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
from itertools import chain

import numpy as np
from scipy.stats import halfnorm

from hyperspy.external.progressbar import progressbar

_logger = logging.getLogger(__name__)


def _thresh(X, lambda1, vmax):
    """Soft-thresholding with clipping."""
    res = np.abs(X) - lambda1
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
    sumsq = np.sqrt(np.sum(W ** 2, axis=0))
    np.maximum(sumsq, 1, out=sumsq)
    return _mrdivide(newW, np.diag(sumsq))


def _solveproj(v, W, lambda1, kappa=1, h=None, r=None, vmax=None):
    m, n = W.shape
    v = v.T
    if vmax is None:
        vmax = v.max()
    if len(v.shape) == 2:
        batch_size = v.shape[1]
        rshape = (m, batch_size)
        hshape = (n, batch_size)
    else:
        rshape = (m,)
        hshape = (n,)
    if h is None or h.shape != hshape:
        h = np.zeros(hshape)

    if r is None or r.shape != rshape:
        r = np.zeros(rshape)

    eta = kappa / np.linalg.norm(W, "fro") ** 2

    maxiter = 1e9
    iters = 0

    while True:
        iters += 1
        # Solve for h
        htmp = h
        h = h - eta * np.dot(W.T, np.dot(W, h) + r - v)
        np.maximum(h, 0.0, out=h)

        # Solve for r
        rtmp = r
        r = _thresh(v - np.dot(W, h), lambda1, vmax)

        # Stop conditions
        stoph = np.linalg.norm(h - htmp, 2)
        stopr = np.linalg.norm(r - rtmp, 2)
        stop = max(stoph, stopr) / m
        if stop < 1e-5 or iters > maxiter:
            break

    return h, r


class ORNMF:
    """This class performs Online Robust NMF with missing or corrupted data.

    Methods
    -------
    fit
        Learn factors from the given data.
    project
        Project the learnt factors on the given data.
    finish
        Return the learnt factors and loadings.

    Notes
    -----
    The ORNMF code is based on a transcription of the OPGD algorithm MATLAB code
    obtained from the authors of [1]_.

    It has been updated to also include L2-normalization cost function that is
    able to deal with sparse corruptions and/or outliers slightly faster
    (please see ORPCA implementation for details).

    A further modification has been made to allow for a changing subspace W,
    where X ~= WH^T + E in the ORNMF framework.

    References
    ----------
    .. [1] Zhao, Renbo, and Vincent YF Tan. "Online nonnegative matrix
           factorization with outliers." Acoustics, Speech and Signal Processing
           (ICASSP), 2016 IEEE International Conference on. IEEE, 2016.

    """

    def __init__(
        self,
        rank,
        lambda1=1.0,
        kappa=1.0,
        store_error=False,
        method=None,
        subspace_learning_rate=None,
        subspace_momentum=None,
    ):
        """Creates Online Robust NMF instance that can learn a representation.

        Parameters
        ----------
        rank : int
            The rank of the representation (number of components/factors)
        lambda1 : float
            Nuclear norm regularization parameter.
        kappa : float
            Step-size for projection solver.
        store_error : bool
            If True, stores the sparse error matrix, False by default.
        method : {None, 'PGD', 'RobustPGD', 'MomentumSGD'}
            'PGD' - Proximal gradient descent
            'RobustPGD' - Robust proximal gradient descent
            'MomentumSGD' - Stochastic gradient descent with momentum
            If None, set to PGD.
        subspace_learning_rate : float | None
            Learning rate for stochastic gradient descent.
        subspace_momentum : float | None
            Momentum parameter for stochastic gradient descent. Must be in
            range 0 <= x <= 1.

        """
        self.robust = False
        self.subspace_tracking = False
        self.n_features = None
        self.rank = rank
        self.lambda1 = lambda1
        self.kappa = kappa
        self.H = []
        self.t = 0

        if method is None:
            _logger.info(
                "No method specified. Defaulting to "
                "'PGD' (proximal gradient descent)"
            )
            method = "PGD"

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

        if method not in ("PGD", "RobustPGD", "MomentumSGD"):
            raise ValueError("'method' not recognised")

        if method == "RobustPGD":
            self.robust = True

        if method == "MomentumSGD":
            self.subspace_tracking = True
            if subspace_momentum < 0.0 or subspace_momentum > 1:
                raise ValueError("'subspace_momentum' must be a float between 0 and 1")

        self.subspace_learning_rate = subspace_learning_rate
        self.subspace_momentum = subspace_momentum

        if store_error:
            self.E = []
        else:
            self.E = None

    def _setup(self, X, normalize=False):
        self.h, self.r, self.v = None, None, None
        if isinstance(X, np.ndarray):
            n, m = X.shape
            if normalize:
                self.X_min = X.min()
                self.X_max = X.max()
                self.normalize = normalize
                # actually scale the data to be between 0 and 1,
                # not just close to it..
                X = _normalize(X, ar_min=self.X_min, ar_max=self.X_max)

            avg = np.sqrt(X.mean() / m)
        else:
            if normalize:
                _logger.warning(
                    "Normalization with an iterator is not possible, option ignored."
                )
            x = next(X)
            m = len(x)
            avg = np.sqrt(x.mean() / m)
            X = chain([x], X)

        self.n_features = m

        self.W = np.abs(
            avg * halfnorm.rvs(size=(self.n_features, self.rank)) / np.sqrt(self.rank)
        )

        if self.subspace_tracking:
            self.vnew = np.zeros_like(self.W)

        self.A = np.zeros((self.rank, self.rank))
        self.B = np.zeros((self.n_features, self.rank))
        return X

    def fit(self, X, batch_size=None):
        """Learn NMF components from the data.

        Parameters
        ----------
        X : {numpy.ndarray, iterator}
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

        r, h = self.r, self.h

        for v in progressbar(X, leave=False, total=num, disable=num == 1):
            h, r = _solveproj(v, self.W, self.lambda1, self.kappa, r=r, h=h)
            self.v = v
            self.r = r
            self.h = h
            self.H.append(h)
            if self.E is not None:
                self.E.append(r)

            # Only need to update A, B when not tracking subspace
            if not self.subspace_tracking:
                self.A += prod(h, h.T)
                self.B += prod((v.T - r), h.T)

            self._solve_W()
            self.t += 1
        self.r = r
        self.h = h

    def _solve_W(self):
        if not self.subspace_tracking:
            eta = self.kappa / np.linalg.norm(self.A, "fro")

        if self.robust:
            # exactly as in the Zhao & Tan paper
            n = 0
            lasttwo = np.zeros(2)
            while n <= 2 or (
                np.abs((lasttwo[1] - lasttwo[0]) / lasttwo[0]) > 1e-5 and n < 1e9
            ):
                self.W -= eta * (np.dot(self.W, self.A) - self.B)
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
                self.vnew = (
                    np.dot(self.W, np.outer(self.h, self.h.T))
                    - np.outer((self.v.T - self.r), self.h.T)
                ) / learn
                self.W -= vold + self.vnew
            else:
                self.W -= eta * (np.dot(self.W, self.A) - self.B)
            np.maximum(self.W, 0.0, out=self.W)
            self.W /= max(np.linalg.norm(self.W, "fro"), 1.0)

    def project(self, X, return_error=False):
        """Project the learnt components on the data.

        Parameters
        ----------
        X : {numpy.ndarray, iterator}
            [n_samples x n_features] matrix of observations
            or an iterator that yields n_samples, each with n_features elements.
        return_error : bool
            If True, returns the sparse error matrix as well. Otherwise only
            the weights (loadings)

        """
        H = []
        if return_error:
            E = []

        num = None
        W = self.W
        lam1 = self.lambda1
        kap = self.kappa
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)
        for v in progressbar(X, leave=False, total=num):
            # want to start with fresh results and not clip,
            # so that chunks are smooth
            h, r = _solveproj(v, W, lam1, kap, vmax=np.inf)
            H.append(h.copy())
            if return_error:
                E.append(r.copy())

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
    lambda1=1,
    kappa=1,
    store_error=False,
    project=False,
    method=None,
    subspace_learning_rate=1.0,
    subspace_momentum=0.5,
    batch_size=None,
):
    """Perform online, robust NMF on the data X.

    This is a wrapper function for the ORNMF class.

    Parameters
    ----------
    X : numpy array
        The [n_samples, n_features] input data.
    rank : int
        The rank of the representation (number of components/factors)
    lambda1 : float
        Nuclear norm regularization parameter.
    kappa : float
        Step-size for projection solver.
    store_error : bool
        If True, stores the sparse error matrix, False by default.
    project : bool, default False
        If True, project the data X onto the learnt model.
    method : {None, 'PGD', 'RobustPGD', 'MomentumSGD'}
        'PGD' - Proximal gradient descent
        'RobustPGD' - Robust proximal gradient descent
        'MomentumSGD' - Stochastic gradient descent with momentum
        If None, set to PGD.
    subspace_learning_rate : float | None
        Learning rate for stochastic gradient descent.
    subspace_momentum : float | None
        Momentum parameter for stochastic gradient descent. Must be in
        range 0 <= x <= 1.
    batch_size : {None, int}
        If not None, learn the data in batches, each of batch_size samples
        or less.

    Returns
    -------
    Xhat : numpy array
        is the [n_features x n_samples] non-negative matrix
        Only returned if store_error is True.
    Ehat : numpy array
        is the [n_features x n_samples] sparse error matrix
        Only returned if store_error is True.
    W : numpy array, shape [n_features, rank]
        is the non-negative factors matrix
    H : numpy array, shape [rank, n_samples]
        is the non-negative loadings matrix

    """
    X = X.T

    _ornmf = ORNMF(
        rank,
        lambda1=lambda1,
        kappa=kappa,
        store_error=store_error,
        method=method,
        subspace_learning_rate=subspace_learning_rate,
        subspace_momentum=subspace_momentum,
    )
    _ornmf.fit(X, batch_size=batch_size)

    if project:
        W = _ornmf.W
        H = _ornmf.project(X)
    else:
        W, H = _ornmf.finish()

    if store_error:
        Xhat = np.dot(W, H)
        Ehat = np.array(_ornmf.E).T

        return Xhat, Ehat, W, H
    else:
        return W, H
