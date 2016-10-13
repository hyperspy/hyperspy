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
from scipy.stats import halfnorm
from hyperspy.external.progressbar import progressbar


_logger = logging.getLogger(__name__)


def _thresh(X, lambda1, vmax):
    res = np.abs(X) - lambda1
    np.maximum(res, 0.0, out=res)
    res *= np.sign(X)
    np.clip(res, -vmax, vmax, out=res)  
    return res
    
def _solveproj(v, W, lambda1, kappa=1, h=None, r=None):
    m, n = W.shape
    v = v.T
    if len(v.shape) == 2:
        batch_size = v.shape[1]
        rshape = (m, batch_size)
        hshape = (n, batch_size)
        # r = np.zeros((m, batch_size))
        # h = np.zeros((n, batch_size))
    else:
        rshape = m,
        hshape = n,
    if h is None or h.shape != hshape:
        h = np.zeros(hshape)

    if r is None or r.shape != rshape:
        r = np.zeros(rshape)

    eta = kappa / np.linalg.norm(W, 'fro')**2

    maxiter = 1e9
    iters = 0

    while True:        
        iters += 1
        if iters % 10 == 0:
            _logger.debug('solveproj iter #{}'.format(iters))
        # Solve for h
        htmp = h
        h = h - eta * np.dot(W.T, np.dot(W, h) + r - v)
        np.maximum(h, 0.0, out=h)
        
        # Solve for r
        rtmp = r
        r = _thresh(v - np.dot(W, h), lambda1, v.max())
        
        # Stop conditions
        stoph = np.linalg.norm(h - htmp, 2) 
        stopr = np.linalg.norm(r - rtmp, 2)
        stop = max(stoph, stopr) / m
        if stop < 1e-5 or iters > maxiter:            
            break
    
    return h, r

class ONMF:

    def __init__(self, rank, lambda1=1., kappa=1., store_r=False):
        self.nfeatures = None
        self.rank = rank
        self.lambda1 = lambda1
        self.kappa = kappa
        self.H = []
        if store_r:
            self.R = []
        else:
            self.R = None

    def _setup(self, X, normalize=False):
        self.h, self.r = None, None
        if isinstance(X, np.ndarray):
            n, m = X.shape
            if normalize:
                self.X_min = X.min()
                self.X_max = X.max()
                self.normalize = normalize
                # actually scale the data to be between 0 and 1, not just close
                # to it..
                X = _normalize(X, ar_min=self.X_min, ar_max=self.X_max)
                # X = (X - self.X_min) / (self.X_max - self.X_min)
            avg = np.sqrt(X.mean() / m)
        else:
            if normalize:
                _logger.warning("Normalization with an iterator is not"
                                " possible, option ignored.")
            x = next(X)
            m = len(x)
            avg = np.sqrt(x.mean() / m)
            X = chain([x], X)

        self.nfeatures = m
        
        self.W = np.abs(avg * halfnorm.rvs(size=(self.nfeatures, self.rank)) /
                        np.sqrt(self.rank))

        self.A = np.zeros((self.rank, self.rank))
        self.B = np.zeros((self.nfeatures, self.rank))
        return X

    def fit(self, X, batch_size=None):
        if self.nfeatures is None:
            X = self._setup(X)

        num = None
        prod = np.outer
        if batch_size is not None:
            prod = np.dot
            length = X.shape[0]
            num = max(length // batch_size, 1)
            X = np.array_split(X, num, axis=0)
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)
        r, h = self.r, self.h
        for v in progressbar(X, leave=False, total=num, disable=num==1):
            h, r = _solveproj(v, self.W, self.lambda1, self.kappa, 
                              r=r, h=h)
            self.H.append(h)
            if self.R is not None:
                self.R.append(r)

            # Solve for W
            self.A += prod(h, h.T)
            self.B += prod((v.T - r), h.T)
            eta = self.kappa / np.linalg.norm(self.A, 'fro')
            self.W -= eta * (np.dot(self.W, self.A) - self.B)
            np.maximum(self.W, 0.0, out=self.W)                 
            self.W /= max(np.linalg.norm(self.W, 'fro'), 1.0)  
        self.r = r
        self.h = h

    def project(self, X, return_R=False):
        # could be called project..?
        H = []
        if return_R:
            R = []

        num = None
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)
        h, r = None, None
        for v in progressbar(X, leave=False, total=num):
            h, r = _solveproj(v, self.W, self.lambda1, self.kappa, h=h, r=r)
            H.append(h.copy())
            if return_R:
                R.append(r.copy())

        H = np.stack(H, axis=-1)
        if return_R:
            return H, np.stack(R, axis=-1)
        else:
            return H

    def finish(self):
        if len(self.H) > 0:
            if len(self.H[0].shape) == 1:
                H = np.stack(self.H, axis=-1)
            else:
                H = np.concatenate(self.H, axis=1)
            return self.W, H
        else:
            return self.W, 0


def onmf(X, rank, lambda1=1, kappa=1, store_r=False, refine=False):
    _onmf = ONMF(rank, lambda1=lambda1, kappa=kappa, store_r=store_r)
    _onmf.fit(X)
    if refine:
        W = _onmf.W
        H = _onmf.project(X)
    else:
        W, H = _nmf.finish()
    return W, H
