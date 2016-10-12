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

from hyperspy.external.progressbar import progressbar

_logger = logging.getLogger(__name__)

def _thresh(R, lambda1, M):
    res = np.abs(R) - lambda1
    res[res<0] *= 0
    res[res>M] = M
    return np.sign(R) * res

def _mrdivide(B, A):
    """like in Matlab! (solves xB = A)
    """
    if isinstance(B, np.ndarray):
        if len(B.shape) == 2 and B.shape[0] == B.shape[1]:
            # square array
            return np.linalg.solve(A.T, B.T).T
        else:
            return np.linalg.lstsq(A.T, B.T)[0].T
    else:
        return B / A

def _project(W):
    newW = W.copy()
    newW[W<0] = 0
    
    sumsq = np.sqrt(np.sum(W**2, axis=0))
    sumsq[sumsq<1] = 1
    return _mrdivide(newW, np.diag(sumsq))

class OPGD:

    def __init__(self, rank, batch_size, lambda1=None, max_value=None,
                 store_r=False):
        self.iterating = None
        self.rank = rank
        self.batch_size = batch_size
        self.lambda1 = lambda1
        self.features = None
        # just random numbers for now
        if max_value is None:
            max_value = 0.7
        self.max_value = max_value
        self.maxItr1 = 1e2
        self.maxItr2 = 1e3
        self.eps1 = 1e-3
        self.eps2 = 1e-6
        self.stepMulp = 1.
        self.H = []
        if store_r:
            self.R = []
        else:
            self.R = None

    def _setup(self, X):
        # figure out how many features, F. K is the rank
        if isinstance(X, np.ndarray):
            _, F = X.shape
            self.iterating = False
        else:
            x = next(X)
            F = len(x)
            X = chain([x], X)
            self.iterating = True
        self.features = F
        if self.lambda1 is None:
            self.lambda1 = 1/np.sqrt(F)
        self.h = np.random.rand(self.rank, self.batch_size)
        self.r = np.random.rand(self.features, self.batch_size)

        self.A = np.zeros((self.rank, self.rank))
        self.B = np.zeros((self.features, self.rank))
        self.W = np.random.rand(self.features, self.rank)
        return X

    def fit(self, values, iterating=None):
        if self.features is None:
            values = self._setup(values)

        if iterating is None:
            iterating = self.iterating
        else:
            self.iterating = iterating

        num = None
        if isinstance(values, np.ndarray):
            # make an iterator anyway
            num = values.shape[0]
            values = iter(values)
        this_batch = []
        # when we run out of samples for the full batch, re-use some
        pbar = progressbar(leave=False, total=num,
                           disable=(self.batch_size==num))
        for val in values:
            this_batch.append(val)
            if len(this_batch) == self.batch_size:
                self._fit_batch(np.stack(this_batch, axis=-1))
                this_batch = []
                pbar.update(self.batch_size)
        left_samples = len(this_batch)
        if left_samples > 0:
            data = np.concatenate(
                [self._last_batch[left_samples-self.batch_size:,:],
                 np.stack(this_batch, axis=-1)], axis=0)
            self._fit_batch(data)
            pbar.update(left_samples)

    def _fit_batch(self, values):
        self._last_batch = values
        self._update_hr(values)
        # store the values to have a "history"
        self.H.append(self.h)
        if self.R is not None:
            self.R.append(self.r)
        self.A += self.h.dot(self.h.T)
        self.B += (values-self.r).dot(self.h.T)
        self._update_W()


    def _update_hr(self, values):
        _logger.debug("start update HR")
        n = 0
        lasttwo = np.zeros(2)
        L = np.linalg.norm(self.W,2)**2
        eta = 1./L*self.stepMulp

        while n<=2 or (np.abs((lasttwo[1] - lasttwo[0])/lasttwo[0]) >
                       self.eps1 and n<self.maxItr1):
            _logger.debug('n = {}'.format(n))
            self.h -= eta*self.W.T.dot(self.W.dot(self.h) + self.r - values)
            self.h[self.h<0] *= 0
            self.r = _thresh(values - self.W.dot(self.h), self.lambda1,
                             self.max_value)
            n += 1
            lasttwo[0] = lasttwo[1]
            lasttwo[1] = 0.5 * np.linalg.norm(
                values - self.W.dot(self.h) - self.r, 'fro')**2 + \
                    self.lambda1*np.sum(np.abs(self.r))

        _logger.debug("end update HR")

    def _update_W(self):
        _logger.debug("start update W")

        n = 0
        lasttwo = np.zeros(2)
        L = np.linalg.norm(self.A,'fro');
        eta = 1./L*self.stepMulp
        A = self.A
        B = self.B

        while n<=2 or (np.abs((lasttwo[1] - lasttwo[0])/lasttwo[0]) >
                       self.eps2 and n<self.maxItr2):
            _logger.debug('n = {}'.format(n))
            self.W = _project(self.W - eta*(self.W.dot(A) - B))
            n += 1
            lasttwo[0] = lasttwo[1]
            lasttwo[1] = 0.5 * np.trace(self.W.T.dot(self.W).dot(A)) - \
                    np.trace(self.W.T.dot(B))
        _logger.debug("end update W")
