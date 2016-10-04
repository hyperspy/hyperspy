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

from hyperspy.external.progressbar import progressbar

_logger = logging.getLogger(__name__)

def _thresh(R, lambda1, M):
    res = np.abs(R) - lambda1
    return np.sign(R) * np.min(np.max(res, 0), M)

def _mrdivide(A, B):
    """like in Matlab! (solves xA = B)
    """
    if isinstance(A, np.ndarray):
        if len(set(A.shape)) == 1:
            # square array
            return np.linalg.solve(A.T, B.T).T
        else:
            return np.linalg.lstsq(A.T, B.T)[0].T
    else:
        return A / B

def _project(W):
    return _mrdivide(np.max(W, 0),
                     np.diag(np.max(np.sqrt(W**2, axis=0),
                                    axis=0)))

class OPGD:

    def __init__(self, rank, batch_size, lambda1=None):
        self.rank = rank
        self.batch_size = batch_size, 
        self.lambda1 = lambda1 # TODO: Can be None, change once data comes

