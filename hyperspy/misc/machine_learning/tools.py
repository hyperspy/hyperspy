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

import numpy as np


def amari(C, A):
    """Amari test for ICA
    Adapted from the MILCA package http://www.klab.caltech.edu/~kraskov/MILCA/

    Parameters
    ----------
    C : numpy array
    A : numpy array
    """
    b, a = C.shape

    dummy = np.dot(np.linalg.pinv(A), C)
    dummy = np.sum(_ntu(np.abs(dummy)), 0) - 1

    dummy2 = np.dot(np.linalg.pinv(C), A)
    dummy2 = np.sum(_ntu(np.abs(dummy2)), 0) - 1

    out = (np.sum(dummy) + np.sum(dummy2)) / (2 * a * (a - 1))
    return out


def _ntu(C):
    m, n = C.shape
    CN = C.copy() * 0
    for t in range(n):
        CN[:, t] = C[:, t] / np.max(np.abs(C[:, t]))
    return CN
