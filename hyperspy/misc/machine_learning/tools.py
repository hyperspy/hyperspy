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

import numpy as np


def amari(W, A):
    """Calculate the Amari distance between two non-singular matrices.

    Convenient for checking convergence in ICA algorithms
    (See [Moreau1998]_ and [Bach2002]_).

    Parameters
    ----------
    W, A : array-like
        The two matrices to measure.

    Returns
    -------
    float
        Amari distance between W and A.

    References
    ----------
    .. [Moreau1998] E. Moreau and O. Macchi, "Self-adaptive source separation.
        ii. comparison of the direct, feedback, and mixed linear network",
        IEEE Trans. on Signal Processing, vol. 46(1), pp. 39-50, 1998.
    .. [Bach2002] F. Bach and M. Jordan, "Kernel independent component analysis",
        Journal of Machine Learning Research, vol. 3, pp. 1-48, 2002.

    """
    P = W @ A
    m, _ = P.shape

    P_sq = P**2

    P_sq_sum_0 = np.sum(P_sq, axis=0)
    P_sq_max_0 = np.max(P_sq, axis=0)

    P_sq_sum_1 = np.sum(P_sq, axis=1)
    P_sq_max_1 = np.max(P_sq, axis=1)

    P_sr_0 = np.sum(P_sq_sum_0 / P_sq_max_0 - 1)
    P_sr_1 = np.sum(P_sq_sum_1 / P_sq_max_1 - 1)

    return (P_sr_0 + P_sr_1) / (2 * m)
