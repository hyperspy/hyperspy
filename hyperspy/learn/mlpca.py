# -*- coding: utf-8 -*-
# This file is a transcription of a MATLAB code obtained from the
# following research paper: Darren T. Andrews and Peter D. Wentzell,
# “Applications of maximum likelihood principal component analysis:
# incomplete data sets and calibration transfer,”
# Analytica Chimica Acta 350, no. 3 (September 19, 1997): 341-352.
#
# Copyright 1997 Darren T. Andrews and Peter D. Wentzell
# Copyright 2007-2016 The HyperSpy developers
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
import scipy.linalg

from hyperspy.misc.machine_learning.import_sklearn import (
    fast_svd, sklearn_installed)

_logger = logging.getLogger(__name__)


def mlpca(X, varX, p, convlim=1E-10, maxiter=50000, fast=False):
    """
    This function performs MLPCA with missing
    data.

    Parameters
    ----------
    X: numpy array
        is the mxn matrix of observations.
    stdX: numpy array
        is the mxn matrix of standard deviations
        associated with X (zeros for missing
        measurements).
    p: int
        The model dimensionality.

    Returns
    -------
    U,S,V: numpy array
        are the pseudo-svd parameters.
    Sobj: numpy array
        is the value of the objective function.
    ErrFlag: {0, 1}
        indicates exit conditions:
        0 = nkmal termination
        1 = max iterations exceeded.

    """
    if fast is True and sklearn_installed is True:
        def svd(X):
            return fast_svd(X, p)
    else:
        def svd(X):
            return scipy.linalg.svd(X, full_matrices=False)
    XX = X
#    varX = stdX**2
    n = XX.shape[1]
    _logger.info("Performing maximum likelihood principal components analysis")
    # Generate initial estimates
    _logger.info("Generating initial estimates")
    CV = np.cov(X)
    U, S, Vh = svd(CV)
    U0 = U

    # Loop for alternating least squares
    _logger.info("Optimization iteration loop")
    count = 0
    Sold = 0
    ErrFlag = -1
    while ErrFlag < 0:
        count += 1
        Sobj = 0
        MLX = np.zeros(XX.shape)
        for i in range(n):
            Q = np.diag((1 / (varX[:, i])).squeeze())
            U0m = np.matrix(U0)
            F = np.linalg.inv((U0m.T * Q * U0m))
            MLX[:, i] = np.array(U0m * F * U0m.T * Q *
                                 (np.matrix(XX[:, i])).T).squeeze()
            dx = np.matrix((XX[:, i] - MLX[:, i]).squeeze())
            Sobj += float(dx * Q * dx.T)
        if (count % 2) == 1:
            _logger.info("Iteration : %s" % (count / 2))
            if (abs(Sold - Sobj) / Sobj) < convlim:
                ErrFlag = 1
            _logger.info("(abs(Sold - Sobj) / Sobj) = %s" %
                         (abs(Sold - Sobj) / Sobj))
            if count > maxiter:
                ErrFlag = 1

        if ErrFlag < 0:
            Sold = Sobj
            U, S, Vh = svd(MLX)
            V = Vh.T
            XX = XX.T
            varX = varX.T
            n = XX.shape[1]
            U0 = V[:]
    # Finished

    U, S, Vh = svd(MLX)
    V = Vh.T
    return U, S, V, Sobj, ErrFlag
