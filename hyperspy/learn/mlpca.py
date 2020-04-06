# -*- coding: utf-8 -*-
# This file is a transcription of a MATLAB code obtained from the
# following research paper: Darren T. Andrews and Peter D. Wentzell,
# “Applications of maximum likelihood principal component analysis:
# incomplete data sets and calibration transfer,”
# Analytica Chimica Acta 350, no. 3 (September 19, 1997): 341-352.
#
# Copyright 1997 Darren T. Andrews and Peter D. Wentzell
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

import numpy as np
import scipy.linalg

from hyperspy.misc.machine_learning.import_sklearn import fast_svd, sklearn_installed

_logger = logging.getLogger(__name__)


def mlpca(X, varX, output_dimension, tol=1e-10, max_iter=50000, fast=False):
    """Performs maximum likelihood PCA with missing data.

    Parameters
    ----------
    X : numpy array, shape (m, n)
        Matrix of observations.
    stdX : numpy array
        Matrix of variances associated with X
        (zeros for missing measurements).
    output_dimension : int
        The model dimensionality.
    tol : float

    max_iter : int

    fast : bool, default False
        Whether to use randomized SVD from sklearn to estimate
        a limited number of components given by output_dimension.

    Returns
    -------
    U, S, V: numpy array
        The pseudo-SVD parameters.
    s_obj : float
        Value of the objective function.
    """
    if fast is True and sklearn_installed is True:

        def svd(X):
            return fast_svd(X, output_dimension)

    else:

        def svd(X):
            return scipy.linalg.svd(X, full_matrices=False)

    m, n = X.shape
    inv_v = 1.0 / varX
    _logger.info("Performing maximum likelihood principal components analysis")

    # Generate initial estimates
    _logger.info("Generating initial estimates")
    U, _, _ = svd(np.cov(X))
    U = U[:, :output_dimension]
    s_old = 0.0

    # Loop for alternating least squares
    _logger.info("Optimization iteration loop")
    for itr in range(max_iter):
        s_obj = 0.0
        F = np.empty(X.shape)
        M = np.zeros(X.shape)

        for i in range(n):
            Q = np.diag(inv_v[:, i])
            F = (U.T @ Q) @ U
            F = np.linalg.inv(F)
            M[:, i] = np.linalg.multi_dot([U, F, U.T, Q, X[:, i].T])
            dx = X[:, i] - M[:, i]
            s_obj += dx @ Q @ dx.T

        # Every second iteration, check the stop criterion
        if itr % 2 == 0:
            stop_criterion = np.abs(s_old - s_obj) / s_obj
            _logger.info(
                "Iteration: {}, convergence: {}".format(itr // 2, stop_criterion)
            )

            if stop_criterion < tol:
                break

        # Transpose for next iteration
        s_old = s_obj
        _, _, V = svd(M)
        X = X.T
        inv_v = inv_v.T
        m, n = X.shape
        U = V[:output_dimension].T

    U, S, V = svd(M)
    V = V.T
    return U, S, V, s_obj
